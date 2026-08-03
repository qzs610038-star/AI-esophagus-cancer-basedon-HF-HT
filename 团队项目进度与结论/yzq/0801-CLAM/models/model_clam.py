import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class DualBranchProjector(nn.Module):
    def __init__(self, path_dim=1536, gene_dim=30, path_proj_dim=512, gene_proj_dim=128, dropout=0.,
                 fusion_projector="linear"):
        super().__init__()
        self.path_dim = path_dim
        self.gene_dim = gene_dim
        self.input_dim = path_dim + gene_dim
        if fusion_projector == "identity":
            self.output_dim = path_dim + gene_dim
            self.path_branch = nn.Identity()
            self.gene_branch = nn.Identity()
        elif fusion_projector == "linear":
            self.output_dim = path_proj_dim + gene_proj_dim
            self.path_branch = nn.Sequential(
                nn.Linear(path_dim, path_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gene_branch = nn.Sequential(
                nn.Linear(gene_dim, gene_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")

    def forward(self, h):
        if h.size(1) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {h.size(1)}")
        path_h = h[:, :self.path_dim]
        gene_h = h[:, self.path_dim:self.path_dim + self.gene_dim]
        return torch.cat([self.path_branch(path_h), self.gene_branch(gene_h)], dim=1)


class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        squeeze_batch = False
        if queries.dim() == 2:
            queries = queries.unsqueeze(0)
            squeeze_batch = True
        elif queries.dim() != 3:
            raise ValueError(f"ExternalAttention expects [N, D] or [B, N, D], got {tuple(queries.shape)}")

        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / (torch.sum(attn, dim=2, keepdim=True) + 1e-6)
        out = self.mv(attn)

        if squeeze_batch:
            out = out.squeeze(0)
        return out


class AdaptiveFusion(nn.Module):
    def __init__(self, img_dim=1536, gene_dim=30, output_dim=512, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear(img_dim + gene_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.img_proj = nn.Linear(img_dim, output_dim)
        self.gene_proj = nn.Linear(gene_dim, output_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, img_feat, gene_feat):
        weights = self.weight_generator(torch.cat([img_feat, gene_feat], dim=1))
        weights = F.softmax(weights, dim=1)
        img_proj = self.img_proj(img_feat)
        gene_proj = self.gene_proj(gene_feat)
        fused = weights[:, 0:1] * img_proj + weights[:, 1:2] * gene_proj
        return self.fusion_proj(fused)


class LowRankFusion(nn.Module):
    def __init__(self, img_dim=1536, gene_dim=30, output_dim=512, rank=64, dropout=0.25):
        super().__init__()
        self.U = nn.Parameter(torch.randn(img_dim + gene_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(output_dim, rank) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, img_feat, gene_feat):
        concat = torch.cat([img_feat, gene_feat], dim=1)
        fused = torch.matmul(torch.matmul(concat, self.U), self.V.t()) + self.bias
        return self.activation(fused)

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
        fusion_mode="none", path_dim=1536, gene_dim=30, path_proj_dim=512, gene_proj_dim=128,
        cross_attn_heads=4, cross_attn_direction="path_to_gene", cross_attn_output="fused",
        external_attn_size=64, fusion_output_dim=512, fusion_hidden_dim=256, fusion_rank=64,
        fusion_projector="linear"):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fusion_mode = "none" if fusion_mode is None else fusion_mode
        self.fusion_mode = fusion_mode
        self.path_dim = path_dim
        self.gene_dim = gene_dim
        self.fusion_projector = fusion_projector
        self.cross_attn_direction = cross_attn_direction
        self.cross_attn_output = cross_attn_output
        attention_cls = Attn_Net_Gated if gate else Attn_Net

        if fusion_mode == "none":
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            attention_input_dim = size[1]
            if gate:
                attention_net = Attn_Net_Gated(L = attention_input_dim, D = size[2], dropout = dropout, n_classes = 1)
            else:
                attention_net = Attn_Net(L = attention_input_dim, D = size[2], dropout = dropout, n_classes = 1)
            fc.append(attention_net)
            self.attention_net = nn.Sequential(*fc)
        elif fusion_mode == "dual_branch":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Dual-branch fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            attention_input_dim = path_proj_dim + gene_proj_dim
            if fusion_projector == "identity":
                attention_input_dim = path_dim + gene_dim
            elif fusion_projector != "linear":
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            fc = [
                DualBranchProjector(
                    path_dim=path_dim,
                    gene_dim=gene_dim,
                    path_proj_dim=path_proj_dim,
                    gene_proj_dim=gene_proj_dim,
                    dropout=dropout,
                    fusion_projector=fusion_projector,
                )
            ]
            if gate:
                attention_net = Attn_Net_Gated(L = attention_input_dim, D = size[2], dropout = dropout, n_classes = 1)
            else:
                attention_net = Attn_Net(L = attention_input_dim, D = size[2], dropout = dropout, n_classes = 1)
            fc.append(attention_net)
            self.attention_net = nn.Sequential(*fc)
        elif fusion_mode == "external_attention":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"External-attention fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            if fusion_projector == "identity":
                attention_input_dim = path_dim + gene_dim
                self.external_projector = nn.Identity()
            elif fusion_projector == "linear":
                attention_input_dim = path_proj_dim + gene_proj_dim
                self.external_projector = DualBranchProjector(
                    path_dim=path_dim,
                    gene_dim=gene_dim,
                    path_proj_dim=path_proj_dim,
                    gene_proj_dim=gene_proj_dim,
                    dropout=dropout,
                    fusion_projector=fusion_projector,
                )
            else:
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            self.external_attention = ExternalAttention(d_model=attention_input_dim, S=external_attn_size)
            self.external_clam_attention = attention_cls(
                L=attention_input_dim,
                D=size[2],
                dropout=dropout,
                n_classes=1,
            )
        elif fusion_mode == "adaptive_fusion":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Adaptive fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            attention_input_dim = fusion_output_dim
            self.feature_fusion = AdaptiveFusion(
                img_dim=path_dim,
                gene_dim=gene_dim,
                output_dim=fusion_output_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
            )
            self.fusion_clam_attention = attention_cls(
                L=attention_input_dim,
                D=size[2],
                dropout=dropout,
                n_classes=1,
            )
        elif fusion_mode == "low_rank_fusion":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Low-rank fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            attention_input_dim = fusion_output_dim
            self.feature_fusion = LowRankFusion(
                img_dim=path_dim,
                gene_dim=gene_dim,
                output_dim=fusion_output_dim,
                rank=fusion_rank,
                dropout=dropout,
            )
            self.fusion_clam_attention = attention_cls(
                L=attention_input_dim,
                D=size[2],
                dropout=dropout,
                n_classes=1,
            )
        elif fusion_mode in ["dual_attention", "gated_modality"]:
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"{fusion_mode} expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            if fusion_projector == "identity":
                path_proj_dim = path_dim
                gene_proj_dim = gene_dim
                self.path_projector = nn.Identity()
                self.gene_projector = nn.Identity()
            elif fusion_projector == "linear":
                self.path_projector = nn.Sequential(
                    nn.Linear(path_dim, path_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.gene_projector = nn.Sequential(
                    nn.Linear(gene_dim, gene_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            attention_input_dim = path_proj_dim + gene_proj_dim
            self.path_attention = attention_cls(L=path_proj_dim, D=size[2], dropout=dropout, n_classes=1)
            self.gene_attention = attention_cls(L=gene_proj_dim, D=size[2], dropout=dropout, n_classes=1)
            if fusion_mode == "gated_modality":
                self.modality_gate = nn.Sequential(
                    nn.Linear(attention_input_dim, max(1, attention_input_dim // 2)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(1, attention_input_dim // 2), 1),
                    nn.Sigmoid(),
                )
        elif fusion_mode == "film":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"FiLM fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            if fusion_projector == "identity":
                path_proj_dim = path_dim
                self.path_projector = nn.Identity()
            elif fusion_projector == "linear":
                self.path_projector = nn.Sequential(
                    nn.Linear(path_dim, path_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            attention_input_dim = path_proj_dim
            self.gene_film = nn.Sequential(
                nn.Linear(gene_dim, gene_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gene_proj_dim, path_proj_dim * 2),
            )
            self.film_attention = attention_cls(L=path_proj_dim, D=size[2], dropout=dropout, n_classes=1)
        elif fusion_mode == "cross_attention":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Cross-attention fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            if cross_attn_direction not in ["path_to_gene", "gene_to_path"]:
                raise ValueError(f"Unsupported cross_attn_direction: {cross_attn_direction}")
            if cross_attn_output not in ["fused", "attended", "fused_plus_path_raw", "fused_plus_query"]:
                raise ValueError(f"Unsupported cross_attn_output: {cross_attn_output}")
            if fusion_projector == "identity":
                path_proj_dim = path_dim
                self.path_projector = nn.Identity()
                self.gene_projector = nn.Sequential(
                    nn.Linear(gene_dim, path_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            elif fusion_projector == "linear":
                self.path_projector = nn.Sequential(
                    nn.Linear(path_dim, path_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.gene_projector = nn.Sequential(
                    nn.Linear(gene_dim, path_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            attention_input_dim = path_proj_dim
            if cross_attn_output == "fused_plus_path_raw":
                attention_input_dim += path_dim
            elif cross_attn_output == "fused_plus_query":
                attention_input_dim += path_proj_dim
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=path_proj_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(path_proj_dim)
            self.cross_dropout = nn.Dropout(dropout)
            self.cross_clam_attention = attention_cls(L=attention_input_dim, D=size[2], dropout=dropout, n_classes=1)
        elif fusion_mode == "modality_attention":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Modality-attention fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            if fusion_projector == "identity":
                path_proj_dim = path_dim
                gene_proj_dim = gene_dim
                self.path_projector = nn.Identity()
                self.gene_projector = nn.Identity()
            elif fusion_projector == "linear":
                self.path_projector = nn.Sequential(
                    nn.Linear(path_dim, path_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.gene_projector = nn.Sequential(
                    nn.Linear(gene_dim, gene_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            attention_input_dim = path_proj_dim + gene_proj_dim
            self.path_attention = attention_cls(L=path_proj_dim, D=size[2], dropout=dropout, n_classes=1)
            self.gene_attention = attention_cls(L=gene_proj_dim, D=size[2], dropout=dropout, n_classes=1)
            self.attention_gate = nn.Sequential(
                nn.Linear(attention_input_dim, max(1, attention_input_dim // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(1, attention_input_dim // 2), 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.classifiers = nn.Linear(attention_input_dim, n_classes)
        instance_classifiers = [nn.Linear(attention_input_dim, 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def split_modalities(self, h):
        if h.size(1) != self.path_dim + self.gene_dim:
            raise ValueError(f"Expected input dim {self.path_dim + self.gene_dim}, got {h.size(1)}")
        path_h = h[:, :self.path_dim]
        gene_h = h[:, self.path_dim:self.path_dim + self.gene_dim]
        return path_h, gene_h

    def run_instance_eval(self, A, h, label):
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label == 1:
                instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else:
                if self.subtyping:
                    instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    continue
            total_inst_loss += instance_loss

        if self.subtyping:
            total_inst_loss /= len(self.instance_classifiers)

        return total_inst_loss, all_preds, all_targets

    def forward_multimodal(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        path_raw, gene_raw = self.split_modalities(h)

        if self.fusion_mode in ["dual_attention", "gated_modality"]:
            path_h = self.path_projector(path_raw)
            gene_h = self.gene_projector(gene_raw)
            A_path, path_h = self.path_attention(path_h)
            A_gene, gene_h = self.gene_attention(gene_h)
            A_path = torch.transpose(A_path, 1, 0)
            A_gene = torch.transpose(A_gene, 1, 0)
            A_raw = (A_path + A_gene) / 2.0
            if attention_only:
                return A_raw

            A_path = F.softmax(A_path, dim=1)
            A_gene = F.softmax(A_gene, dim=1)
            A = F.softmax(A_raw, dim=1)
            M_path = torch.mm(A_path, path_h)
            M_gene = torch.mm(A_gene, gene_h)

            if self.fusion_mode == "gated_modality":
                gate = self.modality_gate(torch.cat([M_path, M_gene], dim=1))
                M = torch.cat([gate * M_path, (1.0 - gate) * M_gene], dim=1)
            else:
                M = torch.cat([M_path, M_gene], dim=1)

            fused_h = torch.cat([path_h, gene_h], dim=1)

        elif self.fusion_mode == "film":
            path_h = self.path_projector(path_raw)
            gamma, beta = torch.chunk(self.gene_film(gene_raw), 2, dim=1)
            fused_h = path_h * (1.0 + torch.tanh(gamma)) + beta
            A_raw, fused_h = self.film_attention(fused_h)
            A_raw = torch.transpose(A_raw, 1, 0)
            if attention_only:
                return A_raw

            A = F.softmax(A_raw, dim=1)
            M = torch.mm(A, fused_h)

        elif self.fusion_mode == "external_attention":
            fused_h = self.external_projector(h)
            fused_h = self.external_attention(fused_h)
            A_raw, fused_h = self.external_clam_attention(fused_h)
            A_raw = torch.transpose(A_raw, 1, 0)
            if attention_only:
                return A_raw

            A = F.softmax(A_raw, dim=1)
            M = torch.mm(A, fused_h)

        elif self.fusion_mode in ["adaptive_fusion", "low_rank_fusion"]:
            fused_h = self.feature_fusion(path_raw, gene_raw)
            A_raw, fused_h = self.fusion_clam_attention(fused_h)
            A_raw = torch.transpose(A_raw, 1, 0)
            if attention_only:
                return A_raw

            A = F.softmax(A_raw, dim=1)
            M = torch.mm(A, fused_h)

        elif self.fusion_mode == "cross_attention":
            path_h = self.path_projector(path_raw)
            gene_h = self.gene_projector(gene_raw)
            if self.cross_attn_direction == "path_to_gene":
                query_h = path_h
                context_h = gene_h
            elif self.cross_attn_direction == "gene_to_path":
                query_h = gene_h
                context_h = path_h
            else:
                raise ValueError(f"Unsupported cross_attn_direction: {self.cross_attn_direction}")
            cross_h, _ = self.cross_attention(
                query=query_h.unsqueeze(0),
                key=context_h.unsqueeze(0),
                value=context_h.unsqueeze(0),
                need_weights=False,
            )
            cross_h = cross_h.squeeze(0)
            base_fused_h = self.cross_norm(query_h + self.cross_dropout(cross_h))
            if self.cross_attn_output == "fused":
                fused_h = base_fused_h
            elif self.cross_attn_output == "attended":
                fused_h = cross_h
            elif self.cross_attn_output == "fused_plus_path_raw":
                fused_h = torch.cat([cross_h, path_raw], dim=1)
            elif self.cross_attn_output == "fused_plus_query":
                fused_h = torch.cat([cross_h, query_h], dim=1)
            else:
                raise ValueError(f"Unsupported cross_attn_output: {self.cross_attn_output}")
            A_raw, fused_h = self.cross_clam_attention(fused_h)
            A_raw = torch.transpose(A_raw, 1, 0)
            if attention_only:
                return A_raw

            A = F.softmax(A_raw, dim=1)
            M = torch.mm(A, fused_h)

        elif self.fusion_mode == "modality_attention":
            path_h = self.path_projector(path_raw)
            gene_h = self.gene_projector(gene_raw)
            A_path, path_h = self.path_attention(path_h)
            A_gene, gene_h = self.gene_attention(gene_h)
            A_path = torch.transpose(A_path, 1, 0)
            A_gene = torch.transpose(A_gene, 1, 0)
            fused_h = torch.cat([path_h, gene_h], dim=1)

            raw_gate = self.attention_gate(fused_h.mean(dim=0, keepdim=True))
            A_raw = raw_gate * A_path + (1.0 - raw_gate) * A_gene
            if attention_only:
                return A_raw

            A = F.softmax(A_raw, dim=1)
            M = torch.mm(A, fused_h)

        else:
            raise ValueError(f"forward_multimodal called for unsupported fusion_mode: {self.fusion_mode}")

        if instance_eval:
            total_inst_loss, all_preds, all_targets = self.run_instance_eval(A, fused_h, label)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if self.fusion_mode in ["dual_attention", "gated_modality", "film", "external_attention", "adaptive_fusion", "low_rank_fusion", "cross_attention", "modality_attention"]:
            return self.forward_multimodal(
                h,
                label=label,
                instance_eval=instance_eval,
                return_features=return_features,
                attention_only=attention_only,
            )

        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss, all_preds, all_targets = self.run_instance_eval(A, h, label)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
        fusion_mode="none", path_dim=1536, gene_dim=30, path_proj_dim=512, gene_proj_dim=128,
        cross_attn_heads=4, cross_attn_direction="path_to_gene", cross_attn_output="fused",
        external_attn_size=64, fusion_output_dim=512, fusion_hidden_dim=256, fusion_rank=64,
        fusion_projector="linear"):
        nn.Module.__init__(self)
        if fusion_mode in ["dual_attention", "gated_modality", "film", "external_attention", "adaptive_fusion", "low_rank_fusion", "cross_attention", "modality_attention"]:
            raise NotImplementedError(f"{fusion_mode} is currently implemented for clam_sb only")
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fusion_mode = "none" if fusion_mode is None else fusion_mode
        if fusion_mode == "none":
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            attention_input_dim = size[1]
        elif fusion_mode == "dual_branch":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Dual-branch fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            attention_input_dim = path_proj_dim + gene_proj_dim
            if fusion_projector == "identity":
                attention_input_dim = path_dim + gene_dim
            elif fusion_projector != "linear":
                raise ValueError(f"Unsupported fusion_projector: {fusion_projector}")
            fc = [
                DualBranchProjector(
                    path_dim=path_dim,
                    gene_dim=gene_dim,
                    path_proj_dim=path_proj_dim,
                    gene_proj_dim=gene_proj_dim,
                    dropout=dropout,
                    fusion_projector=fusion_projector,
                )
            ]
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        if gate:
            attention_net = Attn_Net_Gated(L = attention_input_dim, D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = attention_input_dim, D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(attention_input_dim, 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(attention_input_dim, 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
