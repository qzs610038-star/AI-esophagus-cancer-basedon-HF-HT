import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPoolMIL(nn.Module):
    def __init__(
        self,
        pooling="mean",
        size_arg="small",
        dropout=0.,
        n_classes=2,
        embed_dim=1024,
        fusion_mode="none",
        path_dim=1536,
        gene_dim=30,
        path_proj_dim=512,
        gene_proj_dim=128,
        cross_attn_heads=4,
        cross_attn_direction="path_to_gene",
        cross_attn_output="fused",
        fusion_projector="linear",
    ):
        super().__init__()
        if pooling not in {"mean", "max"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.pooling = pooling
        self.fusion_mode = "none" if fusion_mode is None else fusion_mode
        self.path_dim = path_dim
        self.gene_dim = gene_dim
        self.cross_attn_direction = cross_attn_direction
        self.cross_attn_output = cross_attn_output
        self.size_dict = {"small": [embed_dim, 512], "big": [embed_dim, 512]}
        size = self.size_dict[size_arg]

        if self.fusion_mode == "none":
            self.fc = nn.Sequential(
                nn.Linear(size[0], size[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            pooled_dim = size[1]
        elif self.fusion_mode == "cross_attention":
            if path_dim + gene_dim != embed_dim:
                raise ValueError(
                    f"Cross-attention fusion expects path_dim + gene_dim == embed_dim, "
                    f"got {path_dim} + {gene_dim} != {embed_dim}"
                )
            if cross_attn_direction not in {"path_to_gene", "gene_to_path"}:
                raise ValueError(f"Unsupported cross_attn_direction: {cross_attn_direction}")
            if cross_attn_output not in {"fused", "attended", "fused_plus_path_raw", "fused_plus_query"}:
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

            pooled_dim = path_proj_dim
            if cross_attn_output == "fused_plus_path_raw":
                pooled_dim += path_dim
            elif cross_attn_output == "fused_plus_query":
                pooled_dim += path_proj_dim
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=path_proj_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(path_proj_dim)
            self.cross_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unsupported fusion_mode for fixed pooling baseline: {self.fusion_mode}")

        self.classifier = nn.Linear(pooled_dim, n_classes)

    def split_modalities(self, h):
        if h.size(1) != self.path_dim + self.gene_dim:
            raise ValueError(f"Expected input dim {self.path_dim + self.gene_dim}, got {h.size(1)}")
        path_h = h[:, :self.path_dim]
        gene_h = h[:, self.path_dim:self.path_dim + self.gene_dim]
        return path_h, gene_h

    def fuse_features(self, h):
        if self.fusion_mode == "none":
            return self.fc(h)

        path_raw, gene_raw = self.split_modalities(h)
        path_h = self.path_projector(path_raw)
        gene_h = self.gene_projector(gene_raw)
        if self.cross_attn_direction == "path_to_gene":
            query_h = path_h
            context_h = gene_h
        else:
            query_h = gene_h
            context_h = path_h

        cross_h, _ = self.cross_attention(
            query=query_h.unsqueeze(0),
            key=context_h.unsqueeze(0),
            value=context_h.unsqueeze(0),
            need_weights=False,
        )
        cross_h = cross_h.squeeze(0)
        fused_h = self.cross_norm(query_h + self.cross_dropout(cross_h))
        if self.cross_attn_output == "fused":
            return fused_h
        if self.cross_attn_output == "attended":
            return cross_h
        if self.cross_attn_output == "fused_plus_path_raw":
            return torch.cat([cross_h, path_raw], dim=1)
        if self.cross_attn_output == "fused_plus_query":
            return torch.cat([cross_h, query_h], dim=1)
        raise ValueError(f"Unsupported cross_attn_output: {self.cross_attn_output}")

    def forward(self, h, return_features=False):
        if h.ndim != 2:
            raise ValueError(f"Expected 2D bag tensor, got shape {tuple(h.shape)}")

        h = self.fuse_features(h)
        if self.pooling == "mean":
            M = h.mean(dim=0, keepdim=True)
            A_raw = h.new_zeros(1, h.size(0))
        else:
            M = h.max(dim=0, keepdim=True).values
            A_raw = h.norm(dim=1, keepdim=True).transpose(1, 0)

        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
