
# CLAM

## 1 环境
```
conda create -n clam python=3.10
conda activate clam
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c 
conda install -c conda-forge openslide pip
pip install timm==0.9.8 h5py pandas PyYAML opencv-python matplotlib
pip install scikit-learn scipy tqdm openslide-python tensorboardX
pip install git+https://github.com/oval-group/smooth-topk.git
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
pip install future
```

## 2 数据集划分

保证同一患者的所有切片只会在 train / val / test 其中一个集合里，生成 5 个随机划分。按 label 分层抽样，保证每个 split 中阳性/阴性比例相对均衡。输出在 `.\splits\`
已经生成，直接用就行。
pcr：
```bat
python create_splits_seq.py --task ESCC_pCR --seed 42 --k 5 --val_frac 0.1 --test_frac 0.1
```

## 3 特征处理
### 3.1 图像特征
把UNI2-h特征转换为：
```
ESCC_uni2h_features/
├─ pt_files/
│  ├─ 23G068252-002-1.pt      # N x 1536 tensor，训练用
│  └─ ...
├─ h5_files/
│  ├─ 23G068252-002-1.h5      # features + coords，后续热图可用
│  └─ ...
└─ conversion_summary.csv
```
```bat
python convert_patch_features_to_clam.py ^
  --input_root "D:\PycharmProjects\AIPath-data\patch_329_uni2h" ^
  --output_dir "D:\PycharmProjects\AIPath-data\ESCC_uni2h_features" ^
  --expected_csv "dataset_csv\ESCC_clam_metadata.csv"
```
在服务器 `D:\AIPatho\yzq\ESCC_uni2h_features`

### 3.2 基因特征
把基因集分数csv转换为特征，顺序和UNI2-h特征保持一致：
```bat
python convert_patch_scores_to_clam.py ^
  --input_root 基因集分数csv文件夹路径 ^
  --reference_feature_dir D:\PycharmProjects\AIPath-data\ESCC_uni2h_features ^
  --output_dir D:\PycharmProjects\AIPath-data\ESCC_gene_scores_features
```
基因集分数csv在服务器 `D:\AIPatho\yzq\patch_329_scores_mpp2_20260711`
生成结果在服务器 `D:\AIPatho\yzq\ESCC_mpp2_z_scores_20260711`

### 3.3 特征拼接
把这两种特征直接拼接：
```bat
python concat_clam_features.py ^
  --feature_dir_a "D:\PycharmProjects\AIPath-data\ESCC_uni2h_features" ^
  --feature_dir_b "D:\PycharmProjects\AIPath-data\ESCC_gene_scores_features" ^
  --output_dir "D:\PycharmProjects\AIPath-data\ESCC_uni2h_gene_features"
```

## 4 训练
输出在 `.\results\实验名称_s1`

### 4.1 仅病理图像
```
python main.py ^
  --drop_out 0.25 ^
  --early_stopping ^
  --lr 1e-4 ^
  --k 5 ^
  --exp_code 实验名称 ^
  --weighted_sample ^
  --bag_loss ce ^
  --inst_loss ce ^
  --task ESCC_pCR ^
  --model_type clam_sb ^
  --log_data ^
  --data_root_dir "D:\PycharmProjects\AIPath-data" ^
  --feature_dir ESCC_uni2h_features ^
  --embed_dim 1536
```

### 4.2 特征直接拼接 [patho,gene]
```
python main.py ^
  --drop_out 0.25 ^
  --early_stopping ^
  --lr 1e-4 ^
  --k 5 ^
  --exp_code 实验名称 ^
  --weighted_sample ^
  --bag_loss ce ^
  --inst_loss ce ^
  --task ESCC_pCR ^
  --model_type clam_sb ^
  --log_data ^
  --data_root_dir "D:\PycharmProjects\AIPath-data" ^
  --feature_dir ESCC_uni2h_gene_features ^
  --embed_dim 1566
```

### 4.3 特征融合

**Cross-Modal Attention Fusion 交叉注意力：**
Q=gene,K/V=patho,[Q+C]：
```bat
python main.py ^
  --drop_out 0.25 ^
  --early_stopping ^
  --lr 1e-4 ^
  --k 5 ^
  --exp_code 实验名称 ^
  --weighted_sample ^
  --bag_loss ce ^
  --inst_loss ce ^
  --task ESCC_pCR ^
  --model_type clam_sb ^
  --log_data ^
  --data_root_dir "D:\PycharmProjects\AIPath-data" ^
  --feature_dir ESCC_uni2h_gene_features ^
  --embed_dim 1566 ^
  --fusion_mode cross_attention ^
  --cross_attn_direction gene_to_path ^
  --path_dim 1536 ^
  --gene_dim 30 ^
  --path_proj_dim 512 ^
  --cross_attn_heads 4
```

AF：
根据图像和基因动态生成权重，自适应加权融合。每个 patch 自己学习图像和基因各占多少权重。
```bat
python main.py ^
  --drop_out 0.25 ^
  --early_stopping ^
  --lr 1e-4 ^
  --k 5 ^
  --exp_code 实验名称 ^
  --weighted_sample ^
  --bag_loss ce ^
  --inst_loss ce ^
  --task ESCC_pCR ^
  --model_type clam_sb ^
  --log_data ^
  --data_root_dir "D:\PycharmProjects\AIPath-data" ^
  --feature_dir ESCC_uni2h_gene_features ^
  --embed_dim 1566 ^
  --fusion_mode adaptive_fusion ^
  --path_dim 1536 ^
  --gene_dim 30 ^
  --fusion_output_dim 512 ^
  --fusion_hidden_dim 256
```


# Baseline
其他都和CLAM一样，只是把CLAM的注意力聚合改为mean/max聚合
## mean pooling
```bat
python main.py ^
  --drop_out 0.25 ^
  --early_stopping ^
  --lr 1e-4 ^
  --k 5 ^
  --exp_code 实验名称 ^
  --results_dir baseline_results ^
  --weighted_sample ^
  --bag_loss ce ^
  --task ESCC_pCR ^
  --model_type mean_pool ^
  --log_data ^
  --data_root_dir "D:\PycharmProjects\AIPath-data" ^
  --feature_dir ESCC_uni2h_gene_features ^
  --embed_dim 1566 ^
  --fusion_mode cross_attention ^
  --cross_attn_direction gene_to_path ^
  --path_dim 1536 ^
  --gene_dim 30 ^
  --path_proj_dim 512 ^
  --cross_attn_heads 4
```

## max pooling
换成：
```bat
--model_type max_pool
```