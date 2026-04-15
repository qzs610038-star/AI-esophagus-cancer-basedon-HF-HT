
# PFMval

## 数据
- 使用 `split.py` 按照坐标距离约束将HYZ15040数据集划分为训练集（9806个patch）和测试集（772个patch），确保它们之间的空间距离不小于设定阈值（350px），避免重叠。分别在 `HYZ15040\train_patches` 和 `val_patches` 里面。

- 使用 `zscore.py` 做z-score标准化，基因集分数原文件： `HYZ15040_ssGSEA_scores.csv`，z-score标准化后为： `HYZ15040_ssGSEA_scores_zscore.csv`。


## UNI2-h
https://huggingface.co/MahmoodLab/UNI2-h

输出1536维特征

获取token：https://lxltx.blog.csdn.net/article/details/146328238

### 0 环境
```bash
conda create -n pfmval python=3.10
conda activate pfmval
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
```bash
pip install pandas scikit-learn pillow
pip install numpy==1.26.4
pip install huggingface_hub
pip install timm>=0.9.8
```

### 1 训练
修改 `train.py` 里的token和相关配置参数，运行。
基础模型会下载到 `.cache\huggingface\hub\models--MahmoodLab--UNI2-h`
最优模型和训练过程日志csv在 `.\checkpoints`
特征缓存在 `.\uni2h_cache`

### 2 推理
修改 `infer.py` 里的token和相关配置参数，运行。
预测结果和指标csv在 `.\res`





