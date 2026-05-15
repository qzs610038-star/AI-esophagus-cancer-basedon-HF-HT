(& "D:\miniconda\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate "d:\conda_envs\pfmval_py310"
$env:HF_ENDPOINT = "https://hf-mirror.com"
python egnv1/train.py `
  --dataset_name LMZ12939 `
  --train_patches_dir "data_new_3ST\patch_noov_spilt\LMZ12939_noov_split\train_patches" `
  --val_patches_dir "data_new_3ST\patch_noov_spilt\LMZ12939_noov_split\val_patches" `
  --labels_csv "data_new_3ST\ssGSEA_zscore\LMZ12939_ssGSEA_zscore.csv" `
  --num_epochs 150 `
  --lr 1e-5 `
  --batch_size 16 `
  --hidden_dim 1024 `
  --graph_layers 2 `
  --k_neighbors 10 `
  --dropout 0.5 `
  --freeze_layers 4 `
  --early_stop_patience 15 `
  --n_targets 30 `
  --graph_type knn `
  --weight_decay 1e-4
