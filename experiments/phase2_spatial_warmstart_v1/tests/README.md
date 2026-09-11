本目录是零真实训练的合成验收套件。运行：

```powershell
python -m pytest .\tests -q -p no:cacheprovider
```

覆盖数据/身份合同、严格H/C加载、B零初始化、患者与划分内图、双PCC、三臂冻结与更新、配对dropout、step0选择、完整断点恢复、模型bundle、无标签推理、固定β=1平滑及batch目录启动器。临时文件由pytest管理，不复制到服务器。
