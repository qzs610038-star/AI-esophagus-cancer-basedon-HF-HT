# MPP1-5 统一标准重跑：服务器操作排错与踩坑汇总

本篇文档汇总了在 2026-07-08 进行的 **MPP1-MPP5 统一标准重跑** 阶段，在服务器端部署、冒烟测试、数据拉取及推送时所遇到的所有核心报错、技术踩坑及最终的修复与规避经验。本指南旨在为后续项目维护及在新机器上的快速部署提供参考。

---

## 坑一：PowerShell 临时环境变量声明语法陷阱

### 1. 现象与报错
在服务器的 PowerShell 终端运行带临时环境变量前缀的 python 脚本：
```powershell
PYTHONIOENCODING=utf-8 C:\Users\AIPatho1\pfmval_env\Scripts\python.exe extract_uni2h_mpp.py ...
```
报错：
```text
PYTHONIOENCODING=utf-8: The term 'PYTHONIOENCODING=utf-8' is not recognized as a name of a cmdlet, function, script file, or executable program.
```

### 2. 根因分析
Windows PowerShell 的语法体系不支持 Linux / macOS 的 `KEY=VALUE command` 的单行临时局部环境变量赋值语法。PowerShell 会错误地将 `PYTHONIOENCODING=utf-8` 识别为可执行命令名，进而抛出无法识别的错误。

### 3. 复用经验与最佳实践
在 Windows 服务器（PowerShell 窗口）上执行需要环境变量的命令时，**必须采用全局会话赋值法**。在打开 PowerShell 窗口的第一时间，先独立运行以下命令对该会话设置环境变量：
```powershell
$env:PYTHONIOENCODING="utf-8"
```
运行后，该终端窗口内的所有后续进程（包括后台进程）将自动继承此变量。

---

## 坑二：未定义命令行参数（Unrecognized Arguments）

### 1. 现象与报错
在启动冒烟测试和正式后台重跑时，命令行带入 `--model_dir` 参数：
```powershell
C:\Users\AIPatho1\pfmval_env\Scripts\python.exe train_mpp_uni2h_mlp.py --model_dir checkpoints/smoke_mpp1 ...
```
报错：
```text
train_mpp_uni2h_mlp.py: error: unrecognized arguments: --model_dir checkpoints/mpp_uni2h_mlp/smoke_mpp1
```

### 2. 根因分析
训练脚本 `train_mpp_uni2h_mlp.py` 中并没有在 `argparse` 中定义 `--model_dir` 参数。程序是通过：
*   `--output_root`：控制输出的根目录（默认已为 `checkpoints/mpp_uni2h_mlp`）。
*   `--dataset_name`：控制具体某次实验子文件夹的名称。
*   最终的模型和训练产物输出位置计算逻辑为：`Path(args.output_root) / args.dataset_name`。

### 3. 复用经验与最佳实践
*   如果需要自定义某个 MPP 的实验保存位置（防碰撞），**应当传入 `--dataset_name` 参数**（例如 `--dataset_name smoke_mpp1`），而不要用 `--model_dir`。
*   后续开发若遇到命令行未识别参数报错，可首先在打印出的 `usage: ...` 命令行说明中核对参数的拼写是使用“中划线”还是“下划线”。

---

## 坑三：MPP-3 历史扁平缓存检索未匹配漏洞

### 1. 现象与报错
在批量运行 MPP-3 冒烟测试时，数据加载阶段报错中断：
```text
  [WARN] HYZ15040/train: 4873/4873 未匹配
    patch_x10080_y10080: no .pt
    patch_x10080_y10248: no .pt
...
ValueError: HYZ15040/train: 4873 未匹配 (allow_missing=False)
```

### 2. 根因分析
*   **缓存布局差异**：MPP-3 是以前就提取完毕的历史缓存，保存在 `--flat_cache_root`（即 `mpp_uni2h_cache/3/{patient}/`）的**扁平格式**下。而新提取的 MPP-2/5 则是保存在 `--cache_root`（即 `uni2h_cache/MPP2_UNI/`）的 **mpp_uni 格式**下。
*   **代码死角**：在数据加载器核心脚本 `dataset_mpp_manifest.py` 中，`build_manifest_datasets` 调用 `merge_manifest_patients` 构建训练与验证 Dataset 时，只向其传入了 `cache_root`，**未将 `flat_cache_root` 传给底层 Dataset 实例化对象**，导致 MPP-3 的数据集查找时漏掉了 `flat_cache_root` 这个 fallback 路径，从而报错。

### 3. 修复方案与复用经验
在 [dataset_mpp_manifest.py](file:///d:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/dataset_mpp_manifest.py) 的 `_find_pt_in_cache` 与 `_detect_partner_style` 中新增 `flat_cache_root` 参数并实现自适应 fallback 机制：
```python
# 当在 cache_root 没有检索到 pt 缓存时，自动切换至 flat_cache_root 匹配旧格式扁平目录：
if flat_cache_root:
    candidates.append(flat_cache_root / str(mpp_id) / patient)
    candidates.append(flat_cache_root / f"MPP{mpp_id}_UNI" / patient)
```
此方法彻底打通了扁平和多子目录格式特征的混合检索通道。在以后如果需要兼容多种不同机器、不同批次提取的特征缓存时，可直接通过本补丁方法进行“双通道 fallback 检索”。

---

## 坑四：Git Glob 匹配失败导致命令串强行中断

### 1. 现象与报错
在服务器上把训练指标 CSV 等小文件推送至 Git 时，运行命令：
```powershell
git add checkpoints/mpp_uni2h_mlp/**/*.json mpp_standard_splits/**/*.json
```
报错：
```text
fatal: pathspec 'checkpoints/mpp_uni2h_mlp/**/*.json' did not match any files
```
并且导致后续的 `git commit` 和 `git push` 命令完全没有执行。

### 2. 根因分析
在 PowerShell 环境下，如果对 Git 命令使用了深层通配符（如 `**/*.json`），而当前目录下又确实没有任何符合该匹配的文件（例如有些 MPP 并没有生成独立的 json 评估），`git` 就会直接返回 `fatal` 错误。在 PowerShell 单行多命令串联中，首个命令 `fatal` 会使得整条链条彻底中断。

### 3. 复用经验与最佳实践
在向 Git 仓库提交并过滤推送时：
*   **充分利用 `.gitignore` 文件的保护罩**：我们在 `.gitignore` 中早已设置了 `**/*.pth`，所以大权重文件已被自动排除，绝不会被 add。
*   **不要写死具体的后缀通配符**，直接 add 整个目录即可，这既高效又能杜绝 pathspec 未匹配错误：
    ```powershell
    git add checkpoints/mpp_uni2h_mlp/
    git add mpp_standard_splits/
    ```

---

## 坑五：物理分辨率差异与硬编码 224 导致的隐藏数据泄漏漏洞

### 1. 现象与隐患
在前期 MPP-5 (0.54 MPP, 50% 步长重叠) 的空间划分中，审计脚本将部分重叠样本错误地判定为 `100%_stride`（无重叠，无需 Embargo），从而使得有物理重叠的训练集和验证集样本同时被模型使用，引发了隐藏的 **Data Leakage（数据泄漏）**。而本应被 Embargo 废弃的样本未能正常隔离。

### 2. 根因分析
*   **历史硬编码**：旧的划分/审计代码中硬编码了有效物理斑块尺寸为常数 `PATCH_SIZE = 224`。
*   **物理跨度差异**：医生虽然把 MPP-3/5 都定为了 50% 重叠，但每个患者的原始高分辨率扫描网格物理间距是不同的（`HYZ`/`LMZ` = 0.18 um/px, `JFX` = 0.27 um/px, `TGC`/`XSL`/`ZHZ` = 0.135 um/px），这导致在各自网格下记录的坐标 dx 也有显著差异。
*   在 MPP-5 (0.54 MPP) 下，对于 **JFX**，其物理 patch 翻倍为 896，相邻步长 dx 为 **`448`**；对于 **HYZ**，其相邻步长 dx 为 **`336`**。
*   由于 `448 > 224` 且 `336 > 224`，当算法硬编码使用 224 作为重叠判定阈值时，JFX 和 HYZ 会被误判为“无重叠”，进而完全逃过了 Embargo 丢弃，导致严重的隐藏空间泄漏！

### 3. 修复方案与复用经验
我们在 `generate_standard_splits.py` 和 `audit_mpp_coordinates.py` 中彻底废除了常数硬编码判定，并引入了**基于分患者主步长 `main_dx` 与重叠类型 (`OVERLAP_MPPS`) 的自适应判定机制**：
```python
# 1. 主动检测各患者各自的真实网格主间距 main_dx/main_dy
# 2. 根据 MPP 步长性质，动态决定隔离带物理跨度
is_overlap = (mpp_id in OVERLAP_MPPS)
effective_patch_size = (2 if is_overlap else 1) * main_dx
```
*   针对 JFX (0.54 MPP, main_dx=448)：其 `effective_patch_size` 自适应判定为 `896`（完美识别相邻 448 距离为 50% 重叠，执行 embargo）。
*   针对 HYZ (0.54 MPP, main_dx=336)：其 `effective_patch_size` 自适应判定为 `672`（完美识别相邻 336 距离为 50% 重叠，执行 embargo）。
*   **复用建议**：在处理 Visium HD 等不同扫描网格或者具有多级多比例切片特征的数据集时，**绝对不能硬编码物理图像斑块像素尺寸（如 224）来推算空间防泄漏阈值**。应全权依据该患者切片在对应 MPP 下的主步长 `main_dx` 进行比例映射，实现真正的物理自适应。

---

## 总结：服务器操作五步标准法

后续在服务器维护或部署时，可按照以下标准化动作防止上述问题的再次发生：
1.  **开机必设**：`$env:PYTHONIOENCODING="utf-8"`。
2.  **拉取代码**：`git fetch gitee main --force; git reset --hard gitee/main`。
3.  **提取特征**：提取参数注意下划线 `--output_root`，不带 `--output_style` 则默认为旧扁平路径。
4.  **开始训练**：注意参数 `--dataset_name` 代替 `--model_dir`。
5.  **指标推送**：直接 `git add checkpoints/mpp_uni2h_mlp/`，大权重自动跳过，实现一键无碍推送。
