# OmniReset 接入和环境搭建说明

这份说明是按你当前这个工作区来写的，目标是把 `UWLab` 里的 OmniReset `peg-in-hole` 任务接到 `mpail2` 里。

## 先说结论

- 你的 CPU 和内存够用。
- 你现在最大的阻塞不是算力，而是 **GPU 驱动没准备好**。
- 你现在不适合先上 devcontainer。
- 最稳妥的方式是 **分两个环境**，不要把 `mpail2`、`UWLab`、`WheeledLab-research` 硬塞进一个 Python 环境里。

## 为什么要分环境

这三个仓库不是一套时代的依赖：

- `mpail2`: Python 3.10
- `UWLab`: Python 3.11，Isaac Sim 5.1
- `WheeledLab-research`: 更老的 Isaac / Python 组合

如果你强行共用一个环境，最容易出现：

- `pip` 依赖互相覆盖
- Isaac 版本冲突
- 代码能 import，但一运行仿真就炸

## 推荐环境方案

### 环境 1：`mpail2-core`

用途：

- 看 `mpail2` 代码
- 跑 Gym / MuJoCo
- 做非 Isaac 的开发

推荐版本：

- Python 3.10

### 环境 2：`mpail2-omnireset`

用途：

- 装 `UWLab`
- 跑 OmniReset
- 在同一个环境里把这个 `mpail2` 仓库也以 editable 方式装进去

推荐版本：

- Python 3.11
- Isaac Sim 5.1

## 不建议先用 devcontainer

原因很简单：

- Isaac Sim + NVIDIA 驱动 + 图形栈本来就复杂
- 你现在 `nvidia-smi` 都还没正常
- 容器会把排错难度再抬高一层

建议：

- 第一阶段先在宿主机把 OmniReset 跑通
- 等训练和演示转换稳定后，再考虑 devcontainer

## 你现在这台机器要先做什么

### 第一步：先修 GPU

你必须先满足这一条：

```bash
nvidia-smi
```

如果还报错，就不要继续装 Isaac。

### 第二步：确认磁盘空间

Isaac Sim、缓存、资产、日志会很吃空间。

建议至少留：

- 60GB 可用空间：最低可接受
- 100GB 可用空间：更稳

### 第三步：准备 Conda

如果你没有 conda，先装 `Miniforge` 或 `Miniconda`。

## 搭建步骤

如果你的目标不是在当前电脑上手工排错，而是把仓库迁到一台新的实验室机器，请直接看：

- [实验室机器迁移说明](./OMNIRESET_LAB_HANDOFF_CN.md)
- [给新电脑 Codex 的执行稿](./CODEX_NEW_MACHINE_PROMPT_CN.md)

如果你不想手动敲版本，可以直接用仓库里这两个环境模板：

```bash
cd /home/abc/workspaces/Mpail2\ ——OmniReset\ \(codex/mpail2
conda env create -f envs/mpail2-core.yml
conda env create -f envs/mpail2-omnireset.yml
```

### A. 建 `mpail2-core`

在终端里执行：

```bash
conda create -n mpail2-core python=3.10 -y
conda activate mpail2-core
cd /home/abc/workspaces/Mpail2\ ——OmniReset\ \(codex/mpail2
pip install --upgrade pip
pip install -e .
```

### B. 建 `mpail2-omnireset`

先切到你准备跑 Isaac 的环境里。推荐按 `UWLab` 官方说明安装 Isaac Sim 5.1。

如果你已经有 Python 3.11 环境：

```bash
conda create -n mpail2-omnireset python=3.11 -y
conda activate mpail2-omnireset
```

然后安装 `UWLab` 相关包。按这个工作区的结构：

```bash
cd /home/abc/workspaces/Mpail2\ ——OmniReset\ \(codex/UWLab
pip install --upgrade pip
pip install -e source/uwlab
pip install -e source/uwlab_assets
pip install -e source/uwlab_rl
pip install -e source/uwlab_tasks
```

再把当前 `mpail2` 也装进去：

```bash
cd /home/abc/workspaces/Mpail2\ ——OmniReset\ \(codex/mpail2
pip install -e .
```

注意：

- 如果这里因为 `python_requires` 卡住，说明你在 Python 3.11 环境里安装 `mpail2` 时触发了版本限制。
- 这种情况下，不要乱改一堆包；先确认老板/组里是想统一迁到 3.11，还是保留双环境。

## 已经帮你接好的入口

现在 `mpail2` 里已经加了这个入口：

```bash
python -m mpail2.train.train --env omnireset_peg --headless log.no_wandb=True
```

它对应的是：

- 任务：`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0`
- 默认对象：`peg` + `peghole`

## Demo 怎么处理

`mpail2` 训练需要它自己的 `.pt` 演示格式。

我已经加了转换脚本：

```bash
python scripts/convert_omnireset_demo.py \
    /path/to/source_demo.pt \
    mpail2/train/isaac_franka/demos/omnireset_peg_state.pt
```

如果输入不是 `.pt` 而是 `.zarr`，脚本也可以尝试读，但前提是环境里装了 `zarr`。

## 训练命令

演示文件准备好以后：

```bash
conda activate mpail2-omnireset
cd /home/abc/workspaces/Mpail2\ ——OmniReset\ \(codex/mpail2
python -m mpail2.train.train \
    --env omnireset_peg \
    --headless \
    log.no_wandb=True \
    runner.path_to_demonstrations=/absolute/path/to/omnireset_peg_state.pt
```

## 如果你只想先验证环境通不通

优先做这几个检查：

1. `nvidia-smi`
2. `python -c "import torch; print(torch.cuda.is_available())"`
3. `python -c "import uwlab_tasks; print('uwlab_tasks ok')"`
4. `python -c "import mpail2; print('mpail2 ok')"`

也可以直接跑我加好的自检脚本：

```bash
cd /home/abc/workspaces/Mpail2\ ——OmniReset\ \(codex/mpail2
python scripts/check_omnireset_readiness.py
```

只要前两步没过，就先不要继续折腾训练脚本。
