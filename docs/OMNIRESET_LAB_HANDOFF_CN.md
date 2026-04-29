# OmniReset 实验室机器迁移说明

这份说明对应当前仓库，目标是：

1. 在一台新的 Ubuntu + NVIDIA 机器上快速复现 `mpail2 + UWLab + Isaac Sim + Isaac Lab`
2. 验证 `OmniReset peg-in-hole` 环境能否创建
3. 在你拿到 demo 数据后，直接启动 `mpail2` 的 OmniReset 训练入口

## 机器要求

- Ubuntu 22.04 / 24.04
- NVIDIA 驱动正常，`nvidia-smi` 可用
- Python 环境使用 `conda`
- 建议显存 `>= 16 GB`

说明：

- 我这台 8GB VRAM 笔记本已经把环境链路装通了，但 `OmniReset` 真正跑起来时仍然会因为 PhysX GPU 显存不足而失败。
- 所以这套迁移脚本是给实验室大显存机器准备的。

## 新机器的一键命令

仓库克隆下来后，在仓库根目录执行：

```bash
bash scripts/bootstrap_lab_omnireset.sh
```

这个命令会自动做这些事：

- 创建 `mpail2-omnireset` conda 环境
- 安装 `torch 2.7.0 + cu128`
- 安装 `Isaac Sim 5.1.0`
- 克隆并安装 `IsaacLab v2.3.2`
- 克隆并固定 `UWLab` 到已验证 commit
- 安装 `UWLab` 和 `mpail2` 所需依赖
- 写入 conda 激活钩子，自动接受 EULA
- 运行 readiness check
- 运行 OmniReset smoke test

## 带 demo 的一键训练命令

如果你已经拿到了 OmniReset demo 文件：

- `.pt`
- 或 `.zarr`

可以直接执行：

```bash
bash scripts/bootstrap_lab_omnireset.sh \
  --demo-src /absolute/path/to/your/demo.pt \
  --train \
  --num-envs 1
```

这条命令会：

- 完成环境安装
- 把 demo 转成 `mpail2` 需要的格式
- 然后启动 `omnireset_peg` 训练入口

## 只跑训练

如果环境已经装好，只想直接跑：

```bash
DEMO_PATH=/absolute/path/to/converted_demo.pt \
NUM_ENVS=1 \
bash scripts/run_lab_omnireset.sh
```

## 只跑 smoke test

```bash
conda run -n mpail2-omnireset \
  python scripts/omnireset_env_smoke.py --headless --num-envs 1
```

## 版本钉死信息

- `Isaac Sim`: `5.1.0`
- `IsaacLab`: `v2.3.2`
- `UWLab`: `36d98afe1166f546083fc6e3c5d5bee04b486d84`
- `PyTorch`: `2.7.0 + cu128`
- `pytorch3d`: `0.7.8+pt2.7.0cu128`

## 说明

- `WheeledLab-research` 不是老板这个 OmniReset 任务的必需依赖，所以迁移脚本没有自动拉它。
- 如果实验室机器显存足够，后续可以把 `NUM_ENVS` 从 `1` 往上调。
