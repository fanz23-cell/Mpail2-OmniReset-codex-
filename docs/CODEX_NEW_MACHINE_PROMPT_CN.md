# 给新电脑 Codex 的执行稿

把这个仓库克隆到新电脑以后，直接把下面这段话整段发给 Codex：

```text
你现在在这个仓库根目录工作。不要先给我方案，直接执行并汇报。

目标：
1. 在这台新机器上完整搭好 `mpail2 + UWLab + Isaac Sim + Isaac Lab` 的 OmniReset 环境。
2. 使用仓库自带的一键脚本完成安装。
3. 安装完成后运行 smoke test。
4. 如果我提供了 demo 文件路径，就继续把 demo 转换后启动 `omnireset_peg` 训练入口。

请按下面顺序执行：

1. 先检查当前目录是否包含 `scripts/bootstrap_lab_omnireset.sh`
2. 如果存在，直接运行：
   `bash scripts/bootstrap_lab_omnireset.sh`
3. 如果 smoke test 成功，再告诉我下一条可以直接训练的命令
4. 如果失败，不要泛泛而谈，直接贴出真正的阻塞点：缺驱动、缺显存、缺 demo、还是某个 Python 包失败

如果我后续给你 demo 路径，例如：
`/absolute/path/to/demo.pt`

就继续执行：

`bash scripts/bootstrap_lab_omnireset.sh --demo-src /absolute/path/to/demo.pt --train --num-envs 1`

要求：
- 不要改动脚本逻辑，除非你先确认脚本本身出错
- 不要用 devcontainer
- 优先复用仓库里的固定版本和 pinned 依赖
- 每做完一阶段，告诉我当前卡点是否已经从“环境问题”推进到“硬件问题”
```

## 如果你不想让 Codex决定，自己直接跑

新机器克隆仓库后直接执行：

```bash
bash scripts/bootstrap_lab_omnireset.sh
```

如果你已经有 demo：

```bash
bash scripts/bootstrap_lab_omnireset.sh \
  --demo-src /absolute/path/to/demo.pt \
  --train \
  --num-envs 1
```
