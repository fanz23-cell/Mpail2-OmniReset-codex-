# MPAIL2 + OmniReset 项目交接文档
**日期**：2026-05-04  
**目的**：让下一个 AI 无缝衔接，继续帮助解决 MPAIL2 在 OmniReset peg-in-hole 任务上的训练问题。

---

## 一、老板要求完成的任务

Tyler（导师）要求将 MPAIL2 算法接入 OmniReset peg-in-hole 任务，实现机器人能成功完成插孔操作。具体要求：
- 使用 MPAIL2（模仿学习 + 对抗奖励）算法
- 任务：UR5e + Robotiq 2F85 夹爪做 peg-in-hole 插孔
- 用户录制了约 30 个人手操控的 demo
- Tyler 提供了官网 checkpoint 链接
- 要求从 scratch 训练（目前认为如此，但尚未 100% 确认）

---

## 二、项目基本信息

**工作目录**：`/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-`

**conda 环境**：`mpail2-omnireset`

**任务名**：`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0`

**关键文件**：
- 训练入口：`mpail2/train/train.py`
- 运行脚本：`scripts/run_lab_omnireset.sh`
- Learner：`mpail2/learner.py`
- Runner：`mpail2/runner.py`
- OmniReset 配置：`mpail2/train/configs/omnireset_learner_config.py`
- Demo 文件：`mpail2/train/isaac_franka/demos/omnireset_peg_state_filtered.pt`
- 官网专家 checkpoint：`mpail2/train/isaac_franka/demos/peg_state_rl_expert_seed42.pt`

**UWLab 环境定义**：`_vendor/UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/`

---

## 三、算法背景

**MPAIL2** = MPPI（模型预测路径积分）+ 对抗模仿学习（GAIL/WGAN-GP 风格）
- 用 demo 训练 discriminator（判断 agent 行为像不像专家）
- discriminator 的输出作为 reward 给 MPPI 策略
- 同时训练 dynamics model、value function、policy network
- **不使用** env 自带的 reward（代码里有注释：`THESE ARE NOT TO BE USED IN LEARNING - JUST FOR LOGGING`）

**关键组件**：
- `encoder`：把 215 维 obs 压缩到 latent space
- `dynamics`：在 latent space 预测下一状态
- `discriminator`（reward）：WGAN-GP，判断 latent 转移像不像 demo
- `value`：Q-value 网络
- `policy`：MPPI 采样 + PolicyNetwork 输出动作

---

## 四、观测空间

**215 维 = 43维/帧 × 5帧历史**，每帧包含：
- prev_actions：7 维
- joint_pos（UR5e 6关节 + Robotiq 6关节）：12 维
- 末端位置+姿态（XYZ + 轴角）：6 维
- 插销相对末端的位置+姿态：6 维
- 孔相对末端的位置+姿态：6 维
- 插销相对孔的位置+姿态：6 维

---

## 五、Run 1～7 完整历史

### Run 1（失败）
- **现象**：disc_demo 和 disc_gen 数值爆炸（disc_demo=7519，disc_gen=+2832）
- **根因**：
  1. `gp_coeff=0.1` 太小（标准 WGAN-GP 应为 10.0），Lipschitz 约束无效
  2. `replay_ratio=1.0` × 32 envs × 160 steps = 5120 次 discriminator 更新/iter（正常应约 100 次）
  3. `target_entropy=-3.0`（Franka Push 的默认值，7D 动作应为 -7.0）
- **日志**：`logs_run1_failed/`

### Run 2（1 个环境测试）
- **改动**：num_envs=1，验证 discriminator 是否因并行环境数引起
- **现象**：discriminator 数值稳定，确认是 replay_ratio × num_envs 问题
- **日志**：`logs_run2_1env/`

### Run 3（修复配置后）
- **改动**：修复 `gp_coeff=10.0`、`replay_ratio≈0.0195`、`target_entropy=-7.0`
- **现象**：discriminator 稳定（disc_demo~40，disc_gen~-30），dyn_loss 下降，但机器人仍不动
- **日志**：`logs_run3/`

### Run 4/5（dynamics 冷启动修复）
- **改动**：`update_dynamics` 加入 demo 数据联合训练 encoder，防止 encoder 只见随机轨迹
- **现象**：dyn_loss 收敛更快，从 0.39 降到 0.023（17x 提升）
- **代码改动位置**：`mpail2/learner.py` 的 `update_dynamics` 函数（已合并 demo_obs_traj）

### Run 6（stdout 修复）
- **问题**：conda run 通过 pipe 传递 stdout，tee 重定向失效，看不到输出
- **修复**：`scripts/run_lab_omnireset.sh` 改为 `"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"`

### Run 7（50 iter 诊断）
- **命令**：
  ```bash
  NUM_ENVS=32 \
  DEMO_PATH=mpail2/train/isaac_franka/demos/omnireset_peg_state_filtered.pt \
  LOG_FILE=/tmp/mpail2_run7.log \
  bash scripts/run_lab_omnireset.sh \
    log.no_wandb=True log.video=False enable_cameras=False \
    runner.num_learning_iterations=50 \
    log.checkpoint_every=50
  ```
- **结果**（50 iter）：

  | 指标 | iter 0 | iter 49 | 判断 |
  |------|--------|---------|------|
  | disc_demo | 9.1 | 46.4 | ✅ 正常上升 |
  | disc_gen | -15.5 | -35.3 | ✅ 正常下降 |
  | dyn_loss | 0.39 | 0.023 | ✅ 收敛良好 |
  | rew | 1.76 | 1.73 | ⚠️ 基本平坦 |

- **结论**：infrastructure 完全正常，但机器人不动（爪子在原地随机乱动，从未碰到任何东西）
- **Checkpoint**：`logs/models/model_50.pt`

---

## 六、当前确认修改的代码

### 1. `mpail2/train/configs/omnireset_learner_config.py`（已修改）
```python
policy_learner_cfg: defs.PolicyLearnerConfig = field(
    default_factory=lambda: defs.PolicyLearnerConfig(target_entropy=-7.0)  # 原来是默认 -3.0
)
replay_ratio: float = 100.0 / (32 * 160)  # 原来是默认 1.0，约 0.0195
reward_learner_cfg: defs.RewardLearnerConfig = field(
    default_factory=lambda: defs.RewardLearnerConfig(gp_coeff=10.0)  # 原来是默认 0.1
)
```

### 2. `mpail2/learner.py` 的 `update_dynamics`（已修改）
- 增加 `demo_obs_traj` 和 `demo_next_obs_traj` 可选参数
- 合并 agent + demo 数据联合训练 encoder（防止冷启动）
- JEP loss 只在 agent 部分计算

### 3. `scripts/run_lab_omnireset.sh`（已修改）
- 最后一行改为：`"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"`
- 修复 conda run 吞掉 stdout 的问题

---

## 七、当前核心未解决问题

**机器人在 `Finetune-Play-v0` 环境下完全不动**——50 iter、500 iter 都不会改变这个现象。

**根本原因分析**：
- `Finetune-Play-v0` 是 UWLab 官方 **Stage 2 finetune 的 eval 环境**，设计前提是已有训好的 Stage 1 policy
- OmniReset 官方训练流程（PPO）：Stage 1（从零训）→ Stage 2 Finetune → Eval/Play
- MPAIL 是模仿学习算法，不是 from scratch RL
- Peg-in-hole 精度要求 <5mm，随机动作永远无法偶然成功，discriminator 永远见不到成功状态

**对比 Kinova/Franka 为什么能成功**：
- Kinova：state_dim=6，任务空间极小，随机动作 700 步内能碰到目标
- Franka Push：推方块，随机动作能碰到物体
- OmniReset：精确插孔，随机动作永远无法成功，是本质区别

---

## 八、当前正在进行的操作

**正在收集专家 demo**（后台运行，PID 506231）：
```bash
~/miniconda3/bin/conda run -n mpail2-omnireset \
  python scripts/collect_state_demos_omnireset.py \
    --checkpoint mpail2/train/isaac_franka/demos/peg_state_rl_expert_seed42.pt \
    --output     mpail2/train/isaac_franka/demos/peg_expert_rl_raw.pt \
    --num-envs 32 --num-steps 10000 --headless
```
- 日志：`/tmp/collect_demo_expert.log`
- 使用官网下载的 PPO 专家 checkpoint（`peg_state_rl_expert_seed42.pt`，6.5MB）
- 目标：收集 10000 步专家轨迹，替换原来的 30 个人手 demo

**注意**：这个方向是否正确尚不确定，需要 Tyler 确认。

---

## 九、必须问 Tyler 的问题

以下问题在继续之前必须获得确认：

1. **MPAIL 训 OmniReset 的正确 demo 来源是什么？** 是人手录的，还是用官网 PPO checkpoint 跑出来的，还是别的？
2. **用的是 State 版本还是 Image 版本？** 任务名是 `Finetune-Play-v0`（State）还是 RGB 版本？
3. **MPAIL 应该在哪个任务上跑？** 是 `Finetune-Play-v0` 还是另一个任务名？
4. **from scratch 怎么解决早期探索问题？** peg-in-hole 随机动作无法成功，discriminator 怎么学？

---

## 十、官网资源

- **OmniReset 文档**：https://uw-lab.github.io/UWLab/main/source/publications/omnireset/
- **Distillation 文档**：https://uw-lab.github.io/UWLab/main/source/publications/omnireset/distillation.html
- **Sim2Real 文档**：https://uw-lab.github.io/UWLab/main/source/publications/omnireset/sim2real.html

**官网专家 checkpoint 下载链接（State-based）**：
```bash
# Peg Insertion
wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/peg_state_rl_expert_seed42.pt
wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/peg_state_rl_expert_seed0.pt
wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/peg_state_rl_expert_seed1.pt
```
注意：wget 下载失败，需用 `curl -L` 跟随重定向。

---

## 十一、demo 文件说明

| 文件 | 说明 | 状态 |
|------|------|------|
| `omnireset_peg_state.pt` | 用户人手录制的原始 demo，约 8.3MB | 可用但质量有限 |
| `omnireset_peg_state_filtered.pt` | 过滤掉 idle 状态后的 demo，2318 条，3.9MB | 当前训练使用 |
| `peg_expert_seed42.pt` | 早期下载的某个 checkpoint，内容格式同官网一致 | 实为 model checkpoint，不是 demo |
| `peg_state_rl_expert_seed42.pt` | 官网下载的 PPO 专家 model checkpoint，6.5MB | 用于收集 demo |
| `peg_expert_rl_raw.pt` | 正在收集中（专家 checkpoint rollout 出来的 demo） | 收集中 |

---

## 十二、下一步建议

**优先级 1（立刻做）**：联系 Tyler，问清楚第九节的四个问题。

**优先级 2（等 Tyler 回复前）**：等 demo 收集完成后，可以尝试用新 demo 替换旧 demo 跑 50 iter，观察机器人是否开始移动。

**优先级 3（Tyler 确认后）**：根据 Tyler 的指导，可能需要：
- 换任务名（用 Stage 1 的任务环境而不是 Finetune-Play）
- 换 demo 来源
- 或者确认现在的方向是对的，只是需要更多 iter

---

## 十三、环境和硬件

- OS：Ubuntu 24.04
- GPU：需要 CUDA（当前在 cuda:0 运行）
- Isaac Sim 初始化时间：约 10 分钟
- 训练速度：约 128 fps，50 iter 约 40 分钟（不含初始化）
- conda 环境：`mpail2-omnireset`，conda 路径 `~/miniconda3/bin/conda`
- 注意：后台 shell 没有 conda 初始化，必须用 `~/miniconda3/bin/conda run` 而不是 `conda run`
