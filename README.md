# JEMM × TacClaw Rollout：RoboLab / Tabero 实现核对文档

本文用于**核对实现细节**：在 RoboLab 和 Tabero 上用 VLM 按 Show-Harness 的 meta action 方式控制机器人、收集 rollout 轨迹，末端执行器使用 TacClaw。

**已定（2026-10-24）：策略用 JEMM，取代 GPT-6**（见 1.1 节）。下文沿用的"GPT"字样（包括代码里的 `gpt_rollout/`、`gpt_calls/`）在此之后都指当前策略；GPT-6 的路径仍保留为 `--policy gpt`。

**已定（2026-10-25）：换回 GPT-6（`openai/gpt-6-astra`），`--policy gpt` 成为默认**（见 4.5 节）。JEMM 的路径保留为 `--policy jemm`，1.1 节只适用于它。

写作约定：

- 只记录两类内容：本方案的设计，以及原仓库代码里已有的行为。**不引入新的字段或规则。**
- 每条事实都标出来源：
  - `[设计]`：本方案自己的设计，脚手架代码还没有实现；
  - `[代码]`：本目录下的源码，写明文件路径和行号；
  - `[待对齐]`：设计里还没有定义，或者和原仓库代码冲突，需要人来决定。
- 路径都相对于本目录（`gujie/rollout/`）。上游仓库的行号以第 0 节表中的 commit 为准，本仓库代码的行号以 2026-10-25 的代码为准。

### 当前状态（2026-10-25）

| | 状态 |
|---|---|
| 环境 | 三个运行环境都已建好（第 0 节）：Isaac Sim 5.1、JEMM 服务（int8，单卡）、LeRobot 发布 |
| 采集流程 | 已实现并跑通：策略（现为 GPT-6）选动作 → Show-Harness `AtomicExec` 执行 → 每个回合写出逐帧（LeRobot v3）和逐决策（Show-Harness）两份数据（第 1、4、7 节） |
| 时间上限 | 两个 benchmark 都是任务原值的四倍（2026-10-26 由三倍改），不限决策次数（0.1 节） |
| 试跑结果 | 两倍时长下 7 个任务各 1 个种子，**全部失败**：手离要抓的物体最近也有 17cm，从没下降到物体的高度（4.4 节）。原时长下的冒烟测试和全量采集的前 11 个回合也全部失败 |
| GPT-6 冒烟 | Tabero `libero_object` task 0：仍是超时失败，但手下降到离汤罐 0.5–1.1cm，之后在原地 `GRASP` / `RELEASE` 交替（4.5 节） |
| 采集 | **暂停（2026-10-25）**：GPT-6，Tabero 46 个和 RoboLab 120 个任务各 1 个种子，四倍时长，还没决定要不要全量。同样 7 个任务试跑了四次（4.5、4.6 节）：第一次 2/7、改提示词后 2/7、触觉垫摩擦系数改成 5/4 后 5/7、**再改成四倍时长后 5/7**（成败的任务换了两个，见 4.6 节末尾）。JEMM 的全量采集已停，输出已清空 |
| 结果浏览 | `scripts/site/build_site.sh` 把一个输出目录做成网页：视频、离散决策、逐帧动作、策略的完整输入输出，可以附一节案例分析（7.3 节）。JEMM 和 GPT-6 的输出都支持 |

---

## 0. 目录与代码基线

| 目录 | 内容 | 与本任务的关系 |
|---|---|---|
| `RoboLab/` | NVIDIA RoboLab 上游仓库 | **没有 TacClaw 支持**，只用于对照 |
| `Tabero/` | Tabero（TacManip）上游仓库 | **没有 TacClaw 支持**，只用于对照 |
| `TacHarness/` | 集成仓库，包含 TacClaw 定义，以及接入了 TacClaw 的 RoboLab/Tabero 副本 | **实际要用的基线** |

- TacClaw 就是 Daimon DM25PD04，代码里叫 `tacumi_dm25pd04_v1`，统一定义在 `TacHarness/src/tacumi/embodiment.py`。`[代码]`
- 接入了 TacClaw 的 RoboLab 在 `TacHarness/src/providers/RoboLab/`，Tabero 在 `TacHarness/src/providers/LIBERO/`。`[代码]`
- 脚手架代码由我们自己写，在 `gpt_rollout/` 和 `scripts/` 下。标 `[设计]` 的内容是方案的目标；已经实现的部分会同时标 `[代码]` 并写明位置。

### 上游依赖（不在本仓库里）

本仓库只包含脚手架（`gpt_rollout/`、`scripts/`、`tests/`、`configs/`）。运行前要把下面的仓库和模型放到本目录下，目录名和版本都要一致：

| 目录 | 来源 | 版本 |
|---|---|---|
| `TacHarness/` | https://github.com/LV-Robotics-Lab/TacHarness.git | `c5cf373a50d54151f42fc74dbe87e93922f348a6` |
| `Show-Harness/` | https://github.com/showlab/Show-Harness.git | `5a3713b6d90c5a5a8c9461e298e091f7cbdcded4` |
| `RoboLab/` | https://github.com/NVlabs/RoboLab.git | `ad45d4f974725d020f82c2b0d77d78533aeba2b3`（只用它的 `assets/`，这些是 Git LFS 文件，要 `git lfs pull --include="assets/**"`，约 6GB） |
| `Tabero/` | https://github.com/NathanWu7/Tabero.git | `56a3b9055bfe7dc157583cc10e121c7a7781cf8f`（只用于对照） |
| `models/JEMM/` | https://huggingface.co/MaestroYan/JEMM | `76e3c209e8441fa658221c7ba2725bad2f811176`（LoRA adapter） |
| `models/Qwen3.8-27B/` | https://huggingface.co/Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`（JEMM 的基座，55.6GB） |

运行环境有三个，互相独立：

| 环境 | 用途 | 怎么建 |
|---|---|---|
| `TacHarness/.venv-isaacsim51` | Isaac Sim 5.1 仿真和 rollout 循环 | `scripts/setup_env.sh` |
| `.venv-jemm` | JEMM 服务（torch 2.12.1+cu126、transformers 5.20.2、peft 0.21.2、bitsandbytes 0.50.2、`jemm@6822fe0`） | `uv venv --python 3.12 .venv-jemm`，再装上述包和 `jemm @ git+https://github.com/ypcypc/JEMM@6822fe0fd53c5e6670af6ba99fb2c857a661e532` |
| `.conda/tacharness-lerobot312` | 发布 LeRobot v3（lerobot 0.6.0） | `TacHarness/scripts/setup_lerobot312_runtime.sh`；设 `TACHARNESS_HF_HOME` 指向环境目录之外，否则脚本会先建出环境目录再报"目录已存在"。脚本最后的自检在这个 commit 上会失败（找不到 `tacharness` 包），不影响使用 |

默认的 `--policy gpt` 要把 `configs/llm.example.yaml` 复制成 `configs/llm.yaml`，填入端点和 key（`llm.yaml` 不进仓库）。

**本机遇到的两个环境问题**（已在 `scripts/` 里处理）`[代码]`：

- 系统缺 `libGLU.so.1`，RTX 材质编译用的 MDL SDK 加载失败，相机渲染随即出错。`scripts/setup_env.sh` 用 `apt-get download libglu1-mesa` 把库解到 `.local/lib`（不需要 root），`scripts/isaac_python.sh` 把它加进 `LD_LIBRARY_PATH`。
- Isaac Sim 启动时会在所有可见 GPU 上做 P2P 测试，别的进程占满的 GPU 会让它显存不足。`scripts/isaac_python.sh` 默认 `CUDA_VISIBLE_DEVICES=0`。

**运行方法** `[代码]`：

```bash
scripts/jemm_serve.sh > logs/jemm_serve.log 2>&1 &              # 只有 --policy jemm 要：JEMM 服务，GPU 0，int8，端口 8790
scripts/collect.sh tabero  outputs/collect                        # Tabero 46 个任务，每个 50 个种子，可断点续跑
scripts/collect.sh robolab outputs/collect                        # RoboLab 120 个任务
scripts/collect_parallel.sh outputs/collect_gpt6_x3               # 两个 benchmark 每个任务 1 个种子，3 个进程并行（4.5 节）
scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite libero_object --task-id 7 --num-seeds 1 \
    --output-root outputs/smoke                                   # 单个任务
scripts/site/build_site.sh outputs/smoke                          # 把输出做成结果浏览网页（7.3 节）
```

长时间采集用 `setsid nohup … &` 启动，关掉会话也会继续跑。停止时 Isaac 进程不响应 SIGTERM，要用 SIGKILL。

- [ ] 核对：实现所用的 RoboLab 和 Tabero 都是 `TacHarness/src/providers/` 下的副本，不是上游仓库。
- [ ] 核对：脚手架实现后，逐条对照本文的第 2–4 节。

---

## 0.1 硬约束：任务自己的超参数不能修改

**每个任务自带的超参数一律保持原值，脚手架去适应任务，不能反过来改任务。** 这条约束决定了下面几件事，它们不再是待对齐项：

| 任务超参数 | 原值 | 对脚手架的要求 |
|---|---|---|
| 环境步进频率（`dt`、`decimation`） | RoboLab 15Hz；Tabero 20Hz | GPT 输出的每一行动作对应一次 `env.step`：RoboLab 上是 1/15 秒，Tabero 上是 1/20 秒 |
| 回合时长 `episode_length_s` | RoboLab 每个任务自己定（例如 BananaInBowl 50 秒）；Tabero 8 秒，libero_10 为 16 秒 | 一条轨迹的机器人动作必须在这个时长内完成：Tabero 是 160 步，libero_10 是 320 步 |
| 成功判定的函数和阈值 | 见第 6 节 | 原样使用，不放宽也不收紧 |
| 失败终止条件（物体掉落等） | 见第 6 节 | 原样使用 |
| 开局（初始状态、物体摆放） | benchmark 自带 | 原样使用 |

**例外（已定，2026-10-11）：回合步数上限在收集 rollout 时可以放大。** GPT-6 没有经过训练，原值不够用。表中其他各项仍然不能改。

**已定（2026-10-24）：两个 benchmark 都把每个任务自己的时间上限放大两倍，并去掉"每回合最多 200 次决策"的上限。** 我们是收集 rollout，不是评测。`[代码]`

**已定（2026-10-25）：改为放大三倍。**

**已定（2026-10-26）：改为放大四倍。** 三倍时长下 `libero_object` 在第 471 步成功，离上限 480 只差 9 步（4.6 节）。下面的数字已按四倍更新。

- 做法：环境建好后把 `cfg.episode_length_s` 乘以 `--time-limit-scale`（默认 4，`gpt_rollout/__init__.py` 的 `TIME_LIMIT_SCALE`）。IsaacLab 每次判断超时都从这个值现算步数（`TacHarness/thirdparty/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:96-103`），所以建好后再改也生效。实现见 `gpt_rollout/sim.py` 的 `scale_time_limit`。
- 放大后：Tabero 640 步（`libero_10` 1,280 步）；RoboLab 各任务为原 `episode_length_s` 的四倍，例如 BananaInBowl 从 750 步变成 3,000 步。
- 去掉决策上限后，回合只由成功、放大后的超时、物体掉落结束。每个决策至少占 1 个 env 步，所以回合一定会结束。只有关掉超时的测量模式（`--disable-time-out`）仍用 200 次。
- 每个回合的结果同时记录原上限 `max_episode_steps`、放大后的上限 `episode_step_limit`、倍数 `time_limit_scale`，以及是否在原上限内成功 `success_within_max_episode_steps`（两个 benchmark 都记）。
- Tabero 放大两倍可能仍然不够：按第 4.2 节，一次不出错的抓放约 500–630 步。两倍时长试跑（4.4 节）里 JEMM 离完成还差得很远，瓶颈不在时间，所以暂不再放大。2026-10-25 改为三倍，2026-10-26 改为四倍（见上）。

- 策略思考时仿真暂停，不占回合时间。
- **测量模式（2026-10-11 定，仍可用）：** 加 `--disable-time-out`，Tabero 关掉超时终止，最多 200 次决策，用来测一个任务到底要多少步；结果里同时记录"是否在原上限内成功"（`success_within_max_episode_steps`）。正式采集不用这个模式。

以下参数不属于任务本身，按代码默认值处理（已定，2026-10-04）：

- TacClaw 的闭合角用代码默认值 20°，不用环境变量覆盖。
- **例外（已定，2026-10-26）：触觉垫的摩擦系数从静 2、动 1.5 改成静 5、动 4**，TacClaw 的其他参数不变（5.1 节、4.6 节）。
- Tabero 评测脚本里的 `num_success_steps = 8` 是成功规则的一部分，沿用默认值。
- Tabero 评测脚本里的 `max_inference_steps = 30`、`replan_steps = 10` 是给 VLA 策略循环用的，GPT 的循环用不到，实际起约束作用的是上面的回合时长。

---

## 0.2 采集范围 `[设计]`

采集的目的是产生训练数据，不是评测。以下各项已定（2026-10-04）：

| 项目 | 决定 |
|---|---|
| 范围 | 只在训练集上收集 rollout，不碰任何测试集 |
| Tabero 采哪些任务 | 全部 5 个套件、46 个任务 |
| RoboLab 采哪些任务 | 它自带的全部 120 个 benchmark 任务 |
| 每个任务采多少条 | 50 条，用 50 个不同的随机种子 |
| Tabero 的开局 | 不使用示范轨迹，直接采样：用环境的默认重置 |
| Tabero 的步数上限 | 原值 160 步（`libero_10` 为 320 步）的四倍：640 步（`libero_10` 1,280 步），见 0.1 节（2026-10-24 改为两倍，2026-10-25 改为三倍，2026-10-26 改为四倍）。超时、物体掉落两个终止条件按任务配置保留 |

合计：Tabero 46 × 50 = 2,300 次 rollout，RoboLab 120 × 50 = 6,000 次 rollout。**已定（2026-10-24，取代 2026-10-04 的"只记录成功的轨迹"）：成功和失败的轨迹都完整记录**，两者保存的内容相同，用 `success` 区分。每个任务的 `results.jsonl` 仍然每个回合留一行结果（种子、结束原因、步数、调用次数和花费），用来统计成功率和断点续跑。已实现：`gpt_rollout/episode.py:228` 的 `finish_episode` 只追加这一行结果，不再删除失败回合的数据。`[代码]`

### 两个 benchmark 的现状 `[代码]`

| | 任务和开局 | 位置 |
|---|---|---|
| Tabero | 任务定义在 `benchmarks/datasets/libero/config/<套件>.json`：`libero_10`、`libero_goal`、`libero_object`、`libero_spatial` 各 10 个任务，`libero_tactile` 6 个（文件里写的总数是 7，实际只有 6 条）。每个物体在任务 JSON 里有一个初始区域（`regions` 的 `pose_range`），环境重置时在这个范围里随机采样物体位姿 | `TacHarness/src/providers/LIBERO/source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py:152-166` |
| RoboLab | 仓库里**没有训练集**。它是纯评测用的 benchmark，120 个任务全部是测试任务；被评测的模型是在别处的真机数据（DROID）上训练的。120 个任务都没有定义随机化事件，重置时只是把场景恢复到固定的默认状态 | `TacHarness/src/providers/RoboLab/robolab/tasks/README.md`、`robolab/core/environments/base.py:40-42`、`robolab/core/environments/config.py:153-155` |

- Tabero 原评测脚本另外支持从示范轨迹文件（`{套件}_task{编号}_*_demo.hdf5`）读初始状态。我们不用它；这些文件也不在本机。
- Tabero 原评测脚本会关掉超时和物体掉落两个终止条件，改用自己的上限 300 步（`benchmarks/common/closedloop_policy_inference.py:58-72`）。我们不这样做，按任务配置保留这两个终止条件。

### 50 个种子在两边的效果不同

| | 换种子会改变什么 |
|---|---|
| Tabero | 每个物体在初始区域里重新采样位姿，50 条轨迹的开局各不相同 |
| RoboLab | **开局不变。** 50 个种子得到 50 个完全相同的开局，轨迹之间的差别只来自 GPT 输出本身的随机性 |

RoboLab 上有两种处理方式：

1. **接受开局相同。** 不改任何东西，靠 GPT 自身的随机性产生差异。差异可能很小。
2. **给任务加开局随机化。** RoboLab 有现成的机制（`RandomizeInitPoseUniform`，按给定范围随机挪动物体，见 `RoboLab/docs/environment_run.md`）。但这会改动开局，和 0.1 节"开局原样使用"冲突，而且每个任务挪哪些物体、挪多大范围都要另外定。

- RoboLab 里只有 `robolab/tasks/randomize_initial_pose/` 下的 6 个任务自带开局随机化（BananaInBowl 和 RubiksCubeAndBanana，各有 10cm、20cm、30cm 三档），它们不在 120 个 benchmark 任务里。
- [ ] `[待对齐]` RoboLab 上选 1 还是 2。建议先选 1，因为它不违反 0.1 节的硬约束；看到实际轨迹的差异程度后再决定要不要上 2。

### 采集规模

按两倍时长试跑（4.4 节）的实测速度，假设每个回合都跑到放大后的上限（JEMM 目前每条都失败，所以就是这种情况），单个 Isaac 进程串行：

| | 任务数 × 种子 | env 步总数（上限） | 每个回合的真实用时 | 合计 | 数据量（两份合计，约 25KB/帧） |
|---|---|---|---|---|---|
| Tabero | 46 × 50 = 2,300 | 约 90 万 | 320 步的任务约 3.5 分钟，`libero_10` 约 6.5 分钟 | 约 7 天 | 约 22GB |
| RoboLab | 120 × 50 = 6,000 | 约 1,640 万（各任务 `episode_length_s` 合计 10,940 秒 × 2 × 15Hz × 50） | 每步约 1 秒（含 JEMM 决策） | 约半年 | 约 400GB |

- 一次决策平均约 9 个 env 步、JEMM 约 1.5 秒；一个 env 步加渲染、触觉和录制约 0.7–1 秒。
- 一个 Isaac 进程占 7–9GB 显存，JEMM（int8）占约 31GB。GPU 0 上只放得下 JEMM 加一个 Isaac 进程；每多一张空的 L40S 可以再放 4–5 个 Isaac 进程，共用同一个 JEMM 服务。
- [ ] `[待对齐]` RoboLab 的采集规模：多要几张卡并行、先采一部分任务，还是减少种子数。

### 已知会失败的任务

采全部任务，不做排除，所以下面这些任务大概率失败，产不出轨迹：

- 物体最窄处比 79.7mm 宽的任务，TacClaw 夹不住（见第 5.1 节）；
- 开抽屉、开柜门这类任务，GPT 未必做得了。

核对项：

- [ ] 核对：`libero_tactile` 这个套件能否在 TacClaw 环境下运行。第 5.3 节记录了 TacClaw 没有 `Tactile-*` 环境，这个套件是否依赖它还没有查。
- [ ] 核对：每条轨迹的元数据里记录了随机种子，开局可以复现。
- [ ] 核对：Tabero 的超时和物体掉落终止条件处于开启状态，没有被关掉。

---

## 1. 整体流程 `[设计]` `[代码]`

**策略按 Show-Harness 的 meta action 方式控制（已定，2026-10-11）：每次调用只输出一个离散动作，由解释器在仿真里把它执行完，再用新的观测问下一次。** 提示词里要向策略解释每个输入是什么、每个动作做什么。

```
循环：
  1. 把当前观测发给策略（第 2 节的 6 项输入，加最近 5 个动作）
  2. 策略给出一个离散动作（第 3 节）
  3. 解释器把这个动作展开成若干个 env.step 执行完，每一步录一帧（第 4 节）
  4. 回到 1，直到官方规则判定成功、超时或物体掉落
```

- **不使用 benchmark 里的示范动作**，只借开局状态和官方成功判定。
- 没有子目标规划，也不用 Show-Harness 的其他插件。GPT 只输出动作，不输出思考文字。
- 每次调用互相独立，不带之前的对话。唯一的记忆是提示词里"最近 5 个移动动作，最新的在前"，这是 Show-Harness 仿真 runner 的 meta-action 提示词本来就有的一行（已定，2026-10-11）。
- **已定（2026-10-25）：`GRASP`、`RELEASE` 也记进这一行**，改叫 "Recent actions"，仍是最近 5 个，最新的在前。Show-Harness 只记移动动作；那样 `RELEASE` 之后的输入和 `GRASP` 之前完全一样，三倍时长试跑里策略就在原地一直抓放（4.5 节）。`[代码]` `gpt_rollout/episode.py` 的 `run_episode`
- **已定（2026-10-26）：改写提示词里抓取的两条说明**（`gpt_rollout/prompts.py` 的 `HOW TO ACT`）。原来要求"下降到夹爪位置等于物体位置再 `GRASP`"，并且"夹爪闭合但物体不在夹爪处就算抓失败，`RELEASE` 重来"。移动是 2cm 一格，物体坐标又是根坐标，两者基本对不上，所以三倍时长试跑里 `GRASP` 之后的 73 次决策有 71 次是 `RELEASE`，而当时开口都停在 63–74mm，其实已经夹住了（4.5 节）。现在改成：下降到物体在两指之间就 `GRASP`，不需要位置重合；`GRASP` 之后觉得夹到了就用 `MV_UP` 往上提，物体跟着上来就继续提，掉了或没跟上来就 `RELEASE`，回到物体处再试。同时删掉了"不要选和上一个移动相反的动作，除非走过头了"这一条。`[代码]`
- 代码：`gpt_rollout/episode.py`（循环）、`gpt_rollout/interpreter.py`（解释器）、`gpt_rollout/prompts.py`（提示词）。`[代码]`

### 1.1 策略：JEMM

2026-10-25 起默认策略换回 GPT-6（见文首和 4.5 节），本节只适用于 `--policy jemm`。

**已定（2026-10-24）：用 JEMM 取代 GPT-6。** JEMM 是挂在 Qwen3.8-27B 上的 LoRA（`models/JEMM/`）：给它一段状态、一个问题和 2–32 个候选，它返回每个候选的概率。

| 项目 | 做法 |
|---|---|
| 运行 | 单独的服务进程 `scripts/jemm_serve.sh`（在 `.venv-jemm` 里），接口和 JEMM 官方服务相同：`POST /v1/systemone`。rollout 这边是 `gpt_rollout/jemm.py` 的 `JEMMClient` 和 `JEMMPolicy`。`[代码]` |
| 精度 | **已定（2026-10-24）：先用 int8。** 基座的线性层用 bitsandbytes 的 LLM.int8 量化；LoRA、`lm_head` 和视觉编码器保持原精度，LoRA 不合并（`scripts/jemm_serve.py`）。服务占 GPU 约 30.8GB，单张 46GB 的 L40S 放得下；bf16 约 57GB，要两张卡。JEMM 的温度和阈值是按 bf16 校准的，int8 带来的概率偏移还没测。`[代码]` |
| 提示 | JEMM 的 system prompt 是训练时固定的，不能换。第 2 节要求解释的内容和每次的观测，都放进它的 `State`（`gpt_rollout/prompts.py` 的 `jemm_state`：说明文字和 GPT 提示词相同，只去掉输出格式那一段）。问题是 `JEMM_QUESTION`；8 个动作是 `Candidates`，描述见 `ACTION_DESCRIPTIONS`。三张图按场景、腕部、触觉的顺序发送。`[代码]` |
| 输出 | 8 个动作各自的概率。每次都是合法输出，所以第 4 节"输出不合法就重新调用"的规则只适用于 `--policy gpt`。 |
| 选动作 | **已定（2026-10-24）：按概率采样。** 随机数生成器用回合的种子初始化，同一种子在概率相同时可以复现。RoboLab 50 个种子的开局相同，靠采样才有差异。置信度低于 JEMM 阈值（`decision_config.json` 的 `threshold`，0.987）的调用照常执行，只在日志里标 `below_threshold`。`[代码]` |
| 记录 | 每次调用的概率、置信度、延迟，记在 `gpt_calls/calls.jsonl` 的 `attempts` 里。`[代码]` |

- 一次调用约 2,200 个 token，其中触觉拼图按原尺寸约占 960 个，在 JEMM 的多模态上限 3,072 以内（实测，2026-10-24）。
- 同一份输入调用三次，概率完全相同，所以采样是轨迹之间唯一的随机来源。

### 1.2 一次请求长什么样 `[代码]`

**每做一次决策，就发一个独立的请求**：一个回合有多少次决策，就发多少次。请求之间没有对话历史，也不带之前的图片；唯一的记忆是文字里最后那行"最近 5 个动作"（含 `GRASP`、`RELEASE`，1 节）。JEMM 收到的消息由它自己的 `jemm.contract.mm_messages` 拼成：

```text
[system]    Choose the best available candidate for the question using only the supplied state.
            Return exactly one candidate label.                          ← JEMM 训练时固定
[user]      <图1 场景相机 224×224> <图2 腕部相机 224×224> <图3 触觉拼图 原尺寸>   ← 三张图在前
            State:
            <说明文字：输入是什么、坐标系、8 个动作、怎么抓放>               ← 每次都一样（第 2 节要求解释的内容）
            CURRENT OBSERVATION
            Task instruction: …
            Robot state: position / orientation / gripper                  ← 每次都变
            Object positions (x, y, z): …                                  ← 特权信息，不进训练数据
            Recent actions, newest first: …
            Question: Which ONE action should the robot take next to carry out the task instruction?
            Candidates:
            A) MV_FWD: move the gripper 2 cm along +x.   …   H) RELEASE: stop the arm, then open the gripper.
            Answer with exactly one candidate label.
[assistant] （不生成文字：读下一个 token 是 A–H 的 logits，除以温度 1.395 后 softmax，得到 8 个概率）
```

- rollout 端发给服务的是 JEMM 官方的 `POST /v1/systemone` 请求体：`state`（上面 State 之后的整段文字）、`questions.action`（`type: choice`，`instructions` 是问题，`criteria` 是 8 个动作的描述）、`images`（三张 PNG 的 base64）。拼装代码在 `gpt_rollout/jemm.py` 的 `JEMMClient.decide`，文字在 `gpt_rollout/prompts.py`。
- 服务返回 8 个概率、`confidence`（最高的那个概率）和 `choice`（概率最高的动作）。rollout 不用 `choice`，而是按 8 个概率抽样（1.1 节）。
- 完整的一次请求（文字和三张图）可以在结果浏览网页的"输入格式"一节看到（7.3 节）。
- JEMM 是用网页操作、工具调用、意图分类这类数据训练的，State 通常很短，候选是工具名之类。这里的 State 是一大段机器人说明加一串三维坐标，离它熟悉的输入很远，可能是它表现差的原因之一（4.4 节）。

---

## 2. 每次交给策略的输入 `[设计]`

| # | 内容 | 怎么产生 | 进不进训练数据 |
|---|---|---|---|
| 1 | 任务指令 | benchmark 自带 | 进 |
| 2 | 场景相机图像 | 从仿真器直接读取 | 进 |
| 3 | 腕部相机图像 | 从仿真器直接读取 | 进 |
| 4 | 触觉拼图：左右两块垫各一行，每行四格（压痕 INDENT、形变 DEFORMATION、剪切 SHEAR、形变着色 HSV） | TacHarness 现有代码直接生成 | 进 |
| 5 | 机器人状态：手的位置、朝向（四元数），夹爪开或合 | 从仿真器直接读取 | 进 |
| 6 | 物体位置：能读到的每个物体的名字和坐标（只有位置，不带朝向和尺寸） | 从仿真器直接读取 | **不进**，只有 GPT 能看到 |

- 所有输入都是直接读取，**不用另一个 AI 做总结。**
- **已定（2026-10-04）：场景相机**用各自 VLA 评测原本用的那一个：RoboLab 是 `over_shoulder_left_camera`，Tabero 是 `agentview_cam`。
- **已定（2026-10-04）：特权信息只给物体坐标**，不给朝向，不给尺寸。能读到的物体全部给。坐标就用仿真器给出的那个点（物体的根坐标），提示词里不对这个点做额外说明。
- **已定（2026-10-04）：触觉完全沿用 TacHarness 现有的输出，不改任何计算。** 现有输出见下面的 2.1 节。给 GPT 的只有触觉拼图，不另外生成文字摘要，因为 TacHarness 里没有现成的摘要。
- 第 5 项里的手的位置，和第 3 节动作的方向用同一个坐标系（机器人基座坐标系），指的是同一个点。

提示词里要向 GPT 解释的内容（固定文字，每个 benchmark 写一份）：

| 要解释的内容 | 说明 |
|---|---|
| 两张相机图分别是什么视角 | 场景相机的位置；腕部相机装在夹爪上，是鱼眼镜头 |
| 触觉拼图怎么看 | 上下两行是左右两块垫，每行四格各是什么；全黑表示没有接触 |
| 机器人状态里每个数的含义 | 坐标系原点和三个轴的方向、单位是米、指的是两块触觉垫之间的中心 |
| 物体坐标 | 和机器人状态用同一个坐标系 |
| 每个动作做什么 | 见第 3 节；每个方向在场景相机里朝哪边 |
| 控制步多长 | Tabero 是 1/20 秒，RoboLab 是 1/15 秒；一个移动动作走完 2cm 要几步由手臂决定 |
| 夹爪 | 只有开和合；张开时两块垫相距 79.7mm，闭合到约 58mm 宽，不能调力 |
| 最近的动作 | 最近 5 个，最新的在前，包括 `GRASP`、`RELEASE`（2026-10-25 改，1 节） |

- **已定（2026-10-04）：发给 GPT 的两张相机图缩到 224×224**，和现有 VLA 路径一致：先用黑边补成正方形，再缩放（RoboLab 见 `policies/pi0_family/client.py:97-102`，Tabero 见 `benchmarks/openpi/openpi_inference_client.py:226`、`:802`）。相机本身仍按配置的默认分辨率渲染和录制：RoboLab 的场景相机和腕部相机都是 1280×720；Tabero 的场景相机是 512×512，腕部相机是 1280×720。触觉拼图按 TacHarness 的原始输出发送，不缩放。

### 2.1 TacHarness 现有的触觉输出 `[代码]`

路径相对于 `TacHarness/src/tacumi/`。

| 项目 | 现有行为 | 位置 |
|---|---|---|
| 每块垫每帧的输出 | `depth`、`indent`（压深，米）、`deformation`、`shear`、`mask`、`stats` | `tactile_runtime.py` 的 `TacumiTactileRuntime.step` |
| 拼图 | 左右各一行，每行四格：INDENT、DEFORMATION、SHEAR、HSV | `visualization.py` |
| 是否接触 | `stats["has_contact"]`：物理引擎报告有接触，并且触觉深度图上也看得到压痕，两者同时成立 | `tactile_sensor.py` |
| 无接触时 | 所有触觉数组清零，不泄漏"接近但没碰到"的信号 | `tactile_runtime.py` |
| 接触面积 | `stats["active_ratio"]`：被压到的区域占垫面的比例 | `tactile_sensor.py` |
| 压深上限 | `indent_max_m = 0.0015`（1.5mm） | `tactile_runtime.py` |
| 力（Tabero） | 左右手指各自的接触合力大小（牛）；进入接触的阈值 0.01N，退出阈值 0.001N，连续 3 帧低于退出阈值才算松开 | `tabero_rollout.py` 的 `TacumiContactGate` |
| 力（RoboLab） | 只有一个 `gripper__all_objs` 接触传感器，阈值 0.001N，**左右不分开**，两侧共用同一个结果 | `providers/RoboLab/robolab/eval/episode.py:126-138` |

- TacHarness 里**没有**现成的"触觉摘要文字"，所以给 GPT 的只有拼图。上表里的力、面积等数值照常逐帧录进轨迹，但不发给 GPT。

### 2.2 物体坐标从哪里读 `[代码]`

两个 benchmark 都能直接从仿真里读到物体坐标，不需要新增仿真功能。现有代码只把它用于成功判定，没有任何地方把它发给策略，所以"读出来、换坐标系、写进提示词"这一小段要我们自己写。

| | 读取方式 | 位置 |
|---|---|---|
| RoboLab | `get_world(env).get_pose(物体名, env_id=0)`，返回位置和四元数；物体名列表是 `get_world(env).objects`。另有 `get_centroid`（几何中心）、`get_dimensions`（尺寸） | `TacHarness/src/providers/RoboLab/robolab/core/world/world_state.py:492`、`:678`、`:604` |
| Tabero | 没有封装，直接读 `env.scene[物体名].data.root_pos_w`，成功判定就是这么读的；成功判定用到的物体名来自任务 JSON 的 `goals`（`ref_obj`、`target`），全部物体的名字在任务 JSON 的 `objects` 里 | `TacHarness/src/providers/LIBERO/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/terminations.py:133-135` |

- **坐标系不同，必须换算。** 上面读到的是世界坐标系（RoboLab 的 `get_pose` 默认减去了环境原点），而第 3 节的动作用的是机器人基座坐标系。发给 GPT 之前要用机器人基座的位姿把物体坐标换到基座坐标系。
- 读到的都是物体的**根坐标**。RoboLab 另有现成的几何中心可用，Tabero 没有。

---

## 3. 策略的输出：离散动作 `[设计]`

**已定（2026-10-11）：每次调用只输出一个离散动作，即 Show-Harness 的 meta action。** 本版用其中的 6 个平移和 2 个夹爪动作。JEMM 给出 8 个动作的概率再抽样（1.1 节）；`--policy gpt` 时 GPT 输出这样的 JSON：

```json
{"decision": "MV_FWD"}
```

| 动作 | 解释器做什么 |
|---|---|
| `MV_FWD` / `MV_BACK` | 沿 +x / −x 移动 2cm |
| `MV_RIGHT` / `MV_LEFT` | 沿 +y / −y 移动 2cm |
| `MV_UP` / `MV_DOWN` | 沿 +z / −z 移动 2cm |
| `GRASP` | 先等手臂停稳，再闭合夹爪 |
| `RELEASE` | 先等手臂停稳，再张开夹爪 |

- 动作名和方向定义都沿用 Show-Harness（`Show-Harness/core/action_units.py`、`scripts/trajectory/real2sim/atomic_tokenizer.py` 的 `MOVE_DIRS`），方向是机器人基座坐标系的轴，和第 2 节的机器人状态、物体坐标是同一个坐标系。`[代码]`
- 在场景相机里看（实看 `logs/probe_wp*_agentview_cam.png`、`logs/rprobe_wp*_over_shoulder_left_camera.png`）：Tabero 的 +y 在画面里向右、+x 朝向相机，和 Show-Harness 的约定一致；RoboLab 的肩部相机里 +x 向左、+y 朝向相机。提示词里分别写明。
- **没有 `DONE`**：回合什么时候结束只由官方成功判定和超时决定。
- **没有转腕动作**：整个回合保持开局时的夹爪朝向。
- `[待对齐]` 要不要加 Show-Harness 的旋转动作：`ROTATE_CW/CCW`（腕部视角里绕竖直轴转），或 6 个 `RT_*`（绕世界轴转，每次 10°）。只有平移时，长条物体和开抽屉、拧炉灶这类任务大概率做不了。
- 夹爪的数值仍按各环境自己的约定，由解释器翻译（`gpt_rollout/tabero_run.py`、`robolab_run.py` 的 `gripper_open` / `gripper_close`）：RoboLab 上 0 张开、1 闭合；Tabero 上 1 张开、0 闭合（2026-10-04 实跑确认，`binary_joint_actions.py:196-206`）。`[代码]`
- 输出里没有文字，所以不存在物体坐标通过思考文字漏进训练数据的问题。

---

## 4. 脚手架的执行规则 `[设计]` `[代码]`

- **解释器就是 Show-Harness 的 `AtomicExec`**，直接从 `Show-Harness/` 加载，参数用它的默认值（`gpt_rollout/interpreter.py`）。它怎么把一个离散动作变成每一帧的连续 action，见 4.1 节。`[代码]`
- 环境配置不变：Tabero 用 `Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0`，RoboLab 用 `DroidIKActionCfg`，两者都接收 8 个数的绝对位姿。逐帧数据里的动作就是这 8 个数，每帧一行；每帧属于哪个动作记在 `gpt_calls/calls.jsonl`（`first_frame`、`frame_count`）。离散动作另存为逐决策数据，见第 7 节。
- 回合可能在一个动作的中途结束（成功或超时），这时立即停止。
- GPT 思考期间**仿真暂停**等待，不占回合时间。
- ~~每个回合最多 200 次决策（Show-Harness LIBERO runner 的默认值）。~~ 已去掉（2026-10-24，见 0.1 节），只在关掉超时的测量模式下保留。
- **已定（2026-10-04，沿用）：** GPT 输出不合法时（JSON 解析不了、不是词表里的动作），用同一份观测重新调用；重新调用超过 3 次仍不合法，这次 rollout 记为失败。重新调用期间仿真不前进。只适用于 `--policy gpt`，JEMM 不会有不合法输出（1.1 节）。
- JSON 对象前后多出来的字符会被忽略。实测网关有时会在回复最前面加一个 `_`（`_{"decision":"MV_FWD"}`），temperature 为 0 时同一份观测重新调用得到的结果完全相同，不忽略的话回合在第 0 步就会被记为失败。`[代码]` `gpt_rollout/actions.py`
- GPT-6 astra 不接受 `temperature` 参数，传了就返回 400。`configs/llm.yaml` 里不写它，`gpt_rollout/llm.py` 只在配置了时才发送（2026-10-25）。`[代码]`

原仓库里与执行相关的事实：

| 项目 | RoboLab 副本 | Tabero 副本 |
|---|---|---|
| 环境步进频率 | **15Hz**：`dt=1/120`、`decimation=8`，见 `TacHarness/src/providers/RoboLab/robolab/core/environments/factory.py:126-128` | **20Hz**：`dt=1/60`、`decimation=3`，见 `TacHarness/src/providers/LIBERO/source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py:759-762` |
| 单个回合的时长上限 | 每个任务自己定 `episode_length_s`，例如 BananaInBowl 是 50 秒，默认 600 秒 | **8 秒**（libero_10 是 16 秒），也就是 160（或 320）步，见同一文件 `:763` |
| 原评测脚本自己的上限 | — | 关掉超时和物体掉落终止，改用 30 次推理 × 10 步 = 300 步（`benchmarks/openpi/openpi_inference_client.py:257-258`，`benchmarks/common/closedloop_policy_inference.py:62-72`） |

### 4.1 解释器：一个离散动作怎么变成每一帧的连续 action `[代码]`

**连续 action 不是事后算出来的，而是解释器每一步发给 `env.step` 的指令，发的同时原样录下来。** 仿真器听不懂 `MV_LEFT`，每个 env 步（Tabero 1/20 秒、RoboLab 1/15 秒）只接受 8 个数 `[x, y, z, qw, qx, qy, qz, 夹爪]`：手要去的位置、手的朝向、夹爪开合。手臂有惯性，一步走不到，所以一个离散动作要发好几步，每一步就是逐帧数据里的一帧。

计算分两层：

**第一层：我们的后端拼 8 个数**（`gpt_rollout/interpreter.py:65-73` 的 `apply_delta`）。`AtomicExec` 每一步给出一个想走的位移 `delta`：

```text
delta  = 把每个轴截到 ±2cm 以内
action = [ 手的当前位置 + delta   (x, y, z)              ← 每步从仿真重新读：两块触觉垫之间的中心，机器人基座坐标系
           回合开始时的朝向       (qw, qx, qy, qz)       ← 回合开始读一次，之后不变，所以手不转
           夹爪值                 (1 个数) ]             ← 张开：Tabero 1 / RoboLab 0；闭合：Tabero 0 / RoboLab 1
```

**第二层：`AtomicExec` 决定每一步的 `delta`**（`Show-Harness/scripts/trajectory/real2sim/atomic_tokenizer.py`）：

| 动作 | 过程 | 位置 |
|---|---|---|
| `MV_*` | 1. 定目标（只算一次）：把手的当前位置对齐到 2cm 网格上最近的格点（网格原点是回合开始时手的位置；偏离格点超过 5mm 就不对齐），再沿方向加 2cm。<br>2. 每一步 `delta = 目标 − 当前位置`；离目标小于 1mm 就停，最多 16 步。 | `:380`、`:368` |
| `GRASP` | 1. 等停稳：每步发 `delta = 0`（待在原地），直到手一步移动不到 0.3mm，最多 40 步。<br>2. 夹爪状态改成闭合。<br>3. 再发 10 步 `delta = 0`，让手指合拢。 | `:424`、`:449` |
| `RELEASE` | 同 `GRASP`，只是第 2 步改成张开。 | `:458` |

- 夹爪状态存在 `AtomicExec` 里，之后每个 `MV_*` 的每一步都带着它，所以 `GRASP` 之后搬运时一直是闭合的，直到 `RELEASE`。夹爪只有开和合，不能半开、不能调力。
- 先停稳再动夹爪是 Show-Harness 的规定：手在动时合手指会把物体拖偏；手在动时松手，剩余速度会传给物体（他们在 RoboLab 的魔方任务里实测过魔方被甩出 13.7cm）。等停稳的这几步也录成帧、计入步数。
- 网格对齐让误差不累积：每个动作少走一点的话，下降十几格后手会停在物体上方抓不到（Show-Harness 的 LIBERO runner 也是这样做的）。

**例子**（两倍时长试跑，Tabero `libero_object` task 0，第一个决策 `MV_LEFT`）：

| 帧 | 发出去的 `action` | 执行后手的 y | 离目标 |
|---|---|---|---|
| 0 | `[0.4850, −0.0255, 0.2208, −0.0012, 0.9998, −0.0124, 0.0146, 1.0]` | −0.0090 | 1.65cm |
| 1 | 同上 | −0.0155 | 1.00cm |
| 2 | 同上 | −0.0209 | 0.46cm |
| 3 | 同上 | −0.0242 | 0.12cm |
| 4 | 同上 | −0.0258 | 0.03cm，小于 1mm，停 |

手开始在 y = −0.0055，目标是 −0.0255。因为 `action` 的位置 = 当前位置 + (目标 − 当前位置) = 目标，只要差距不超过 2cm，每一帧发的都是目标点本身，所以一个 `MV_*` 内部的 `action` 是常数，手实际怎么走过去记录在 `observation.state` 里。同一回合的 `GRASP` 展开成 11 帧：1 帧等停稳（夹爪值 1），10 帧原地保持闭合（夹爪值 0）。

**要训输出增量的连续策略时**，可以从数据里离线算，不用重新采：

- 指令增量 = 本帧 `action` 的位置 − 上一帧手的位置。这就是第二层的 `delta`，部署时加到当前位置上就还原出绝对目标。第 0 帧的"上一帧位置"用逐决策数据 `actions.jsonl` 第 0 行的 `ee_pose`。
- 实际位移 = 相邻两帧手的位置之差。它是结果，比指令小得多（上表第 0 帧：指令 −2.00cm，实际 −0.35cm），不适合当动作标签。
- 夹爪不做差，用绝对的开或合；朝向不变，增量为 0。如果要在相对 IK 的环境里部署（RoboLab 的 `DroidRelIKActionCfg`、Tabero 的相对模式），动作还有缩放系数，跟踪特性也不同，不能直接套用。

发出去之后的事不归解释器管：环境自己的 IK 把目标位姿换算成 7 个关节的目标角度，物理仿真驱动关节走一个控制周期。

### 4.2 实测：一个动作要多少步 `[代码]`

GPT 端点暂时连不上（2026-10-11），先用脚本代替 GPT（`gpt_rollout/scripted.py`：按真值坐标，每次选误差最大的那个轴走一步），把解释器在仿真里跑通。脚本策略很简单，这里的数字只说明解释器每个动作的代价，**不代表 GPT 需要的步数**。

| | Tabero libero_object task 7（牛奶放进篮子，种子 0） | RoboLab BananaInBowlTask（种子 0） |
|---|---|---|
| 结果 | 成功，但在第 1470 步，原上限 160 步 | 成功，第 727 步，原上限 750 步 |
| 动作个数 | 134 | 72 |
| 空中的 2cm 移动 | 每个 5–7 步 | 每个约 9 步 |
| 碰到物体或桌面时 | 常常用满 16 步，被推偏，实际位移 1.6–4.3cm | 下降碰到桌面时用满 16 步，实际只走了 0.2–1.5cm |
| `GRASP` | 约 11 步 | 14 步 |
| `RELEASE` | 平均 33.5 步（要等手臂停稳） | 4 步 |

- Tabero 那条：牛奶在搬运中两次滑脱；第三次抓起后，被夹着的牛奶碰到篮子边缘，按原版规则（水平距离 < 0.2m、高度差 < 0.2m、两者有接触，连续 8 步）判为成功，这时牛奶还在夹爪里。规则是任务自己的，不改。
- 按空中每个动作 5–9 步算，一次不出错的抓取放置（约 55–70 个动作）就要几百步。Tabero 的 160 步肯定不够，RoboLab 的原值勉强够。
- 一个动作要几步，主要由 `AtomicExec` 的"闭环走到 1mm 以内"决定。Show-Harness 在 RoboLab 上部署时用的是另一种解释器（`interpreters/robolab_atomic_controller.py`：每个动作固定 8 个 env 步、不闭环）。**已定（2026-10-24）：用 `AtomicExec`。** Show-Harness 在 RoboLab 上生成训练数据时用的也是它（`Show-Harness/scripts/trajectory/real2sim/robolab/oracle.py:210`），固定 8 步的那个只在部署时用；我们的目的是采训练数据。`[代码]`
- 运行时间：两个 Isaac 进程同时跑，每个回合约 20 分钟（不含 GPT 思考）。逐帧录制每帧约 7MB，RoboLab 这一条就是 5.2G；测量时用 `--no-record` 只保留调用日志和结果。

运行方法：

```bash
scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite libero_object --task-id 7 --num-seeds 1 \
    --output-root outputs/measure --disable-time-out --no-record          # 默认 --policy jemm；--policy scripted 不调模型
scripts/isaac_python.sh -m gpt_rollout.robolab_run --task BananaInBowlTask --num-seeds 1 \
    --output-root outputs/measure --no-record                             # 脚本策略另加 --scripted-object banana --scripted-target bowl
TacHarness/.venv-isaacsim51/bin/python scripts/progress.py outputs/measure     # 每种动作的步数统计
```

### 4.3 实测：JEMM 冒烟测试（2026-10-24）`[代码]`

两个 benchmark 各跑一个任务、一个种子，开启录制，两份数据都写出并读回校验。

| | Tabero libero_object task 7（种子 0） | RoboLab BananaInBowlTask（种子 0） |
|---|---|---|
| 结果 | 失败，160 步超时 | 失败，749 步超时 |
| 决策次数 | 18 | 81 |
| 平均每个决策的 env 步数 | 8.9 | 9.2 |
| JEMM 每次调用的平均耗时 | 1.56 秒 | 1.43 秒 |
| 回合耗时（不含启动和发布） | 96 秒 | 548 秒 |
| 发布逐帧数据 | 约 15 秒 | 约 2 分钟 |
| GPU 0 峰值显存（JEMM 加一个 Isaac 进程） | 38.1GB | 39.4GB |
| 数据大小 | LeRobot 0.7MB，Show-Harness 2.2MB | LeRobot 9.7MB，Show-Harness 13MB |

- **JEMM 两条都没有完成任务。** 所有调用的置信度都低于阈值 0.987。RoboLab 那条平均置信度 0.31，手始终没有低于起始高度（z 从 0.321 到 0.422），动作在各个方向之间来回。
- 一个 Isaac 进程占 7–9GB。GPU 0 上 JEMM 加一个 Isaac 已经到 39GB，放不下第二个 Isaac 进程。
- 这两条是在原时间上限、有 200 次决策上限时跑的。之后按原设置开始的全量采集，`libero_object` task 0 的前 11 个回合也全部超时失败，手离要抓的罐头最近 0.14m，于是暂停并清空了输出。

### 4.4 实测：两倍时长试跑（2026-10-24）`[代码]`

时间上限放大两倍、不限决策次数（0.1 节），Tabero 每个套件的第一个任务加 RoboLab 两个任务，各 1 个种子，开启录制。输出在 `outputs/trial_x2/`（不进仓库），脚本是 `outputs/trial_x2/run.sh`。

| 任务 | 步数 / 放大后上限（原上限） | 决策 | 手离要抓的物体：开始 → 最近 | 手最低 / 物体高度 | `GRASP` | 平均置信度 | 回合用时 |
|---|---|---|---|---|---|---|---|
| Tabero `libero_object` 0 | 320 / 320（160） | 37 | 0.29 → 0.21m | 0.12 / 0.04m | 1 | 0.44 | 191 秒 |
| Tabero `libero_spatial` 0 | 320 / 320（160） | 39 | 0.35 → 0.19m（黑碗） | 0.14 / 0.00m | 6 | 0.37 | 196 秒 |
| Tabero `libero_goal` 0（开抽屉） | 320 / 320（160） | 46 | — | 0.14m | 2 | 0.34 | 210 秒 |
| Tabero `libero_10` 0 | 640 / 640（320） | 67 | 0.22 → 0.17m | 0.16 / 0.06m | 7 | 0.35 | 394 秒 |
| Tabero `libero_tactile` 0（开抽屉） | 320 / 320（160） | 46 | — | 0.20m | 4 | 0.31 | 227 秒 |
| RoboLab BananaInBowlTask | 1499 / 1500（750） | 141 | 0.35 → 0.33m | 0.32 / 0.02m | 17 | 0.31 | 1230 秒 |
| RoboLab GrabABagelTask | 899 / 900（450） | 94 | 0.44 → 0.35m | 0.26 / 0.03m | 17 | 0.38 | 1149 秒 |

- **7 个全部失败，都是超时。** 多出来的时间只是让 JEMM 多转了几圈：手离要抓的物体最近也还差 17cm，从没下降到物体的高度；RoboLab 上基本没往下走。
- **抓取几乎全是空的。** 54 次 `GRASP` 里 52 次合到底（开口约 58mm）。只有 `libero_10` 有两次停在约 68mm，是夹在篮子边上，离要抓的罐头还有 18cm。
- 置信度平均只有 0.31–0.44，远低于 JEMM 的阈值 0.987。
- **结论：瓶颈是 JEMM，不是时间。** 继续采集之前，`[待对齐]` 选一个方向：
  1. 离线诊断：用已经录下的观测换几种写法重新问 JEMM（只给文字、简化说明、只给坐标），看它能不能选对方向，判断问题在提示词还是模型；
  2. 换策略：回到 GPT-6，或者换一个做过操作任务训练的模型；
  3. 接受失败数据，照样采 Tabero（约 7 天，0.2 节）。

- [ ] 核对：每个 env.step 都计入步数，包括 `GRASP` / `RELEASE` 前等手臂停稳的那几步。
- [ ] 核对：GPT 输出不合法时重新调用，超过 3 次记为失败。

### 4.5 实测：GPT-6 astra（2026-10-25）`[代码]`

**已定（2026-10-25）：按 4.4 节的方向 2，换回 GPT-6**，模型是 `openai/gpt-6-astra`（`configs/llm.yaml`）。

Tabero `libero_object` task 0（汤罐放进篮子），种子 0，两倍时长。先不录制跑一次，再在正式采集里录制跑一次：

| | 不录制 | 录制（正式采集的第一条） |
|---|---|---|
| 结果 | 失败，320 步超时 | 失败，320 步超时 |
| 决策 | 33 | 32 |
| 手离汤罐：开始 → 最近 | 0.30 → 0.005m | 0.30 → 0.011m |
| 每次调用 | 平均 3.7 秒，约 2,300 个 prompt token | 平均 3.3 秒 |
| 回合用时 | 188 秒 | 235 秒（含发布） |
| 不合法输出 | 0 | 0 |

- 两次的动作序列几乎一样：先 12 次 `MV_LEFT` 水平对准，再一路 `MV_DOWN` 到汤罐的高度，然后在原地 `GRASP`、`RELEASE` 交替，直到超时。同一任务上 JEMM 最近也还差 21cm（4.4 节）。
- 一个回合约 7.7 万个 token，没有一次不合法输出，也没有网络失败。

**本机的采集** `[代码]`：

- 机器：一张 RTX 5090（32GB），没有 `/data`。GPT-6 不占 GPU，3 个 Isaac 进程共占约 21GB 显存。LeRobot 环境按第 0 节的方法新建在 `.conda/tacharness-lerobot312`。为了腾出暂存空间，删掉了 `models/Qwen3.8-27B`；以后再用 `--policy jemm` 要重新下载。
- 范围（2026-10-25 定）：Tabero 46 个任务和 RoboLab 120 个任务，各 1 个种子（种子 0），四倍时长（2026-10-25 由两倍改为三倍，2026-10-26 改为四倍）。两倍时长下已采的 3 条（`libero_object` task 0–2）在 `outputs/collect_gpt6/`，不并入。
- 脚本 `scripts/collect_parallel.sh <输出目录>`：3 个进程（`WORKERS`）共用一个任务队列，先跑 Tabero，再按时长从长到短跑 RoboLab。暂存按最坏情况预留磁盘：放大后的步数（倍数读 `TIME_LIMIT_SCALE`）× 每帧 7.5MB（`MB_PER_FRAME`，4.2 节实测约 7MB），另留 8GB（`SAFETY_GB`）。放不下的任务先等，单独跑也放不下的记进 `<输出目录>/.sched/skipped`。按 69GB 可用空间算，RoboLab 最长的 8 个任务（200–300 秒，9,000–13,500 步，66–99GB）会被跳过，180 秒的 17 个任务各要 60GB，只能一个一个跑。重新运行脚本就能续跑，已有种子 0 结果的任务不再排队。`TASKS` 只跑列出的任务（Tabero 写成 `套件:编号`，RoboLab 写类名）。
- 试跑（2026-10-25）：先跑 4.4 节那 7 个任务，`TASKS="libero_object:0 libero_spatial:0 libero_goal:0 libero_10:0 libero_tactile:0 BananaInBowlTask GrabABagelTask" scripts/collect_parallel.sh outputs/trial_gpt6_x3`。
- 进度看 `<输出目录>/logs/run.log`；停止时用 SIGKILL 结束 `collect_parallel.sh` 和它启动的 Isaac 进程。

**三倍时长试跑结果（2026-10-25）**：7 个任务各 1 个种子，3 个进程并行，共约 1 小时。输出在 `outputs/trial_gpt6_x3/`。

| 任务 | 结果 | 步数 / 三倍上限（原上限） | 决策 | 手离要抓的物体：开始 → 最近 | `GRASP` | 回合用时 |
|---|---|---|---|---|---|---|
| Tabero `libero_object` 0 | 失败，超时 | 480 / 480（160） | 49 | 0.29 → 0.009m | 11 | 420 秒 |
| Tabero `libero_spatial` 0 | **成功** | 274 / 480（160） | 38 | 0.35 → 0.008m | 1 | 236 秒 |
| Tabero `libero_goal` 0（开抽屉） | 失败，超时 | 480 / 480（160） | 51 | — | 10 | 487 秒 |
| Tabero `libero_10` 0 | 失败，超时 | 960 / 960（320） | 94 | 0.20 → 0.009m | 39 | 1124 秒 |
| Tabero `libero_tactile` 0（开抽屉） | 失败，超时 | 480 / 480（160） | 51 | — | 3 | 738 秒 |
| RoboLab BananaInBowlTask | 失败，超时 | 2249 / 2250（750） | 168 | 0.35 → 0.009m | 22 | 3058 秒 |
| RoboLab GrabABagelTask | **成功** | 543 / 1350（450） | 47 | 0.37 → 0.014m | 2 | 1294 秒 |

- 7 个里成功 2 个，都超过了原上限（`success_within_max_episode_steps` 为假）。同样 7 个任务，JEMM 两倍时长时全部失败，最近也差 17cm（4.4 节）。GPT-6 在每个有目标物体的任务里都把手移到了离物体 1.4cm 以内。共 498 次调用、约 118 万 token，没有不合法输出。
- **主要的失败方式：抓住了却当成没抓住。** `libero_object`、`libero_10` 里，`GRASP` 后开口停在约 70.7mm（没有合到约 58mm 的底，说明夹住了罐子），但下一次决策选的是 `RELEASE`，在同一个位置 `GRASP` / `RELEASE` 交替，直到超时。这时它看到的是：文字里只有 `gripper: closed`（开口宽度不发给策略，第 2 节）；触觉拼图只有一块垫的角上有一点接触，另一块全黑；手比物体的根坐标低约 7mm，只夹到了罐子底边。`libero_tactile` 后半段则是夹爪已经张开还连续 `RELEASE` 20 多次。
- BananaInBowl：先是抓放循环，每抓一次香蕉被推开一点；后来香蕉卡在张开的夹爪上被带走，又连续 57 次 `MV_LEFT`（−y）把它带到 y = −0.84m，而碗在 +y（0.15m）。可能是把动作名里的 LEFT 和肩部相机画面里的"左"弄混了（这台相机里 +x 才是画面左边，3 节）。
- RoboLab 暂存实测每帧 6.8MB（BananaInBowl 2,156 帧 14.7GB），比调度脚本预留用的 7.5MB 小约 10%。
- 循环能一直持续，还因为最近动作那一行只记移动动作：`RELEASE` 之后的输入和 `GRASP` 之前一模一样。**已定（2026-10-25）：`GRASP`、`RELEASE` 也记进这一行**（1 节），之后用同样 7 个任务重新试跑。
- 之后改了提示词里抓取的两条说明（1 节，2026-10-26），没有改开口宽度、物体坐标和 RoboLab 方向的说明。

**改提示词后的试跑结果（2026-10-26）**：同样 7 个任务、三倍时长，最近动作含 `GRASP`/`RELEASE`，提示词是 1 节 2026-10-26 那一版。按 `TASKS` 的顺序从长到短排队，共 35 分钟。输出在 `outputs/trial_gpt6_x3_prompt/`；上一次的输出已删，对比用上表。

| 任务 | 结果 | 步数 / 三倍上限（原上限） | 决策 | `GRASP` | 回合用时 | 上一次 |
|---|---|---|---|---|---|---|
| RoboLab BananaInBowlTask | **成功** | 543 / 2250（750） | 52 | 1 | 19.5 分钟 | 失败 |
| RoboLab GrabABagelTask | **成功** | 331 / 1350（450） | 33 | 1 | 13.9 分钟 | 成功，543 步 |
| Tabero `libero_object` 0 | 失败，超时 | 480 / 480（160） | 48 | 2 | 6.6 分钟 | 失败 |
| Tabero `libero_spatial` 0 | 失败，超时 | 480 / 480（160） | 61 | 5 | 6.8 分钟 | 成功，274 步 |
| Tabero `libero_goal` 0（开抽屉） | 失败，超时 | 480 / 480（160） | 54 | 2 | 10.1 分钟 | 失败 |
| Tabero `libero_10` 0 | 失败，超时 | 960 / 960（320） | 132 | 8 | 20.5 分钟 | 失败 |
| Tabero `libero_tactile` 0（开抽屉） | 失败，超时 | 480 / 480（160） | 53 | 1 | 11.6 分钟 | 失败 |

- 成功仍是 2 个，但两个 RoboLab 任务都在**原上限以内**完成，都只抓了一次：对准、下降、`GRASP`、`MV_UP` 提起，BananaInBowl 再向 +y 移到碗上方放下。上一次 BananaInBowl 往 −y 走的问题这次没出现。
- **抓放循环没有了**：有目标物体的 5 个任务里，`GRASP` 之后的 16 次决策全部是 `MV_UP`（上一次同样这 5 个任务是 73 次里 71 次 `RELEASE`）。两个开抽屉任务的 3 次 `GRASP` 都是空抓（合到 58mm），之后选了 `MV_RIGHT`。
- **新的主要失败方式：提起时罐子从夹爪里滑掉。** `libero_object` 和 `libero_10` 里罐子夹住了（开口约 71mm）、也跟着上升，但每提 2cm 罐子只升约 1.5cm，和手的高度差越拉越大，提到 14–20cm 或开始水平移动时掉回桌面；之后按提示词 `RELEASE` 再抓。`libero_10` 抓起来 7 次，每次都掉。TacClaw 闭合角固定 20°，夹持力不能调（3 节），靠提示词解决不了。
- `libero_spatial` 这次失败：抓的是碗沿（开口 63–70mm），提起 2–3 格碗没跟上来，按提示词 `RELEASE` 再试，5 次都一样。上一次成功的那条碗其实也没被提起来（当时碗高度没变），是被推到了盘子边上满足了判定。GPT-6 不传 temperature，同一开局两次的动作本来就不同。
- 两个开抽屉的任务：到柜子前下降后，在 `MV_LEFT`/`MV_RIGHT` 之间来回十几次，中间夹着很多次夹爪已经张开时的 `RELEASE`。删掉的"不要选和上一个移动相反的动作"那一条原本就是防这种来回的。
- 共 433 次调用、约 104 万 token，没有不合法输出。

### 4.6 实测：Tabero 里罐子为什么从夹爪里滑掉（2026-10-26）`[代码]`

用脚本策略（真值坐标，不调用 GPT）在 `libero_object` task 0、种子 0 上抓汤罐，和 4.5 节同一个开局。`--scripted-grasp-dz` 设抓取高度（手比罐子根坐标高多少），移动是 2cm 一格，只能取到下表这几档。

| 抓取高度 | 左垫触觉 | 右垫触觉 | 往上提 |
|---|---|---|---|
| −0.7cm | 接触 | 无 | 手升 15.9cm，罐子 12.1cm，一路下滑 |
| +1.3cm（GPT-6 的高度） | 接触 | 无 | 手升 13.8cm，罐子 8.3cm，之后掉落 |
| +3.3cm | 接触 | 无 | 罐子几乎不动 |
| +5.3cm | 接触 | 接触 | 罐子完全不动 |
| +7.3cm | 无 | 无 | 高过罐顶，空抓（合到 58mm） |

- **手指最大力不是原因。** `TABERO_TACUMI_GRIPPER_EFFORT_LIMIT` 从 20 改成 50（确认 PhysX 里的值变了），三个高度的结果和 20 时完全一样。
- **决定滑不滑的是触觉垫的动摩擦。** 抓取高度 +1.3cm：只把静摩擦改成 5，罐子照样滑；静 5、动 4 时罐子和手一起升了 13.8cm，夹着搬到离篮子约 6 步时才因为时间上限结束；把整根手指（另外 3 个碰撞体，原来是 PhysX 默认的 0.5）也改成 5/4，结果和只改垫子完全一样。
- 于是定为只改触觉垫的摩擦系数（0.1 节的例外、5.1 节）。实现：`configs/tacclaw_pad_friction.usda` 叠在 TacHarness 的 `assets/robots/franka_tacumi_dm25pd04.usda` 上，只覆盖 `tactile_grip_material` 的两个摩擦系数（逐项比较合成后的 USD，2,214 个属性里只有这两个不同）；`scripts/isaac_python.sh` 把 `TACUMI_FRANKA_USD_PATH` 指向它，Tabero 和 RoboLab 都从这个路径读机器人。TacHarness 自己的测试 `test_tacumi_droid_source.py:396` 检查的是原文件里的 `staticFriction = 2`，原文件没动。`[代码]`
- `[待对齐]` **右垫的触觉漏报。** 开口停在 71mm 说明两根手指都被罐子挡住（右手指若没碰到会合到 20°），视频里罐子也被夹起来过；改手指其他碰撞体的摩擦对结果没有影响，说明右侧是用垫子夹住的。但右垫的触觉一直报无接触（Tabero 里两块垫同时有接触的帧只占 76/2,880）。策略看到的触觉拼图和录下的触觉数据右侧可能是错的。可能和 5.5 节"5.1 上力传感器绑到胶面会出问题"有关，还没查。

**改摩擦系数后的试跑结果（2026-10-26）**：同样 7 个任务、三倍时长、提示词同 1 节 2026-10-26 那一版，只多了触觉垫摩擦系数 5/4。输出在 `outputs/trial_gpt6_x3_friction/`，共 31 分钟。

| 任务 | 结果 | 步数 / 三倍上限（原上限） | 决策 | `GRASP` | 4.5 节第二次 |
|---|---|---|---|---|---|
| RoboLab BananaInBowlTask | **成功** | 520 / 2250（750） | 52 | 1 | 成功，543 步 |
| RoboLab GrabABagelTask | **成功** | 294 / 1350（450） | 31 | 1 | 成功，331 步 |
| Tabero `libero_object` 0 | **成功** | 471 / 480（160） | 54 | 1 | 失败，汤罐滑落 |
| Tabero `libero_spatial` 0 | **成功** | 336 / 480（160） | 45 | 1 | 失败，碗没跟上来 |
| Tabero `libero_10` 0 | **成功** | 738 / 960（320） | 100 | 2 | 失败，番茄酱 7 次滑落 |
| Tabero `libero_goal` 0（开抽屉） | 失败，超时 | 480 / 480（160） | 53 | 3 | 失败 |
| Tabero `libero_tactile` 0（开抽屉） | 失败，超时 | 480 / 480（160） | 55 | 5 | 失败 |

- 5/7 成功。有目标物体的 5 个任务都只抓了一次（`libero_10` 两个物体各一次），提起和搬运途中没有滑落。`libero_spatial` 这次是把碗提起来搬过去的，不是推过去。
- 只有两个 RoboLab 任务在原上限内成功；三个 Tabero 成功都超过了原上限（`libero_object` 在 471 步，离三倍上限只差 9 步）。
- `libero_object` 和 `libero_10` 判定成功时，最后一次决策的观测里罐子还在夹爪里、在篮子上方，按任务自己的规则（水平 < 0.2m、高度差 < 0.2m、有接触）判为成功，和 4.2 节牛奶那条一样。
- 两个开抽屉任务照旧在柜子前左右来回、空抓，摩擦系数帮不上（只有平移动作，3 节）。
- 共 390 次调用、约 93 万 token，没有不合法输出。

**四倍时长的试跑结果（2026-10-26）**：和上面完全相同，只把时间上限从三倍改成四倍（0.1 节）。输出在 `outputs/trial_gpt6_x4/`，共 36 分钟。

| 任务 | 结果 | 步数 / 四倍上限（原上限） | 决策 | `GRASP` | 三倍时（上表） |
|---|---|---|---|---|---|
| RoboLab BananaInBowlTask | **成功** | 529 / 3000（750） | 53 | 1 | 成功，520 步 |
| RoboLab GrabABagelTask | **成功** | 313 / 1800（450） | 32 | 1 | 成功，294 步 |
| Tabero `libero_object` 0 | **成功** | 423 / 640（160） | 51 | 1 | 成功，471 步 |
| Tabero `libero_spatial` 0 | **成功** | 263 / 640（160） | 36 | 1 | 成功，336 步 |
| Tabero `libero_tactile` 0（开抽屉） | **成功** | 214 / 640（160） | 27 | 1 | 失败 |
| Tabero `libero_10` 0 | 失败，超时 | 1280 / 1280（320） | 133 | 8 | 成功，738 步 |
| Tabero `libero_goal` 0（开抽屉） | 失败，超时 | 640 / 640（160） | 61 | 4 | 失败 |

- 成功的 5 个都在三倍上限（480 步，RoboLab 各自三倍）以内结束，**这次的成败变化都不是时间带来的**，是 GPT-6 每次输出不同。回合一满足成功判定就立即结束，所以时间上限放宽不会改变"罐子还在夹爪里就判成功"（`libero_object` 最后一次观测里罐子仍在手里、在篮子上方）。
- `libero_tactile` 第一次成功：在把手高度 `GRASP`（开口停在 68mm，夹住了把手），连续 3 次 `MV_RIGHT` 把抽屉拉开。
- `libero_10` 失败的原因是**搬运时撞到篮子**：番茄酱夹得很稳，提到约 17cm 高后水平移向篮子，途中篮子被推动（x 0.521 → 0.542），番茄酱被撞落在篮子外。之后几次重抓有空抓（开口 78–79mm）也有提起后没跟上（按提示词 `RELEASE` 重试），超时前汤罐一直没动。
- `libero_goal` 照旧在柜子前左右来回，4 次 `GRASP` 都合到 60–63mm，没夹住把手。
- 共 393 次调用、约 94 万 token，没有不合法输出。

---

## 5. TacClaw 的实现事实 `[代码]`

### 5.1 统一定义：`TacHarness/src/tacumi/embodiment.py`

| 项目 | 值 |
|---|---|
| embodiment ID | `tacumi_dm25pd04_v1`；与 Franka 组合时的 robot_type 是 `franka_tacumi_dm25pd04` |
| Franka + TacClaw 的 USD | `assets/robots/franka_tacumi_dm25pd04.usda`；Tabero 用 `assets/robots/isaaclab_franka_tacumi_dm25pd04.usda` |
| 夹爪关节 | `Link4_joint`、`Link5_joint`，都是转动关节 |
| 张开 / 闭合的关节角 | 张开 `[0, 0]`；闭合 `[+20°, -20°]`（`TACUMI_CLOSE_ANGLE_RAD`） |
| 策略 TCP | `tacumi-contact-center-v1`，在 `base_link` 上 `(0, 0, 0.104275)` m 处，是两块触觉接触面之间的中心 |
| TCP 到接触中心的偏移 | `(-0.000194, -0.000373, 0.010483)` m |
| 法兰安装点（base_link 坐标系） | `(-0.010346, 0, -0.064711)` m |
| 触觉左右映射 | left = `Link5`，right = `Link4` |
| 腕部相机 | prim 路径 `.../Gripper/Tacumi/Link3/wrist_cam`，OpenCV 鱼眼模型，渲染 1280×720，四元数顺序 `wxyz`，偏移约定 `ros` |
| 夹持宽度 | 两块胶面在关节角为 0 时相距 79.7mm（`TacHarness/skills/tacharness-retarget-corpus/SKILL.md`）；闭合到 20° 对应约 58mm 宽的物体（`embodiment.py` 注释） |
| 摩擦系数 | 触觉垫的接触面绑 `tactile_grip_material`：原值静 2、动 1.5，合并方式 `max`（`assets/robots/franka_tacumi_dm25pd04.usda:63-69`）；手指其余碰撞体用 PhysX 默认 0.5。**已定（2026-10-26）：触觉垫改成静 5、动 4**，见 4.6 节 |

策略 TCP 和厂商相机标定里叫 "TCP" 的 `base_link` 是**两个不同的坐标系**，不能混用。

- [ ] 核对：动作里 `x, y, z` 对应的点就是 `tacumi-contact-center-v1`（已定，见第 3 节）。
- [ ] 核对：任务物体中最窄的水平截面比 79.7mm 宽的，已经列出来。这些物体用 TacClaw 夹不住。

### 5.2 RoboLab 副本（`TacHarness/src/providers/RoboLab/`）

| 项目 | 事实 | 位置 |
|---|---|---|
| 切换方式 | 环境变量 `ROBOLAB_DROID_EMBODIMENT`，默认 `tacumi`，可选 `robotiq` | `robolab/registrations/droid/auto_env_registrations_jointpos.py:55` |
| 默认相机 | `WRIST_LEFT` | 同一文件 `:67-76` |
| 机器人 USD | 可以用 `TACUMI_FRANKA_USD_PATH` 覆盖 | `robolab/robots/droid.py:92-106` |
| 闭合角 | 可以用 `ROBOLAB_TACUMI_CLOSE_ANGLE_RAD` 覆盖，默认 20° | `robolab/robots/droid.py:73-78` |
| 动作配置 | `DroidJointPositionActionCfg`（8 维）、`DroidIKActionCfg`（绝对位姿，8 维）、`DroidRelIKActionCfg`（相对位姿，7 维） | `robolab/robots/droid.py:467-546` |
| 坐标约定（上游 RoboLab） | 笛卡尔动作表达在机器人根坐标系；四元数 `(w,x,y,z)`；夹爪值 `> 0.5` 表示闭合 | `RoboLab/robolab/robots/README.md`、`RoboLab/docs/robots.md` |
| **接触夹爪** | 只指向**右指 `Link4`** | `robolab/robots/droid.py:168-171` |
| 夹爪闭合判定 | `gripper_fully_closed` 已经针对 `Link4_joint` / `Link5_joint` 改过 | `robolab/core/task/conditionals.py:1123` 起 |
| 触觉 | `tacumi.tactile_runtime.TacumiTactileRuntime`，配置通过 `TACUMI_TACTILE_CONFIG` 指定 | `robolab/eval/episode.py:82-85` |

- **已由第 3 节的输出格式决定：** 用绝对 IK（`DroidIKActionCfg`），它接收的正是 `x, y, z, qw, qx, qy, qz` 加夹爪这 8 个数。
- [ ] 核对：接触夹爪只有 `Link4`。这会影响第 6.1 节"已松手"的判定：物体只碰到 `Link5` 时，也会被判为已松手。

### 5.3 Tabero 副本（`TacHarness/src/providers/LIBERO/`）

已注册的 TacClaw 环境（`source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/__init__.py`）：

| 环境 ID | 配置类 |
|---|---|
| `Isaac-Libero-Tacumi-Replay-Camera-v0` | `JointPositionTacumiLiberoCameraEnvCfg` |
| `Isaac-Libero-Tacumi-Replay-Camera-ContactForce-v0` | `JointPositionTacumiContactForceLiberoCameraEnvCfg` |
| `Isaac-Libero-Tacumi-IK-Camera-v0` | `IKTacumiLiberoCameraEnvCfg` |
| `Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0` | `IKTacumiContactForceLiberoCameraEnvCfg` |

**没有** TacClaw 版的 `Hybrid-*`（力位混合）和 `Tactile-*` 环境。这两类环境只有 Panda 手爪版本。

以下配置都在 `franka_tacumi_libero_env_cfg.py`：

| 项目 | 事实 |
|---|---|
| 夹爪动作 | `AbsBinaryJointPositionActionCfg`，`threshold=0.5`，张开为 0，闭合到 ±闭合角。相对 IK 模式下改用 `BinaryJointPositionActionCfg` |
| 闭合角 | 可以用 `TABERO_TACUMI_CLOSE_ANGLE_RAD` 覆盖，其次读 `ROBOLAB_TACUMI_CLOSE_ANGLE_RAD`，默认 20° |
| 末端坐标系 | 在 `panda_link0` 坐标系下，取 `Tacumi/base_link` 加上 TCP 偏移 |
| IK 动作 | 控制 `base_link`，`body_offset` 等于 TCP 偏移；设 `USE_RELATIVE_MODE=true` 时为相对模式，缩放系数 `(0.05, 0.05, 0.05, 0.5, 0.5, 0.5)` |
| 抓取接触传感器 | `contact_grasp_<obj>` 只绑在**右指 `Link4`**（IsaacLab 的过滤接触传感器每个环境只支持一个刚体） |
| 手指力观测 | 把 PhysX 按名字排序的输出重新排成 left/right 顺序（`TACUMI_POLICY_FINGER_INDICES`） |
| 腕部相机 | `Tacumi/Link3/wrist_cam`，鱼眼模型，1280×720 |
| 初始臂位姿 | 写死在 `_install_tacumi_embodiment` 的 `default_pose` 中 |

关于力度：夹爪按位置闭合到固定角度，夹持力由闭合角和物体宽度决定。在第 3 节的输出格式下，指令里写 gently 或 firmly 都改变不了夹持力。

- **已定（2026-10-04，按默认）：** 用评测脚本给 TacClaw 的默认环境 `Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0`，动作为绝对位姿（`relative_actions = False`，也是默认值）。见 `benchmarks/openpi/openpi_inference_client.py:379-382`、`:281`。
- Tabero 的 TacClaw 触觉由 `TacHarness/src/tacumi/tabero_rollout.py` 的 `TaberoTacumiRolloutHook` 产生，评测脚本在 `benchmarks/openpi/openpi_inference_client.py:524` 接入它。`[代码]`
- [ ] 核对：Tabero 原版用 GelSight Mini（`gsmini`），TacClaw 对应 `dmtac`，两者的数据字段不同，见 `source/tac_manip/tacbenchhub/sensors/tactile_profile.py`。

### 5.4 相机标定：有冲突

- `TacHarness/skills/tacumi-migrate-end-effector/references/adapter-registry.json` 规定只接受 `tacumi-opencv-fisheye-link3-dag921265c00543e-v4`，其他一律拒绝。
- `embodiment.py` 中 `TACUMI_DEFAULT_WRIST_CLUSTER = "260808_majority"`，对应 `tacumi-opencv-fisheye-wrist-260808majority-v5`，是新生成 USD 的默认值。

- **已定（2026-10-04，按默认）：** 用代码默认值 `TACUMI_DEFAULT_WRIST_CLUSTER = "260808_majority"`（v5），并在每条轨迹的元数据里记录。
- [ ] 核对：实际加载的机器人 USD 里烘焙的确实是 v5 标定。

### 5.5 仿真版本

- RoboLab 默认 Isaac Sim 5.0 / IsaacLab 2.2.0，也可以用 5.1 / 2.3.2.post1（`RoboLab/README.md`）。
- TacClaw 腕部相机要求 Isaac Sim 5.1 及以上（`adapter-registry.json` 中的 `minimum_isaac_sim_version`）。
- Tabero 论文结果基于 5.0。在 5.1 上力传感器绑到胶面会出问题，需要改绑到外壳（minicase），指标会下降（`Tabero/README.md`、`Tabero/docs/REPRODUCTION.md`）。

- [ ] 核对：实际运行用的 Isaac Sim / IsaacLab 版本，并写进每条轨迹的元数据。

---

## 6. 成功判定（原仓库规则）`[代码]`

成败只由下面的官方规则判定，GPT 的输出不参与。

### 6.1 RoboLab

本节的路径都相对于 `TacHarness/src/providers/RoboLab/`。

- **判定来源：** 每个任务类里 `terminations.success` 对应的判定函数（`robolab/core/task/task.py:81`）。
- **结束方式：**
  - 某一步终止了，且不是超时，记为成功；
  - 超时（`time_out`，由 `episode_length_s` 决定），记为失败；
  - 前 2 步内就终止的，视为物理伪影，重置后重新开始（`robolab/core/environments/env.py:95` 附近）。
- **子任务 `subtasks`：** 只输出进度分数，不参与成功判定。

**例子：** `BananaInBowlTask`（`robolab/tasks/benchmark/banana_in_bowl_task.py`）

```python
success = DoneTerm(func=object_in_container, params={
    "object": "banana", "container": "bowl", "gripper_name": "gripper",
    "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True})
episode_length_s = 50
```

这个配置要求三条同时满足（`robolab/core/task/conditionals.py:275`）：
1. **质心在容器内：** 香蕉质心换算到碗的局部坐标系后，落在碗的包围盒内；碗口向上额外放宽一个碗高。
2. **接触容器：** 香蕉和碗之间有接触，接触力阈值 0.1N（`robolab/core/world/world_state.py:691`）。
3. **已经松手：** 香蕉不接触 `gripper` 对应的任何 prim。在 TacClaw 下，`gripper` 只指向 `Link4`（见 5.2 节）。

其他常用的判定函数有 `object_on_top`、`stacked`、`object_upright`、`object_in_container` 等，完整列表见 `RoboLab/docs/task.md` 和 `RoboLab/docs/task_conditionals.md`。**每个任务的判定要看它自己的文件。**

- [ ] 核对：列出本次选用的每个 RoboLab 任务的 success 判定和 `episode_length_s`。

### 6.2 Tabero

- **判定函数：** `libero_goals_reached`（`TacHarness/src/providers/LIBERO/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/terminations.py:103`）。规则写在任务 JSON 的 `goals` 字段中（`benchmarks/datasets/libero/config/<suite>.json`）。
- **`libero_object` 的 10 个任务规则相同。** 以 task 7（牛奶放进篮子）为例：

```json
{"relationship": "in", "ref_obj": "milk_1", "target": "basket_1",
 "xy_threshold": 0.2, "height_threshold": 0.2, "height_diff": 0.0,
 "enable_force_threshold": "True", "force_threshold": 0.05}
```

  意思是以下三条同时满足：
  - 物体根坐标和篮子根坐标的水平距离小于 0.2m；
  - 高度差减去 `height_diff` 后小于 0.2m；
  - 如果场景中存在 `contact_basket_1_milk_1` 传感器，接触力还要大于 0.05N。

- **评测脚本的计数规则**（`benchmarks/openpi/openpi_inference_client.py:1078-1086`）：`native_success` 为真，并且满足以下任一条件时，记为成功：
  - 连续为真达到 `num_success_steps = 8` 步（`:259`）；
  - 或者这一步 env 已经终止。
- **失败的情况：**
  - 物体掉落：物体根坐标高度低于 -0.05m（`franka_libero_env_cfg.py:373`）；
  - 超时：8 秒，libero_10 为 16 秒（`:763`）；
  - 原评测脚本不用上面两条，它把超时和掉落终止都关掉，改用自己的上限 `max_inference_steps = 30`（每次推理执行 `replan_steps = 10` 步，共 300 步，`:257-258`）。**我们用上面两条，不用评测脚本的上限**（已定，见 0.2 节）。

**两份 Tabero 代码的差异：**
- 上游 `Tabero/` 多一个"物体被捏坏"的终止条件：挤压力连续 `consecutive_frames`（默认 4）帧超过阈值就终止（`Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/terminations.py:74`）。
  - 它只在任务 JSON 里配置了 `physics.objects.<obj>.damage` 时才生效；
  - 默认的 `libero_object.json` 没有配置；
  - 部分实验配置有，例如 `benchmarks/datasets/libero/config_profiles/firm_damage_from_rlinf_sft_20k/` 中牛奶的上限是 `max_squeeze_force = 48.9534` N。
- **TacHarness 副本里没有这个终止条件。**
- gentle/firm **不影响成功判定**。原仓库只额外报告平均和最大夹持力等指标。

- [ ] 核对（已由 0.1 节的硬约束决定）：`goals` 里的阈值、掉落高度、超时都用原值。
- **已定（2026-10-04，按默认）：** 以 TacHarness 副本为准，只有它接入了 TacClaw。它没有"物体被捏坏"的终止条件；默认的 `libero_object.json` 本来也没有配置 damage，所以结果和上游相同。
- [ ] 核对："连续 8 步"的规则沿用原值，并且在每个 env step 检查，和原评测脚本一致。

---

## 7. 输出轨迹 `[设计]` `[代码]`

**已定（2026-10-24）：每次 rollout 产出两份训练数据，两份都要。** 成功和失败的回合都记录（见 0.2 节）。

| | 逐帧数据 | 逐决策数据 |
|---|---|---|
| 一条样本 | 一次 `env.step`（RoboLab 1/15 秒，Tabero 1/20 秒） | 一次调用，也就是一个离散动作，对应若干个 env 步 |
| 动作 | 解释器交给 `env.step` 的 8 个数 | 第 3 节 8 个离散动作中的一个 |
| 观测 | 这一步执行之后的画面、触觉和机器人状态（见 7.1 节的帧约定） | 这次调用时的输入：第 2 节第 1–5 项，加最近 5 个动作；**不含**第 6 项物体坐标 |
| 格式 | LeRobot v3，字段沿用 TacHarness 的 rollout 数据集 | Show-Harness 格式，见 7.2 节 |
| 用来训 | 输出连续动作的策略 | 输出离散动作的模型 |

- 两份数据靠决策编号对应。第 j 次决策执行时产生的帧是 `[first_frame, first_frame + frame_count)`，记在 `gpt_calls/calls.jsonl` 里（`gpt_rollout/episode.py:178`），也写进逐决策数据的 `actions.jsonl`。逐帧数据每帧再加一列 `decision_index`，记它属于第几次决策（`scripts/publish_rollout.py`）。`[代码]`
- 两份数据都不含物体真值坐标。
- 现有的相关转换工具 `[代码]`：RoboLab 有 `robolab/core/export/lerobot_exporter.py`（episode 级字段 `success`）；Tabero 有 `benchmarks/common/`（HDF5 和视频转 LeRobot/OpenPI 格式）；TacHarness 有 `src/tacharness/data_engine/`（LeRobot v3 目录和回放）和 `tools/lerobot_v3_rollouts.py`。

### 7.1 逐帧数据

- **录制**沿用 TacHarness 的录制器（已定，2026-10-04）：`TacHarness/src/tacumi/recorder.py` 每帧写一个压缩 NPZ（`:480-481`），其中包括每块垫的原始触觉数组。不另外输出 LIBERO 的 HDF5 形式。`[代码]`
- **帧约定**：第 k 帧存的是第 k 个动作，以及执行它之后的观测。RoboLab 见 `gpt_rollout/robolab_run.py:139`、`:148`；Tabero 见 `TacHarness/src/tacumi/tabero_rollout.py:382` 的 `record_step`。`[代码]`
- **TacHarness 现有的发布工具接不上我们的 rollout**（`TacHarness/tools/lerobot_v3_rollouts.py`）`[代码]`：
  - 只发布 `success` 和 `qualified` 都为真的回合（`:1361`）；
  - 要靠回放流程写出的审核文件（`frontend_ready.json` 等）才能找到回合（`discover_tabero` `:1150`、`discover_robolab` `:1210`），我们的 rollout 没有这些文件；
  - RoboLab 的 `action` 字段名写死为 `joint_target_1..7` 加 `gripper_command`（`:1851-1855`），而我们的动作是位姿。
- **所以用自己的发布步骤** `[代码]`：`scripts/publish_rollout.py`，在 LeRobot 环境里运行（`scripts/lerobot_python.sh`），追加回合的部分直接调用 TacHarness 的 `_append_preencoded_episode`，视频用 PyAV 编成 H.264/yuv420p。要求：
  - 输出标准的 LeRobot v3 目录，字段名和视频编码沿用 TacHarness 的 rollout 数据集（见下表），`action` 的字段名用 `gpt_rollout/actions.py` 的 `ACTION_LAYOUT`；
  - 每帧加一列 `decision_index`；
  - 失败的回合也发布，`success` 为假，`reward` 全为 0；
  - 和 TacHarness 的回放数据集（`outputs/datasets/lerobot_v3/tacumi_rollouts/`）分开存放，因为两边 `action` 的含义不同。
- **每帧的字段**，以 RoboLab 为例，沿用 `tools/lerobot_v3_rollouts.py` 的 `_dataset_features`（`:1845` 起）：

| 字段 | 形状 | 内容 |
|---|---|---|
| `observation.state` | float32[15] | 7 个关节角、夹爪闭合比例、手的位置 xyz、朝向四元数（和 `gpt_rollout/robolab_run.py` 的 `STATE_LAYOUT` 一致） |
| `action` | float32[8] | x, y, z, qw, qx, qy, qz, 夹爪值 |
| `observation.tactile.summary` | float32[12] | 左右两块垫各 6 个数：是否接触、接触面积比例、深度有效比例、最大压深、平均压深、平均剪切 |
| `observation.images.rgb_dual_view` | 视频 | 场景相机和腕部相机左右拼接，H.264 |
| `observation.images.tacumi_tactile_composite` | 视频 | 触觉拼图，H.264 |
| `reward` / `success` | float32[1] / bool[1] | 成功回合的最后一帧 `reward` 为 1，其余为 0；`success` 是整个回合的成败 |
| `seed`、`rollout_id_hash`、`qualified` | 各 1 个数 | 种子、回合 ID 的哈希、是否通过审核（本方案没有单独的审核，`qualified` 与 `success` 相同） |
| `decision_index` | int64[1] | 本方案新增，见上 |
| LeRobot 自带 | 各 1 个数 | `timestamp`、`frame_index`、`episode_index`、`index`、`task_index` |

  Tabero 的 `observation.state` 是录制钩子给的 8 维：位置、轴角朝向、左右手指开度（`TacHarness/src/tacumi/tabero_rollout.py:324-333`）。`[代码]`
- **不进数据集的内容**：每块垫的原始触觉数组（只留摘要和拼图视频）、单独的场景图和腕部图（只留拼接后的视频）、物体坐标。
- **存储** `[代码]`：暂存 NPZ 每帧约 7MB（第 4.2 节实测）。失败的回合会一直跑到超时，如果全部保留暂存要几十 TB，而 `/data` 只剩约 2.5TB。所以每个回合结束后立即发布、校验，然后删掉它的暂存目录。TacHarness 的 Tabero 批量回放也是这样做的（`TacHarness/docs/lerobot_v3_rollouts.md` 的 "Direct rollout retention"）。
- **实现** `[代码]`：回合结束后，runner 调 `gpt_rollout/datasets.py` 的 `export_episode`，先写逐决策数据，再调 `publish_rollout.py`。发布脚本把回合追加进数据集，用 LeRobot 的读取器读回并解码首尾帧，再删掉暂存的 `frames/`。所有步骤都成功后，才往 `results.jsonl` 写这一回合，所以中途出错时断点续跑会重跑这个回合。已发布的回合记在数据集的 `meta/published_rollouts.jsonl`，按回合 ID 去重。多个 Isaac 进程同时发布时，用文件锁串行。
- **输出目录**（`--output-root` 下）：

```text
<benchmark>/.../episode_<种子>/       # 暂存：发布后只剩 gpt_calls/calls.jsonl（含物体坐标的完整调用日志）、summary.json、meta/
<benchmark>/.../results.jsonl         # 每回合一行结果，断点续跑用
lerobot_v3/<benchmark>/               # 逐帧数据，每个 benchmark 一个 LeRobot v3 数据集
showharness/<benchmark>/<任务>/rollout_<种子>/   # 逐决策数据（7.2 节）
```

### 7.2 逐决策数据

- **一条样本是"决策时的观测 + 选中的离散动作"**。Show-Harness 存离散数据也是这样，它的 `RolloutWriter` 存的是动作执行之前的观测（`Show-Harness/scripts/trajectory/real2sim/atomic_tokenizer.py:552`）。`[代码]`
- **标签在决策时直接记录，不从逐帧数据反推。** Show-Harness 实测，事后给连续轨迹打单轴标签时，65–72% 的位移不在单个轴上，不能用（同一文件 `:11-18`）。`[代码]`
- **来源是调用日志**：`gpt_calls/calls.jsonl`，加上每次调用存下的三张图 `call_XXXX_{scene,wrist,tactile}.png`（JEMM 见 `gpt_rollout/jemm.py:138`，GPT 见 `gpt_rollout/episode.py:81`）。`[代码]`
- **决策时的观测必须单独存**，因为逐帧数据里没有它：按 7.1 节的帧约定，第 j 次决策时的观测在第 `first_frame − 1` 帧，而第 0 次决策时的初始观测不在任何一帧里。
- **去掉物体坐标** `[代码]`：`calls.jsonl` 每次调用除了拼好的整段输入文字（`input_text`，含物体坐标），还分项存了一份 `observation`：指令、手的位置和朝向、夹爪状态和开口宽度、物体坐标、最近动作（`gpt_rollout/episode.py:149-158`）。导出逐决策数据时只取其中的机器人状态，不带物体坐标和整段文字。
- 调用日志里的相机图是缩到 224×224 的版本。需要更高分辨率时，从逐帧数据里取第 `first_frame − 1` 帧（第 0 次决策没有对应帧）。

**已定（2026-10-24）：逐决策数据采用 Show-Harness 格式**，也就是它的 `RolloutWriter` 写出的目录结构（`Show-Harness/scripts/trajectory/real2sim/atomic_tokenizer.py:552-630`）。`[代码]` 这样可以直接用 `Show-Harness/train/data_preparation/rollouts_to_alpaca.py` 转成训练数据。写出的代码是 `gpt_rollout/datasets.py` 的 `export_showharness`，它把调用日志里的三张图移过去，`calls.jsonl` 留在原处。`[代码]`

一个回合一个目录，同一任务的回合放在同一个父目录下（转换脚本按父目录展开其中带 `actions.jsonl` 的子目录，`rollouts_to_alpaca.py:326-329`）`[代码]`：

```text
<benchmark>/<任务>/rollout_<种子>/
├── agentview/NNNN.png     # 第 NNNN 次决策时的场景相机图（就是发给模型的那张）
├── wrist/NNNN.png         # 同一时刻的腕部相机图
├── tactile/NNNN.png       # 同一时刻的触觉拼图（本方案新增，Show-Harness 没有触觉）
├── actions.jsonl          # 每次决策一行
└── metadata.json          # 回合信息
```

`actions.jsonl` 每行的字段 `[代码]`：

| 字段 | 来源 | 内容 |
|---|---|---|
| `step` | Show-Harness | 决策编号，等于逐帧数据里的 `decision_index` |
| `token` | Show-Harness | 选中的离散动作 |
| `kind` | Show-Harness | `move`、`grasp` 或 `release`（`atomic_tokenizer.py:143` 的 `token_kind`） |
| `gripper_closed` | Show-Harness | 决策前夹爪是否闭合 |
| `ee_pose` | Show-Harness | 决策前的 `[x, y, z, qw, qx, qy, qz]`，即第 2 节第 5 项的手的位置和朝向（机器人基座坐标系） |
| `gripper_width` | Show-Harness | 决策前两块垫之间的开口宽度，单位米。由两个手指关节角的平均值，在 0° 时 79.7mm、20° 时 58mm 之间线性插值得到，是估计值（`gpt_rollout/sim.py` 的 `gripper_width_m`） |
| `agentview`、`wrist` | Show-Harness | 两张相机图的相对路径 |
| `time` | Show-Harness | 写这一行时的时间戳 |
| `tactile` | 本方案新增 | 触觉拼图的相对路径 |
| `policy_output` | 本方案新增 | 策略这次调用的原始输出 |
| `first_frame`、`frame_count` | 本方案新增 | 这次决策在逐帧数据里对应的帧 |
| `travelled_m` | 本方案新增 | `MV_*` 的实际位移，其他动作为空 |

- 新增的字段不影响转换脚本：它只读 `token`、`agentview`、`wrist`、`gripper_closed`（`rollouts_to_alpaca.py:380-431`）。`[代码]` 触觉图因此不会自动进入训练样本，要用的话需要扩展转换脚本。
- **不含物体坐标**：`actions.jsonl` 里没有物体坐标这一项，也不保存拼好的输入文字。
- `metadata.json`：沿用 Show-Harness 的 `task`、`task_text`、`seed`、`env_id`、`step_m`（参照 `Show-Harness/scripts/trajectory/real2sim/robolab/oracle.py:420` 起），另加 `success`、`end_reason`，以及 `results.jsonl` 里这个回合的整行结果。写入器关闭时会自动补上 `num_steps` 和 `token_counts`。`[代码]`
- 最近动作不用另存：转换脚本按"最近 5 个 `MV_*`，最新的在前"从前面的 `token` 重新算（`rollouts_to_alpaca.py:110`）。`[代码]`
- `[待对齐]` 2026-10-25 起策略看到的最近动作包括 `GRASP`、`RELEASE`（1 节），和转换脚本重算出来的不再一致。策略实际看到的那一行记在 `calls.jsonl` 每次调用的 `observation.recent_moves` 里。用这份数据训练时，要改转换脚本的重算规则，否则训练样本里的历史和采集时策略看到的不同。
- **注意：转换脚本会给每个回合的最后一帧合成一条 `DONE` 样本，不管回合成功与否**（`rollouts_to_alpaca.py:116-118`、`:467-474`）。`[代码]` 本方案的动作里没有 `DONE`（第 3 节），而且失败的回合也会被记录。所以转换时要按 `metadata.json` 的 `success` 排除失败的回合，或者去掉合成的 `DONE` 样本。

- [ ] 核对：两份数据的字段里都没有物体真值坐标。
- [ ] 核对：每帧的 `decision_index` 和 `calls.jsonl` 里的 `first_frame`、`frame_count` 一致。
- [ ] 核对：失败的回合在两份数据里都有，`success` 为假。
- [ ] 核对：逐决策数据能被 `rollouts_to_alpaca.py` 正常读取；转换时失败回合或合成的 `DONE` 已按上面的规则处理。

### 7.3 怎么看输出 `[代码]`

一个输出目录（`--output-root`，例如 `outputs/trial_x2/`）里：

| 目录 | 是什么 | 用途 |
|---|---|---|
| `showharness/<benchmark>/<任务>/rollout_<种子>/` | **离散**：每次决策一条（7.2 节） | 训输出离散动作的模型 |
| `lerobot_v3/<benchmark>/` | **连续**：每个 env 步一帧，每个 benchmark 一个数据集，视频也在这里（7.1 节） | 训输出连续动作的策略 |
| `<benchmark>/…/results.jsonl` | 每个回合一行结果 | 统计成功率、断点续跑 |
| `<benchmark>/…/episode_<种子>/gpt_calls/calls.jsonl` | 每次调用的完整输入输出，**含物体真值坐标** | 调试，不能当训练数据 |

**两份数据怎么对应**：`actions.jsonl` 第 j 行的 `first_frame`、`frame_count` 是它在逐帧数据里的帧范围；逐帧数据每帧的 `decision_index` 等于 `actions.jsonl` 的 `step`。注意观测时刻不同：离散数据存动作执行**之前**的画面，连续数据第 k 帧存第 k 个动作和执行**之后**的画面，所以第 j 次决策看到的画面在连续数据的第 `first_frame − 1` 帧。

**读离散数据**（普通 Python）：

```python
import json
rows = [json.loads(l) for l in open("outputs/trial_x2/showharness/tabero/libero_object/task_000/rollout_000/actions.jsonl")]
print([r["token"] for r in rows])                     # 整个回合的动作序列；图片在同目录的 agentview/ wrist/ tactile/
```

**读连续数据**（在 LeRobot 环境里，图像从 mp4 按时间戳解码）：

```bash
scripts/lerobot_python.sh - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/gpt-rollouts-tabero", root="outputs/trial_x2/lerobot_v3/tabero", episodes=[0], video_backend="pyav")
frame = ds[100]
print(frame["action"], frame["decision_index"], frame["observation.images.rgb_dual_view"].shape)
PY
```

只要数值的话，直接用 pandas 读 `lerobot_v3/<benchmark>/data/chunk-000/file-*.parquet`。哪个回合在哪个文件、视频的起止时间，记在 `meta/episodes/*.parquet`；回合 ID 和 `episode_index` 的对应在 `meta/published_rollouts.jsonl`。现在每个回合由单独一次发布追加，所以每个回合各占一个 parquet 和一个 mp4（`file-000`、`file-001` …）。

**视频**：每个回合两路，都在 `lerobot_v3/<benchmark>/videos/` 下，H.264，帧率等于控制频率（Tabero 20fps、RoboLab 15fps）：

- `observation.images.rgb_dual_view/`：场景相机和腕部鱼眼相机左右拼接（Tabero 1422×512，RoboLab 2560×720）；
- `observation.images.tacumi_tactile_composite/`：触觉拼图（1542×626）。整段没碰到东西时几乎全黑，文件很小。

离散数据里没有视频，只有每次决策那一刻的三张 PNG。

**结果浏览网页**：

```bash
scripts/site/build_site.sh outputs/trial_x2 outputs/trial_x2/site "TacClaw 两倍时长试跑"
scripts/site/build_site.sh outputs/trial_gpt6_x3_prompt outputs/trial_gpt6_x3_prompt/site "GPT-6 三倍时长试跑" \
    scripts/site/cases/gpt6_x3.html                         # 第 4 个参数：案例分析片段（可选）
```

在 `<site_dir>` 下生成 `index.html`、`data.json` 和 `media/`，静态网页，可以直接发布或用任意静态服务器打开。内容按 benchmark 分开，每个回合有：

- 同步播放的两路视频（缩小到一半的网页版）；
- 离散结果：决策时间线和每次决策的策略输出（JEMM 是 8 个概率；GPT-6 是原始输出、调用次数和用时），点一下跳到视频对应位置；
- 连续结果：逐帧目标位置曲线、夹爪和触觉接触状态、手的俯视轨迹、手到要抓的物体的距离，以及每次决策前手和要抓的物体的高度（标出 `GRASP`、`RELEASE`，悬停看开口宽度）；
- 同一个动作在两份数据里的原始记录。

另有"输入格式"一节，展示第一个回合第 0 次决策时策略收到的完整文字和三张图以及返回的结果（JEMM 用它自己的 `mm_messages` 重建；GPT-6 直接取调用日志里的 `system_prompt.txt` 和 `input_text`），和"数据格式"一节。给了第 4 个参数时，页面开头多一节案例分析，内容是 `scripts/site/cases/` 下的 HTML 片段，里面的回合链接会打开下面对应的回合。**几次试跑合成一页**：把各自的网页目录放在同一个目录下（第一个可以直接放在根目录，其余放子目录），在根目录写 `runs.json`，例如 `[{"id": "jemm", "label": "JEMM · 两倍时长", "data": "data.json"}, {"id": "gpt6", "label": "GPT-6 · 三倍时长", "data": "gpt6/data.json"}]`，再把 `index.html` 放到根目录。页面顶部会多一排切换按钮，回合链接写成 `#gpt6-tabero-2`。旧版生成器写出的 `data.json` 没有 `title`、`generated`、`output_root`，可以在 `runs.json` 的那一项里补上。2026-10-26 JEMM 两倍时长试跑和 GPT-6 三倍时长试跑就是这样合在一页的（本地拼装目录 `outputs/site_merged/`）。

脚本在 `scripts/site/`：`build_prompts.py` 读出提示词（JEMM 输出要在 `.venv-jemm` 里跑，GPT 输出用任意 Python 3），`build_site_data.py` 在 LeRobot 环境里读两份数据并剪视频（需要 `ffmpeg`），`index.html` 是页面模板。

---

## 8. 核对总表

**代码与版本**
- [ ] RoboLab 和 Tabero 都用 `TacHarness/src/providers/` 下的副本
- [ ] 脚手架实现后已与本文对照
- [ ] Isaac Sim / IsaacLab 版本已记录
- [ ] 腕部相机标定用 260808 v5（默认值），已记录进元数据

**策略（1.1 节）**
- [ ] 每个回合的结果里记录了 JEMM 的模型和精度（`policy_model`）
- [ ] 动作按 JEMM 的概率采样，随机数生成器用回合的种子初始化
- [ ] int8 相对 bf16 的概率偏移已测，或者确认不测

**采集范围（0.2 节）**
- [ ] Tabero 46 个任务、RoboLab 120 个任务，每个任务 50 个种子各一条
- [ ] Tabero 的开局是环境默认重置的随机采样，没有用示范轨迹
- [ ] 时间上限是原值的四倍（Tabero 640 步，`libero_10` 1,280 步；RoboLab 各任务四倍），没有决策次数上限，超时和掉落终止条件开启
- [ ] RoboLab 开局相同的处理方式已定
- [ ] RoboLab 的采集规模已确认
- [ ] 每条轨迹记录了随机种子

**输入与输出**
- [ ] 输入正好是第 2 节的 6 项，没有多也没有少
- [ ] 触觉输出完全沿用 TacHarness 现有代码，没有改动（见 2.1 节）
- [ ] 策略的输出是第 3 节 8 个动作之一（JEMM 给出概率后抽样；`--policy gpt` 时输出 `{"decision": 动作}`）
- [ ] 提示词里已解释第 2 节列出的全部内容
- [ ] 动作的方向用机器人基座坐标系，移动的点是 `tacumi-contact-center-v1`，实现与此一致

**硬约束（0.1 节）**
- [ ] 没有修改任何任务的 `dt`、`decimation`、成功判定参数、失败终止条件、开局（步数上限的例外见 0.1 节）
- [ ] 每次调用只执行一个动作，执行它的每个 env.step 都计入步数
- [ ] 结果里记录了是否在任务原上限内成功（`success_within_max_episode_steps`）
- [ ] TacClaw 闭合角是默认的 20°，`num_success_steps` 是默认的 8

**执行**
- [ ] 解释器是 Show-Harness 的 `AtomicExec`，用它的默认参数，每一帧的 8 个数按 4.1 节计算
- [ ] `GRASP` / `RELEASE` 先等手臂停稳，再把夹爪指令保持 10 步
- [ ] GPT 输出不合法时重新调用，超过 3 次记为失败（只适用于 `--policy gpt`）

**判定**
- [ ] 每个选用任务的官方 success 判定已列出
- [ ] 知晓并接受 TacClaw 接触夹爪只有 `Link4` 的影响
- [ ] Tabero 用的是 TacHarness 副本和默认环境 `Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0`

**数据**
- [ ] 每次 rollout 都产出逐帧和逐决策两份数据（第 7 节）
- [ ] 逐帧数据用自己的发布步骤写成 LeRobot v3，每个回合发布、校验后删掉暂存
- [ ] 逐决策数据按 Show-Harness 格式写出（7.2 节），每个回合的图片、`actions.jsonl`、`metadata.json` 齐全
- [ ] 两份数据都不含特权信息；成功和失败的回合都记录，用 `success` 区分
- [ ] 每条轨迹的元数据记录了：策略模型和精度、调用次数、耗时、仿真版本、标定版本、环境 ID、时间上限倍数
- [ ] `scripts/site/build_site.sh` 能把输出目录做成结果浏览网页（7.3 节）
