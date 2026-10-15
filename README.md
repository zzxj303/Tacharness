# GPT-6 × TacClaw Rollout：RoboLab / Tabero 实现核对文档

本文用于**核对实现细节**：在 RoboLab 和 Tabero 上用 GPT-6 作为 VLM 收集 rollout 轨迹，末端执行器使用 TacClaw。

写作约定：

- 只记录两类内容：本方案的设计，以及原仓库代码里已有的行为。**不引入新的字段或规则。**
- 每条事实都标出来源：
  - `[设计]`：本方案自己的设计，脚手架代码还没有实现；
  - `[代码]`：本目录下的源码，写明文件路径和行号；
  - `[待对齐]`：设计里还没有定义，或者和原仓库代码冲突，需要人来决定。
- 路径都相对于本目录（`gujie/rollout/`）。行号以 2026-09-30 的代码为准。

---

## 0. 目录与代码基线

| 目录 | 内容 | 与本任务的关系 |
|---|---|---|
| `RoboLab/` | NVIDIA RoboLab 上游仓库 | **没有 TacClaw 支持**，只用于对照 |
| `Tabero/` | Tabero（TacManip）上游仓库 | **没有 TacClaw 支持**，只用于对照 |
| `TacHarness/` | 集成仓库，包含 TacClaw 定义，以及接入了 TacClaw 的 RoboLab/Tabero 副本 | **实际要用的基线** |

- TacClaw 就是 Daimon DM25PD04，代码里叫 `tacumi_dm25pd04_v1`，统一定义在 `TacHarness/src/tacumi/embodiment.py`。`[代码]`
- 接入了 TacClaw 的 RoboLab 在 `TacHarness/src/providers/RoboLab/`，Tabero 在 `TacHarness/src/providers/LIBERO/`。`[代码]`
- 脚手架代码**还没有实现**，由我们自己写。本文里所有标 `[设计]` 的内容是要实现的目标，不是已有代码的行为。`[设计]`

### 上游依赖（不在本仓库里）

本仓库只包含脚手架（`gpt_rollout/`、`scripts/`、`tests/`、`configs/`）。运行前要把下面三个仓库 clone 到本目录下，目录名和 commit 都要一致：

| 目录 | 仓库 | commit |
|---|---|---|
| `TacHarness/` | https://github.com/LV-Robotics-Lab/TacHarness.git | `c5cf373a50d54151f42fc74dbe87e93922f348a6` |
| `Show-Harness/` | https://github.com/showlab/Show-Harness.git | `5a3713b6d90c5a5a8c9461e298e091f7cbdcded4` |
| `RoboLab/` | https://github.com/NVlabs/RoboLab.git | `ad45d4f974725d020f82c2b0d77d78533aeba2b3`（只用它的 `assets/`） |

然后运行 `scripts/setup_env.sh` 安装 Isaac Sim 5.1 运行环境，再把 `configs/llm.example.yaml` 复制成 `configs/llm.yaml`，填入端点和 key（`llm.yaml` 不进仓库）。

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

**例外（已定，2026-10-11）：回合步数上限在收集 rollout 时可以放大。** GPT-6 没有经过训练，原值不够用。放大到多少还没定，先测 GPT 实际需要多少步（第 4.1 节）。表中其他各项仍然不能改。

- **Tabero 的时间很紧。** 160 步里要完成"接近、抓取、搬运、放下"。GPT 思考时仿真暂停，不占这 160 步。第 4.1 节的脚本实测：一次不出错的抓取放置就要几百步。
- **测量时的做法（已定，2026-10-11）：** Tabero 关掉超时终止，让回合跑过 160 步，最多 200 次决策；结果里同时记录"是否在原上限内成功"（`success_within_max_episode_steps`）。前 160 步的轨迹和不关超时时完全一样，所以这一项是准确的。RoboLab 保留原超时。

以下参数不属于任务本身，按代码默认值处理（已定，2026-10-04）：

- TacClaw 的闭合角用代码默认值 20°，不用环境变量覆盖。
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
| Tabero 的步数上限 | 160 步（`libero_10` 为 320 步），即任务配置的原值。超时、物体掉落两个终止条件也按任务配置保留 |

合计：Tabero 46 × 50 = 2,300 次 rollout，RoboLab 120 × 50 = 6,000 次 rollout。**已定（2026-10-04）：只记录成功的轨迹**，和 TacHarness 现有的发布工具一致。失败的 rollout 不保存轨迹数据，只在每个任务的 `results.jsonl` 里留一行结果（种子、结束原因、步数、调用次数和花费），用来统计成功率和断点续跑。

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

> 本小节按旧设计（每次调用输出 4 行动作，N = 4）估算，已不适用。改成每次一个离散动作后，调用次数等于动作个数，见第 4.1 节的实测。

N = 4，按每条轨迹都跑到超时的最坏情况算：

| | 任务数 | 轨迹数 | GPT 调用次数上限 |
|---|---|---|---|
| Tabero（步数上限 160，`libero_10` 为 320） | 46 | 2,300 | 约 112,000 次 |
| RoboLab（各任务 `episode_length_s` 合计 10,940 秒） | 120 | 6,000 | 约 2,050,000 次 |

- RoboLab 的任务时限长（平均约 91 秒，有的任务长达 300 秒），所以调用次数上限远高于 Tabero。一个 300 秒的任务单条轨迹就是 1,125 次调用。
- 成功的轨迹会提前结束，实际次数会少于上限。但即便只有上限的一半，RoboLab 也是百万次量级，单线程跑不完。
- 如果这个量不可接受，有三个办法，可以组合使用：
  - RoboLab 上把 N 调大。例如 N = 15（和 pi05 一致），调用次数降到约四分之一；
  - RoboLab 先只采一部分任务；
  - 多个仿真实例并行采集。
- [ ] `[待对齐]` RoboLab 的采集规模是否接受；不接受的话用上面哪个办法。

### 已知会失败的任务

采全部任务，不做排除，所以下面这些任务大概率失败，产不出轨迹：

- 物体最窄处比 79.7mm 宽的任务，TacClaw 夹不住（见第 5.1 节）；
- 开抽屉、开柜门这类任务，GPT 未必做得了。

核对项：

- [ ] 核对：`libero_tactile` 这个套件能否在 TacClaw 环境下运行。第 5.3 节记录了 TacClaw 没有 `Tactile-*` 环境，这个套件是否依赖它还没有查。
- [ ] 核对：每条轨迹的元数据里记录了随机种子，开局可以复现。
- [ ] 核对：Tabero 的超时和物体掉落终止条件处于开启状态，没有被关掉。

---

## 1. 整体流程 `[设计]`

**GPT-6 按 Show-Harness 的 meta action 方式控制（已定，2026-10-11）：每次调用只输出一个离散动作，由解释器在仿真里把它执行完，再用新的观测问下一次。** 提示词里要向 GPT 解释每个输入是什么、每个动作做什么。

```
循环：
  1. 把当前观测发给 GPT（第 2 节的 6 项输入，加最近 5 个移动动作）
  2. GPT 输出一个动作（第 3 节）
  3. 解释器把这个动作执行完，期间的每个 env.step 逐帧录制（第 4 节）
  4. 回到 1，直到官方规则判定成功，或者超时
```

- **不使用 benchmark 里的示范动作**，只借开局状态和官方成功判定。
- 没有子目标规划，也不用 Show-Harness 的其他插件。GPT 只输出动作，不输出思考文字。
- 每次调用互相独立，不带之前的对话。唯一的记忆是提示词里"最近 5 个移动动作，最新的在前"，这是 Show-Harness 仿真 runner 的 meta-action 提示词本来就有的一行（已定，2026-10-11）。
- 代码：`gpt_rollout/episode.py`（循环）、`gpt_rollout/interpreter.py`（解释器）、`gpt_rollout/prompts.py`（提示词）。`[代码]`

---

## 2. 每次交给 GPT-6 的输入 `[设计]`

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
| 最近的移动动作 | 最近 5 个，最新的在前 |

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

## 3. GPT-6 的输出 `[设计]`

**已定（2026-10-11）：每次调用只输出一个离散动作，即 Show-Harness 的 meta action。** 本版用其中的 6 个平移和 2 个夹爪动作：

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

## 4. 脚手架的执行规则 `[设计]`

- **解释器就是 Show-Harness 的 `AtomicExec`**，直接从 `Show-Harness/` 加载，参数用它的默认值（`gpt_rollout/interpreter.py`）：`[代码]`
  - `MV_*`：每个控制步按"离目标还差多少"下指令（每步最多 2cm），走到离目标 1mm 以内为止，最多 16 个 env 步。目标对齐到以回合起点为原点的 2cm 网格，误差不累积（Show-Harness 的 LIBERO runner 也是这样做的）。
  - `GRASP` / `RELEASE`：先保持不动，直到手臂每步位移小于 0.3mm（最多 40 步），再保持夹爪指令 10 步。
- 每个 env.step 收到的仍是 8 个数的绝对位姿：当前位置加上这一步的位移、回合开始时的朝向、夹爪值。环境配置不变：Tabero 用 `Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0`，RoboLab 用 `DroidIKActionCfg`。录进训练数据的动作就是这 8 个数，每帧一行；每帧属于哪个动作记在 `gpt_calls/calls.jsonl`（`first_frame`、`frame_count`）。
- 回合可能在一个动作的中途结束（成功或超时），这时立即停止。
- GPT 思考期间**仿真暂停**等待，不占回合时间。
- 每个回合最多 200 次决策（Show-Harness LIBERO runner 的默认值）。
- **已定（2026-10-04，沿用）：** GPT 输出不合法时（JSON 解析不了、不是词表里的动作），用同一份观测重新调用；重新调用超过 3 次仍不合法，这次 rollout 记为失败。重新调用期间仿真不前进。
- JSON 对象前后多出来的字符会被忽略。实测网关有时会在回复最前面加一个 `_`（`_{"decision":"MV_FWD"}`），temperature 为 0 时同一份观测重新调用得到的结果完全相同，不忽略的话回合在第 0 步就会被记为失败。`[代码]` `gpt_rollout/actions.py`

原仓库里与执行相关的事实：

| 项目 | RoboLab 副本 | Tabero 副本 |
|---|---|---|
| 环境步进频率 | **15Hz**：`dt=1/120`、`decimation=8`，见 `TacHarness/src/providers/RoboLab/robolab/core/environments/factory.py:126-128` | **20Hz**：`dt=1/60`、`decimation=3`，见 `TacHarness/src/providers/LIBERO/source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py:759-762` |
| 单个回合的时长上限 | 每个任务自己定 `episode_length_s`，例如 BananaInBowl 是 50 秒，默认 600 秒 | **8 秒**（libero_10 是 16 秒），也就是 160（或 320）步，见同一文件 `:763` |
| 原评测脚本自己的上限 | — | 关掉超时和物体掉落终止，改用 30 次推理 × 10 步 = 300 步（`benchmarks/openpi/openpi_inference_client.py:257-258`，`benchmarks/common/closedloop_policy_inference.py:62-72`） |

### 4.1 实测：一个动作要多少步 `[代码]`

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
- 一个动作要几步，主要由 `AtomicExec` 的"闭环走到 1mm 以内"决定。Show-Harness 在 RoboLab 上部署时用的是另一种解释器（`interpreters/robolab_atomic_controller.py`：每个动作固定 8 个 env 步、不闭环）。`[待对齐]` 用哪一种。
- 运行时间：两个 Isaac 进程同时跑，每个回合约 20 分钟（不含 GPT 思考）。逐帧录制每帧约 7MB，RoboLab 这一条就是 5.2G；测量时用 `--no-record` 只保留调用日志和结果。

运行方法：

```bash
scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite libero_object --task-id 7 --num-seeds 1 \
    --output-root outputs/measure --disable-time-out --no-record          # --policy scripted 不调 GPT
scripts/isaac_python.sh -m gpt_rollout.robolab_run --task BananaInBowlTask --num-seeds 1 \
    --output-root outputs/measure --no-record                             # 脚本策略另加 --scripted-object banana --scripted-target bowl
TacHarness/.venv-isaacsim51/bin/python scripts/progress.py outputs/measure     # 每种动作的步数统计
```

- [ ] 核对：每个 env.step 都计入步数，包括 `GRASP` / `RELEASE` 前等手臂停稳的那几步。
- [ ] 核对：GPT 输出不合法时重新调用，超过 3 次记为失败。

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

## 7. 输出轨迹 `[设计]`

每个开局跑完得到一条轨迹，包含：

- 逐帧的画面（场景相机、腕部相机）和触觉；
- 逐帧的机器人状态和动作；
- GPT 每一次调用的完整输入和原始输出，并标注每一帧属于第几次调用；
- 成败标记、耗时和调用花费。**只记录成功的轨迹**（已定，见 0.2 节）；
- 最后统一整理成 **LeRobot v3** 格式；
- 物体精确坐标**不进训练数据**。

现有的相关转换工具 `[代码]`：

- RoboLab：`robolab/core/export/lerobot_exporter.py`（episode 级字段 `success`）；
- Tabero：`benchmarks/common/`，HDF5 和视频转 LeRobot/OpenPI 格式；
- TacHarness：`src/tacharness/data_engine/`（LeRobot v3 目录和回放）、`tools/lerobot_v3_rollouts.py`。

- **已定（2026-10-04）：** 用 TacHarness 现有的保存链路输出 LeRobot v3：先由 `TacHarness/src/tacumi/recorder.py` 写出每条轨迹的逐帧 NPZ 暂存目录，再由 `TacHarness/tools/lerobot_v3_rollouts.py` 发布成 LeRobot v3 数据集（`outputs/datasets/lerobot_v3/tacumi_rollouts/` 下每个来源一个）。不另外输出 LIBERO 的 HDF5 形式。
- 训练数据里每帧存的"动作"是解释器交给 `env.step` 的那 8 个数；每帧属于 GPT 的哪个离散动作，记在 `gpt_calls/calls.jsonl`。`[待对齐]` 训练数据里的动作用这 8 个数，还是用离散动作（Show-Harness 的格式是每个动作一帧），取决于要蒸馏给什么模型。
- [ ] 核对：训练数据的字段里确实没有物体真值坐标。

---

## 8. 核对总表

**代码与版本**
- [ ] RoboLab 和 Tabero 都用 `TacHarness/src/providers/` 下的副本
- [ ] 脚手架实现后已与本文对照
- [ ] Isaac Sim / IsaacLab 版本已记录
- [ ] 腕部相机标定用 260808 v5（默认值），已记录进元数据

**采集范围（0.2 节）**
- [ ] Tabero 46 个任务、RoboLab 120 个任务，每个任务 50 个种子各一条
- [ ] Tabero 的开局是环境默认重置的随机采样，没有用示范轨迹
- [ ] Tabero 步数上限 160（`libero_10` 为 320），超时和掉落终止条件开启
- [ ] RoboLab 开局相同的处理方式已定
- [ ] RoboLab 的采集规模已确认
- [ ] 每条轨迹记录了随机种子

**输入与输出**
- [ ] 输入正好是第 2 节的 6 项，没有多也没有少
- [ ] 触觉输出完全沿用 TacHarness 现有代码，没有改动（见 2.1 节）
- [ ] 输出只有 `{"decision": 动作}`，动作在第 3 节的 8 个之内
- [ ] 提示词里已解释第 2 节列出的全部内容
- [ ] 动作的方向用机器人基座坐标系，移动的点是 `tacumi-contact-center-v1`，实现与此一致

**硬约束（0.1 节）**
- [ ] 没有修改任何任务的 `dt`、`decimation`、成功判定参数、失败终止条件、开局（步数上限的例外见 0.1 节）
- [ ] 每次调用只执行一个动作，执行它的每个 env.step 都计入步数
- [ ] 结果里记录了是否在任务原上限内成功（`success_within_max_episode_steps`）
- [ ] TacClaw 闭合角是默认的 20°，`num_success_steps` 是默认的 8

**执行**
- [ ] GPT 输出不合法时重新调用，超过 3 次记为失败

**判定**
- [ ] 每个选用任务的官方 success 判定已列出
- [ ] 知晓并接受 TacClaw 接触夹爪只有 `Link4` 的影响
- [ ] Tabero 用的是 TacHarness 副本和默认环境 `Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0`

**数据**
- [ ] LeRobot v3 的转换链路已定
- [ ] 训练字段不含特权信息，只记录成功的轨迹
- [ ] 每条轨迹的元数据记录了：GPT 模型版本、调用次数、花费、仿真版本、标定版本、环境 ID
