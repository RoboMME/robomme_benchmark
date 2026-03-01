# RoboMME Benchmark 测试说明

这套测试覆盖了该基准框架下的逻辑断言、动作重放以及数据集录制的正确性，主要可以分为两大板块：**`tests/dataset`** (侧重底层环境与数据集交互) 和 **`tests/lightweight`** (侧重轻量级逻辑分支和单元测试)。

以下是各项测试分别实现的功能及如何运行测试的说明。

## 1. `dataset/` 目录：环境交互及数据集对齐测试

此目录下的测试主要验证底层基于物理引擎的环境调用、强化学习的 Wrapper 观测包装，以及海量数据集录制与重放的尺寸对齐。

*   **`test_obs_config.py`**: 验证 `make_env_for_episode` 中传入的 `include_*` 控制开关（例如开启/关闭前方深度图、腕部相机内参外参等）。测试是否能按需在 `reset()` 和 `step()` 的 `obs` 和 `info` 返回对应字段而不报错。
*   **`test_obs_numpy.py`**: 验证 `DemonstrationWrapper` 处理数据转换的正确性。此测试校验生成的临时数据集中，原生 `obs` 和 `info` 字典包含的值是被正确转化成合规的 NumPy ndarray 数据类型且具备预期的数据结构形状（Shape）。
*   **`test_record_stick.py`**: 验证在使用 `RecordWrapper` 录制演示为 HDF5 数据集格式时，具有特殊轨迹要求（如 `PatternLock`/`RouteStick`）和普通要求（如 `PickXtimes`）的任务，其夹爪状态 (`gripper_state`)、机械臂关节 (`joint_action`) 及末端位姿 (`eef_action`) 的存储维度正确无误。
*   **`test_replay_stick.py`**: 反向验证测试。用于读取测试：检验上一步生成好的特殊或普通数据集交由 `EpisodeDatasetResolver` 重新解析重放时，返回的数据维度和数值能否像录制时一样精准对齐。
*   **`test_eepose_error_handling.py`**: 重度环境交互测试。验证当传入超出机械臂可达范围的末端位姿目标 (`ee_pose` 动作空间) 时，`DemonstrationWrapper` 能够优雅地捕获到底层物理引擎或 IK 求解的报错，并通过返回 `info["status"] = "error"` 上报异常信息以避免仿真程序崩溃。
*   **`test_heavy_replay_4_modes.py`**: 基于视频拼接等复杂任务 (`VideoUnmaskSwap`) 的轨迹重放。针对四种动作空间 (`joint_angle`, `ee_pose`, `waypoint`, `multi_choice`)，验证其能否正确回放临时数据集，且行为终止/截断状态流转保持一致。
*   **`test_route_stick_waypoint_boundary.py`**: 特有路线类任务验证。确保演示生成的数据从离线示范过渡到在线交互 (Demo -> Non-demo) 的边界时，记录的首个在线航点 (Waypoint) 数据具有足够的保真度。
*   **`test_waypoint_phase_isolation.py`**: 验证演示录制与在线交互之间对于动作命令（尤其是航点 Waypoint）的数据隔离，防止缓冲中的演示动作残留到在线阶段污染录制的数据。

### 数据集生成的共享机制 (Pytest Fixture + 缓存)

由于渲染与调用底层运动规划求解器来录制合格的数据集极其耗费时间，`tests/dataset/` 下的各集测试使用了一套完善的数据生成缓存机制 (`_shared/dataset_generation.py`)，确保同一个用例的完整演示轨迹仅生成一次：

1.  **Session 级别的生成器 (`dataset_factory`)**: 在 `dataset/conftest.py` 中，定义了一个全局唯一、生存周期贯穿整个 Test Session 的工厂函数 `dataset_factory`。
2.  **基于哈希标识的临时目录缓存 (`DatasetFactoryCache`)**: 数据集在第一次请求时会被落盘到 Pytest 提供的临时目录机制 `tmp_path_factory` 中。此数据的唯一特征由环境名称、步数、难度甚至是动作控制模式组装为 `cache_key`。
3.  **后续测试的直接调取**: 例如多个测试都在共同请求 `video_unmaskswap_train_ep0_dataset` 这个包含前置录制数据的 Fixture，引擎会察觉被命中，直接返回同一个已准备好的 HDF5 测试文件。这样避免了多个测试用例对类似任务轨迹的重复物理演算。

## 2. `lightweight/` 目录：轻量级功能单元测试

主要针对一些内部特定逻辑如标签匹配、数据后处理和各种特定任务场景的状态进行单元级或分支级别的断言验证。

*   **`test_ChoiceLabel.py`**: 测试动作推断时的回放匹配逻辑 (`oracle_action_matcher`)，验证如“精准提取选项标签”、“忽略空标签文本”与目标字典选项正确映射绑定的流程。
*   **`test_ChoicePositionNearest.py`**: 位置匹配逻辑测试。覆盖 `select_target_with_position` 的 3D 最近邻行为，包括无效候选跳过、无有效输入返回 `None`、以及等距时按候选展平顺序稳定选取。
*   **`test_choice_action_pixel_mapping.py`**: 像素级映射与选择逻辑。测试从世界 3D 坐标投影到相机像素 2D 平面的映射器算法 (`project_world_to_pixel`)，以及验证在屏幕像素级选择目标 (`select_target_with_pixel`) 的精准就近选取能力。
*   **`test_StopcubeIncrement.py`**: 针对特有任务 `StopCube` 的时序功能验证。验证其选项中的 "remain static (保持静止)" 触发时，内部调度器的绝对时间步长 (`absTimestep`) 增量是不是按预期递增并最终触及上限阶段（Saturation）。包含模拟时间步后退是否可以正确重置计数器的情况。
*   **`test_TaskGoal.py` / `test_TaskGoalI_isList.py`**: 对 `task_goal.py` 内部生成自然语言描述 (`get_language_goal`) 的逻辑进行分支覆盖率测试。检验包括多达十几个子任务能够为特定场景组装出准确数量的双语目标描述。
*   **`test_choice_action_is_keyframe_flow.py`**: 针对于以离散项和像素位置为特征提取的工作流测试。判定它记录是否如实满足设定的关键帧（Keyframe）准入条件，并确保 `position_3d` 仅作为录制补充字段。
*   **`test_waypoint_dense_dedup.py`**: 测试基于示教航点（Waypoint）动作空间的密集轨迹筛选与相邻去重（Dedup）逻辑。
*   **`test_record_info_is_completed.py`**: 轻量级验证。分析 AST 语法树确保 `RecordWrapper` 在生成 HDF5 文件过程中正确处理了在线阶段的进展任务（例如进度标记字段 `is_completed` 等）。
*   **`test_record_video_metadata_fields.py`**: 数据元字段轻量级检测。通过语法树扫描，确保 `RecordWrapper` 会将正确的指令标签、行为选项、完成标志等写在数据记录 Buffer 中以供回放渲染和可视化验证调用。
*   **`test_record_waypoint_pending_flow.py`**: 语法流分析。确保在录制过程中数据流动（Waypoint Pending 状态更新、缓存）具备正确的生命周期管理逻辑及清理隔离手段。
*   **`test_step_error_handling.py`**: 轻量级结构测试和语法分析。利用 Mock 和 AST 检测确认核心环境包装层能够有效捕获报错将 `info["status"]` 置为 `"error"`；并核对其它调用侧（如 `run_example.py`）是否正确捕获了安全错误标识，以取代生硬的 `try-except`。

## 3. 公共设置与辅助脚本 (`conftest.py` 与 `_shared/`)

*   **`conftest.py` & `dataset/conftest.py`**: 定义了 Pytest 各层级的 Fixture（包含如何提前通过 `BenchmarkEnvBuilder` 注册相关环境，构建专用的临时存储工厂）。
*   **`_shared/`**: 内含如 `dataset_generation.py` 类工具脚本，用来配合临时 HDF5 mock 结构，统一管理基准测试项目路径位置。

## 如何进行测试

本项目强烈依赖 **`uv`** 管理虚拟环境，所有的测试执行命令都必须在代码根目录由 `uv run` 引导。

### 1. 运行全部测试

```bash
uv run python -m pytest tests/
```

若你想看到实时的 `print()` 和环境构建标准输出提示，可以通过 `-s` 关掉日志捕捉：

```bash
uv run python -m pytest tests/ -s
```

### 2. 分板块运行

**运行偏纯逻辑验证的无物理渲染类测试（速度极快）：**

```bash
uv run python -m pytest tests/lightweight/
```

**运行需要调用实际 Mujoco 仿真物理以及数据装载封装的测试（略耗时）：**

```bash
uv run python -m pytest tests/dataset/
```

### 3. 执行特定的某个测试脚本或者单个测试方法

精确到文件：

```bash
uv run python -m pytest tests/lightweight/test_TaskGoal.py
```

精确运行到某文件下的单个用例（例如）：

```bash
uv run python -m pytest tests/lightweight/test_TaskGoal.py::test_binfill_two_colors
```

### 4. 通过装饰标记 (Pytest Mark) 运行

对于某些被赋予 `@pytest.mark.dataset` 的文件，你还可以通过以下匹配执行：

```bash
uv run python -m pytest -m dataset
```
