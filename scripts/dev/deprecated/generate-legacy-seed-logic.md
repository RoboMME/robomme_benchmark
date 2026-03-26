# `generate-legacy.py` 的 seed 生成逻辑

本文说明 [`generate-legacy.py`](/data/hongzefu/robomme_benchmark-heldOutSeed/scripts/dev/deprecated/generate-legacy.py) 里 `seed` 是怎么生成、怎么重试、以及最后怎么被记录和复现的。

## 1. 总体思路

这个脚本不是简单地用一个全局起始 seed，然后按 `+1` 往后加。

它把一个最终使用的 seed 拆成 3 个层次：

1. `env_id` 决定一个大的 seed 区间
2. `episode` 决定这个区间里的一个子段
3. `attempt` 决定这个 episode 在失败重试时具体用哪个 seed

对应公式是：

```text
base_seed = SEED_OFFSET + env_code * 10000 + episode * 100
seed = base_seed + attempt
```

其中：

- `SEED_OFFSET = 500_000 * 2 = 1_000_000`
- `env_code = ENV_ID_TO_CODE[env_id]`
- `attempt` 从 `0` 开始，每失败一次就 `+1`

## 2. `env_code` 是怎么来的

`env_code` 不是外部配置文件读出来的，而是直接按当前脚本里的 `DEFAULT_ENVS` 顺序枚举：

```python
DEFAULT_ENVS = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
]

ENV_ID_TO_CODE = {name: idx + 1 for idx, name in enumerate(DEFAULT_ENVS)}
```

所以当前文件里实际映射是：

| env_id | env_code |
| --- | --- |
| `PickXtimes` | `1` |
| `StopCube` | `2` |
| `SwingXtimes` | `3` |
| `BinFill` | `4` |

这意味着不同环境天然落在不同的 seed 大区间里：

| env_id | episode 0 的 `base_seed` |
| --- | --- |
| `PickXtimes` | `1_010_000` |
| `StopCube` | `1_020_000` |
| `SwingXtimes` | `1_030_000` |
| `BinFill` | `1_040_000` |

## 3. episode 如何映射到 seed

每个 `episode` 预留一整段长度为 `100` 的 seed 空间：

```text
episode 0 -> [base_seed + 0,  base_seed + 99]
episode 1 -> [base_seed + 100, base_seed + 199]
episode 2 -> [base_seed + 200, base_seed + 299]
...
```

更准确地说，某个 episode 的首个候选 seed 是：

```text
SEED_OFFSET + env_code * 10000 + episode * 100
```

例如：

- `PickXtimes`, `episode=0`
  - `base_seed = 1_000_000 + 1*10_000 + 0*100 = 1_010_000`
- `PickXtimes`, `episode=3`
  - `base_seed = 1_000_000 + 1*10_000 + 3*100 = 1_010_300`
- `BinFill`, `episode=7`
  - `base_seed = 1_000_000 + 4*10_000 + 7*100 = 1_040_700`

## 4. 失败重试时 seed 怎么变化

真正运行时，脚本不是只试一次。

在 [`run_env_dataset()`](/data/hongzefu/robomme_benchmark-heldOutSeed/scripts/dev/deprecated/generate-legacy.py) 里，每个 episode 会进入一个 retry loop：

```python
max_attempts = 100
attempt = 0
while attempt < max_attempts:
    seed = base_seed + attempt
```

行为是：

- 第 1 次尝试用 `seed = base_seed + 0`
- 如果失败，第 2 次尝试用 `seed = base_seed + 1`
- 如果再失败，就继续 `+1`
- 最多尝试 `100` 次，也就是使用 `[base_seed, base_seed + 99]`

这说明：

- 同一个 `env_id + episode` 的所有重试，都会被限制在自己那 100 个 seed 槽位里
- 成功后停止重试，并把“成功时实际使用的 seed”记录下来
- 如果 100 次都失败，这个 episode 最终不会进入 metadata

## 5. 为什么是 `*10000` 和 `*100`

这是一个手工编码过的“分段 seed”方案：

- `env_code * 10000`
  - 给每个环境预留 `10000` 个连续 seed
- `episode * 100`
  - 给每个 episode 预留 `100` 个连续 seed
- `attempt`
  - 在这 100 个位置里线性重试

因此它隐含了两个边界：

1. 每个 episode 最多只能安全使用 100 个重试 seed
2. 每个环境最多只能安全容纳 100 个 episode

原因是：

- `episode=0` 占用 `0..99`
- `episode=1` 占用 `100..199`
- ...
- `episode=99` 占用 `9900..9999`
- `episode=100` 的起点正好变成 `10000`

而 `10000` 已经等于下一个 `env_code` 的起始偏移。

所以当前公式在“无碰撞”意义上只对下面这个范围严格成立：

```text
0 <= episode <= 99
0 <= attempt <= 99
```

当前代码里 `max_attempts = 100`，正好和这个设计一致，所以“episode 内的 attempt”不会越界到下一个 episode 段。

但如果未来把 `args.episodes` 跑到 `100` 以上，就会出现跨环境 seed 冲突。

## 6. parallel 执行为什么不会打乱 seed

这个脚本虽然可能用 `ProcessPoolExecutor` 并行跑多个 episode chunk，但 seed 不是“按执行顺序分配”的，而是纯函数：

```text
seed = f(env_id, episode, attempt)
```

也就是说：

- worker 数量不同，不会改变某个 episode 的候选 seed 集合
- chunk 划分不同，不会改变某个 episode 的 seed
- 只要 `env_id / episode / attempt` 一样，最终 seed 就一样

所以并行只影响执行顺序，不影响 seed 编码本身。

## 7. difficulty 和 seed 的关系

这里要分两层看。

### 7.1 脚本层：difficulty 主要由 episode 下标决定

[`_get_difficulty_from_ratio()`](/data/hongzefu/robomme_benchmark-heldOutSeed/scripts/dev/deprecated/generate-legacy.py) 会把参数 `--difficulty` 解析成一个循环。

例如默认值 `211` 会展开成：

```text
[easy, easy, medium, hard]
```

然后按：

```text
difficulty = difficulties[episode % total]
```

来分配。

所以默认情况下：

- `episode 0 -> easy`
- `episode 1 -> easy`
- `episode 2 -> medium`
- `episode 3 -> hard`
- `episode 4 -> easy`
- 之后循环

也就是说，在 `generate-legacy.py` 里，difficulty 默认主要由 `episode` 控制，而不是由 seed 控制。

### 7.2 环境层：如果没有显式 difficulty，很多环境会退化成 `seed % 3`

底层环境类通常会：

- 先保存 `self.seed = seed`
- 再用这个 seed 去初始化内部随机数生成器
- 如果没有传入 difficulty，就用 `seed % 3` 推出 `easy/medium/hard`

例如 [`PickXtimes.py`](/data/hongzefu/robomme_benchmark-heldOutSeed/src/robomme/robomme_env/PickXtimes.py) 和 [`BinFill.py`](/data/hongzefu/robomme_benchmark-heldOutSeed/src/robomme/robomme_env/BinFill.py) 都是这样。

但在 `generate-legacy.py` 里，`gym.make()` 时已经显式传入了：

```python
env_kwargs = dict(
    ...,
    seed=seed,
    difficulty=difficulty,
)
```

所以对这个脚本来说：

- `seed` 主要负责场景随机性、对象配置、内部采样
- `difficulty` 主要由 ratio 直接指定
- 环境内部的 `seed % 3 -> difficulty` 退化逻辑大多不会生效

## 8. seed 最后被记录到哪里

成功 episode 的 seed 会被记录三次：

1. `episode_records`
   - 脚本在成功后会把 `{task, episode, seed, difficulty}` 加入内存列表
2. metadata JSON
   - `_save_episode_metadata()` 会把成功 episode 写到
     `record_dataset_{env_id}_metadata.json`
3. HDF5 / video 文件名和 setup 字段
   - `RobommeRecordWrapper` 会生成
     - `.../{env_id}_ep{episode}_seed{seed}.h5`
     - `.../{env_id}_ep{episode}_seed{seed}....mp4`
   - 并且在 HDF5 的 `setup/seed` 里再次保存这个值

所以最终“一个 episode 真正对应哪个 seed”，不是看 `base_seed`，而是看最终成功的那个 `seed`。

## 9. 后续如何复现同一个 episode

后续复现不是重新跑公式，而是直接读 metadata。

[`episode_config_resolver.py`](/data/hongzefu/robomme_benchmark-heldOutSeed/src/robomme/env_record_wrapper/episode_config_resolver.py) 会：

1. 从 metadata JSON 里找到 `(env_id, episode)` 对应记录
2. 取出里面保存的 `seed` 和 `difficulty`
3. 用这两个值重新 `gym.make(...)`

所以：

- 生成阶段靠公式选候选 seed
- 复现阶段靠 metadata 锁定最终成功 seed

## 10. 一句话总结

`generate-legacy.py` 的 seed 逻辑本质上是一个分段编码方案：

```text
最终 seed = 固定偏移 + 环境段偏移 + episode 段偏移 + 重试偏移
```

优点是：

- 并行时稳定
- 失败重试可追踪
- 成功 seed 能精确复现

缺点是：

- 设计上只安全覆盖每个环境 `100` 个 episode
- 每个 episode 只安全覆盖 `100` 次尝试
- 这是手工约定，不是自动扩展的命名空间
