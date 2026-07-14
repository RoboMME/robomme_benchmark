# No-Patch 数据生成报告

## 运行来源

- 状态：passed
- 当前 HEAD：68557e9a2ce25caae375637bd00b5d5f80ac20c4
- uv.lock SHA-256：983de83f7b22c98b96c3c25a39958b4f5920e3232cfaa209c89542ef5639ac03
- 报告时间（UTC）：2026-07-14T14:50:59.526651+00:00
- 报告模式：existing_output_revalidation

## 调试环境

- 快照时间（UTC）：2026-07-14T14:51:53.119783+00:00
- 主机：sled-vail
- OS：Linux 6.8.0-1018-nvidia-lowlatency
- 内核：6.8.0-1018-nvidia-lowlatency
- 架构：64bit；机器：x86_64
- libc：glibc 2.39

### CPU、内存与存储

| 项目 | 值 |
| --- | --- |
| OS CPU 数 | 32 |
| CPU affinity 数 | 32 |
| CPU affinity IDs | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 |
| CPU 型号 | AMD EPYC 9334 32-Core Processor |
| CPU sockets | 1 |
| 每核线程数 | 1 |
| 总内存 | 405257068544 bytes (377.43 GiB) |
| 仓库文件系统 | /data/hongzefu/robomme_benchmark-restore-DataGen |
| 文件系统总容量 | 15240752955392 bytes (13.86 TiB) |
| 文件系统已用 | 10102317232128 bytes (9.19 TiB) |
| 文件系统可用 | 4370269446144 bytes (3.97 TiB) |
- 完整 lscpu --json 原始字段保存在 JSON 的 debug_environment.cpu.lscpu.raw。

### GPU（nvidia-smi）

- nvidia-smi 可用：True
- nvidia-smi 版本：570.211.01
- nvidia-smi CUDA 版本：12.8

| GPU | 类型 | 字段 | 值 |
| --- | --- | --- | --- |
| 0 | 静态 | index | 0 |
| 0 | 静态 | uuid | GPU-590b693c-56db-a2b6-c4e3-5e20108f92d7 |
| 0 | 静态 | serial | 1713224014062 |
| 0 | 静态 | name | NVIDIA RTX 6000 Ada Generation |
| 0 | 静态 | pci_bus_id | 00000000:01:00.0 |
| 0 | 静态 | pci_device_id | 0x26B110DE |
| 0 | 静态 | pci_sub_device_id | 0x16A110DE |
| 0 | 静态 | driver_version | 570.211.01 |
| 0 | 静态 | vbios_version | 95.02.59.00.09 |
| 0 | 静态 | compute_capability | 8.9 |
| 0 | 静态 | memory_total_mib | 46068 |
| 0 | 动态 | memory_used_mib | 1664 |
| 0 | 动态 | memory_free_mib | 43802 |
| 0 | 静态 | power_limit_w | 300.00 |
| 0 | 动态 | power_draw_w | 141.18 |
| 0 | 动态 | temperature_c | 48 |
| 0 | 动态 | utilization_gpu_percent | 99 |
| 0 | 动态 | utilization_memory_percent | 0 |
| 0 | 动态 | pstate | P2 |
| 0 | 动态 | graphics_clock_mhz | 2715 |
| 0 | 动态 | memory_clock_mhz | 9501 |
| 0 | 静态 | max_graphics_clock_mhz | 3105 |
| 0 | 静态 | max_memory_clock_mhz | 10001 |
| 0 | 动态 | pcie_link_gen_current | 4 |
| 0 | 静态 | pcie_link_gen_max | 4 |
| 0 | 动态 | pcie_link_width_current | 16 |
| 0 | 静态 | pcie_link_width_max | 16 |
| 0 | 静态 | persistence_mode | Disabled |
| 0 | 静态 | addressing_mode | 未设置 |
| 0 | 动态 | fan_speed_percent | 30 |
| 1 | 静态 | index | 1 |
| 1 | 静态 | uuid | GPU-c58b7cba-c0e5-971b-e77a-0dd73bf26a9e |
| 1 | 静态 | serial | 1713224013719 |
| 1 | 静态 | name | NVIDIA RTX 6000 Ada Generation |
| 1 | 静态 | pci_bus_id | 00000000:02:00.0 |
| 1 | 静态 | pci_device_id | 0x26B110DE |
| 1 | 静态 | pci_sub_device_id | 0x16A110DE |
| 1 | 静态 | driver_version | 570.211.01 |
| 1 | 静态 | vbios_version | 95.02.59.00.09 |
| 1 | 静态 | compute_capability | 8.9 |
| 1 | 静态 | memory_total_mib | 46068 |
| 1 | 动态 | memory_used_mib | 6 |
| 1 | 动态 | memory_free_mib | 45461 |
| 1 | 静态 | power_limit_w | 300.00 |
| 1 | 动态 | power_draw_w | 28.02 |
| 1 | 动态 | temperature_c | 32 |
| 1 | 动态 | utilization_gpu_percent | 0 |
| 1 | 动态 | utilization_memory_percent | 0 |
| 1 | 动态 | pstate | P8 |
| 1 | 动态 | graphics_clock_mhz | 210 |
| 1 | 动态 | memory_clock_mhz | 405 |
| 1 | 静态 | max_graphics_clock_mhz | 3105 |
| 1 | 静态 | max_memory_clock_mhz | 10001 |
| 1 | 动态 | pcie_link_gen_current | 1 |
| 1 | 静态 | pcie_link_gen_max | 4 |
| 1 | 动态 | pcie_link_width_current | 16 |
| 1 | 静态 | pcie_link_width_max | 16 |
| 1 | 静态 | persistence_mode | Disabled |
| 1 | 静态 | addressing_mode | 未设置 |
| 1 | 动态 | fan_speed_percent | 30 |

### Python、工具与运行时

| 项目 | 值 |
| --- | --- |
| Python implementation | CPython |
| Python 完整版本 | 3.11.14 (main, Feb  3 2026, 22:51:56) [Clang 21.1.4 ] |
| Python ABI / cache tag | cpython-311-x86_64-linux-gnu / cpython-311 |
| Python executable | /data/hongzefu/robomme_benchmark-restore-DataGen/.venv/bin/python3 |
| venv / prefix | /data/hongzefu/robomme_benchmark-restore-DataGen/.venv / /data/hongzefu/robomme_benchmark-restore-DataGen/.venv |
| base prefix | /home/hongzefu/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu |
| uv | uv 0.10.2 (/home/hongzefu/.local/bin/uv) |
| git | git version 2.43.0 (/usr/bin/git) |
| Torch | 2.9.1+cu128 |
| Torch 编译 CUDA | 12.8 |
| cuDNN | 91002 |
| Torch CUDA 可用 / 可见数 | True / 2 |

| Torch CUDA index | 名称 | 算力 | 总显存 |
| --- | --- | --- | --- |
| 0 | NVIDIA RTX 6000 Ada Generation | 8.9 | 47673769984 bytes (44.40 GiB) |
| 1 | NVIDIA RTX 6000 Ada Generation | 8.9 | 47673769984 bytes (44.40 GiB) |

### 受限运行环境

| 类别 | 变量 | 值 |
| --- | --- | --- |
| 运行 | CUDA_VISIBLE_DEVICES | 未设置 |
| 运行 | OMP_NUM_THREADS | 未设置 |
| 运行 | MKL_NUM_THREADS | 未设置 |
| 运行 | CUDA_HOME | 未设置 |
| 运行 | PYTHONPATH | 未设置 |
| Slurm 分配 | SLURM_JOB_ID | 未设置 |
| Slurm 分配 | SLURM_JOB_GPUS | 未设置 |
| Slurm 分配 | SLURM_GPUS_ON_NODE | 未设置 |
| Slurm 分配 | SLURM_CPUS_PER_TASK | 未设置 |
| Slurm 分配 | SLURM_CPUS_ON_NODE | 未设置 |
| Slurm 分配 | SLURM_MEM_PER_NODE | 未设置 |
| Slurm 分配 | SLURM_MEM_PER_CPU | 未设置 |
| Slurm 分配 | SLURM_NNODES | 未设置 |
| Slurm 分配 | SLURM_NODELIST | 未设置 |

### 依赖

- 全部依赖版本数：111。
- 完整依赖清单位于同一 JSON 的 debug_environment.packages.distributions。

| 核心依赖 | 版本 |
| --- | --- |
| torch | 2.9.1 |
| torchvision | 0.24.1 |
| mani-skill | 3.0.0b21 |
| sapien | 3.0.2 |
| mplib | 0.1.1 |
| gymnasium | 0.29.1 |
| numpy | 1.26.4 |
| h5py | 3.15.1 |
| opencv-python | 4.11.0.86 |
| setuptools | 80.9.0 |
| robomme | 0.1.0 |

## 参数

    {
      "output_dir": "/data/hongzefu/robomme_benchmark-restore-DataGen/artifacts/generated/no-patch-full-16x9",
      "env": "all",
      "tasks": [
        "PickXtimes",
        "StopCube",
        "SwingXtimes",
        "BinFill",
        "VideoUnmaskSwap",
        "VideoUnmask",
        "ButtonUnmaskSwap",
        "ButtonUnmask",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
        "PickHighlight",
        "InsertPeg",
        "MoveCube",
        "PatternLock",
        "RouteStick"
      ],
      "episodes": 9,
      "workers": 9,
      "gpus": [
        "0",
        "1"
      ],
      "metadata_root": "/data/hongzefu/robomme_benchmark-restore-DataGen/src/robomme/env_metadata/train",
      "reference_root": "/data/hongzefu/robomme_benchmark-restore-DataGen/data/robomme_data_h5",
      "seed_attempts_per_episode": 1,
      "save_video_for_recording": true,
      "max_abs_diff": 1e-08
    }

## 范围

- 任务数：16
- episode：[0, 1, 2, 3, 4, 5, 6, 7, 8]
- 期望轨迹数：144
- 完整 16×9：True

## 生成与合约

- worker 成功数：144
- worker 失败数：0
- metadata 错误数：0
- 生成 HDF5 错误数：0
- 官方 HDF5 错误数：0
- 官方最终完成：144/144
- 生成最终完成：144/144

## joint_action 逐元素比较

- 向量数：73907
- 元素数：591256
- 不同元素数：16380
- 比较错误数：0
- 最大绝对差：5.661269342205344e-09
- 最大差位置：{'task': 'PatternLock', 'episode': 1, 'timestep': 121, 'element_index': 6, 'reference_value': 0.7855640085949516, 'generated_value': 0.7855640029336822}
- 最大允许绝对差：1e-08
- 容差内：True

## 结论

- 完整验收通过：True

## 产物

- JSON：/data/hongzefu/robomme_benchmark-restore-DataGen/scripts/data-generation-v2-noPatch/reports/no_patch_generation_report.json
- Markdown：/data/hongzefu/robomme_benchmark-restore-DataGen/scripts/data-generation-v2-noPatch/reports/no_patch_generation_report.md
