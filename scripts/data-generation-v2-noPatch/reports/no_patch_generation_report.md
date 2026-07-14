# No-Patch 数据生成报告

## 运行来源

- 状态：passed
- 当前 HEAD：7a4362d9f8891649631cc7ef01c4935af684f942
- uv.lock SHA-256：983de83f7b22c98b96c3c25a39958b4f5920e3232cfaa209c89542ef5639ac03
- 报告时间（UTC）：2026-07-14T13:28:54.521494+00:00
- 报告模式：existing_output_revalidation

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
