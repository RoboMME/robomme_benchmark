# No-Patch Dataset Generation Report

## Run Provenance

- Status: failed
- Current HEAD: 9430e20bfcf59116d525778b60663520b22f63e6
- uv.lock SHA-256: 983de83f7b22c98b96c3c25a39958b4f5920e3232cfaa209c89542ef5639ac03
- Report time (UTC): 2026-07-15T04:14:21.522575+00:00
- Report mode: generation

## Debug Environment

- Snapshot time (UTC): 2026-07-15T06:30:04.917791+00:00
- Host: sled-vail
- OS: Linux 6.8.0-1018-nvidia-lowlatency
- Kernel: 6.8.0-1018-nvidia-lowlatency
- Architecture: 64bit; Machine: x86_64
- libc: glibc 2.39

### CPU, Memory, and Storage

| Item | Value |
| --- | --- |
| OS CPU count | 32 |
| CPU affinity count | 32 |
| CPU affinity IDs | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 |
| CPU model | AMD EPYC 9334 32-Core Processor |
| CPU sockets | 1 |
| Threads per core | 1 |
| Total memory | 405257068544 bytes (377.43 GiB) |
| Repository filesystem | /data/hongzefu/robomme_benchmark-restore-DataGen |
| Filesystem capacity | 15240752955392 bytes (13.86 TiB) |
| Filesystem used | 11129608626176 bytes (10.12 TiB) |
| Filesystem available | 3342978052096 bytes (3.04 TiB) |
- The complete lscpu --json raw fields are available in JSON at debug_environment.cpu.lscpu.raw.

### GPU (nvidia-smi)

- nvidia-smi available: True
- nvidia-smi version: 570.211.01
- nvidia-smi CUDA version: 12.8

| GPU | Category | Field | Value |
| --- | --- | --- | --- |
| 0 | Static | index | 0 |
| 0 | Static | uuid | GPU-590b693c-56db-a2b6-c4e3-5e20108f92d7 |
| 0 | Static | serial | 1713224014062 |
| 0 | Static | name | NVIDIA RTX 6000 Ada Generation |
| 0 | Static | pci_bus_id | 00000000:01:00.0 |
| 0 | Static | pci_device_id | 0x26B110DE |
| 0 | Static | pci_sub_device_id | 0x16A110DE |
| 0 | Static | driver_version | 570.211.01 |
| 0 | Static | vbios_version | 95.02.59.00.09 |
| 0 | Static | compute_capability | 8.9 |
| 0 | Static | memory_total_mib | 46068 |
| 0 | Dynamic | memory_used_mib | 1017 |
| 0 | Dynamic | memory_free_mib | 44449 |
| 0 | Static | power_limit_w | 300.00 |
| 0 | Dynamic | power_draw_w | 82.77 |
| 0 | Dynamic | temperature_c | 43 |
| 0 | Dynamic | utilization_gpu_percent | 0 |
| 0 | Dynamic | utilization_memory_percent | 0 |
| 0 | Dynamic | pstate | P2 |
| 0 | Dynamic | graphics_clock_mhz | 2715 |
| 0 | Dynamic | memory_clock_mhz | 9501 |
| 0 | Static | max_graphics_clock_mhz | 3105 |
| 0 | Static | max_memory_clock_mhz | 10001 |
| 0 | Dynamic | pcie_link_gen_current | 4 |
| 0 | Static | pcie_link_gen_max | 4 |
| 0 | Dynamic | pcie_link_width_current | 16 |
| 0 | Static | pcie_link_width_max | 16 |
| 0 | Static | persistence_mode | Disabled |
| 0 | Static | addressing_mode | Not set |
| 0 | Dynamic | fan_speed_percent | 30 |
| 1 | Static | index | 1 |
| 1 | Static | uuid | GPU-c58b7cba-c0e5-971b-e77a-0dd73bf26a9e |
| 1 | Static | serial | 1713224013719 |
| 1 | Static | name | NVIDIA RTX 6000 Ada Generation |
| 1 | Static | pci_bus_id | 00000000:02:00.0 |
| 1 | Static | pci_device_id | 0x26B110DE |
| 1 | Static | pci_sub_device_id | 0x16A110DE |
| 1 | Static | driver_version | 570.211.01 |
| 1 | Static | vbios_version | 95.02.59.00.09 |
| 1 | Static | compute_capability | 8.9 |
| 1 | Static | memory_total_mib | 46068 |
| 1 | Dynamic | memory_used_mib | 19681 |
| 1 | Dynamic | memory_free_mib | 25785 |
| 1 | Static | power_limit_w | 300.00 |
| 1 | Dynamic | power_draw_w | 299.58 |
| 1 | Dynamic | temperature_c | 82 |
| 1 | Dynamic | utilization_gpu_percent | 100 |
| 1 | Dynamic | utilization_memory_percent | 28 |
| 1 | Dynamic | pstate | P2 |
| 1 | Dynamic | graphics_clock_mhz | 945 |
| 1 | Dynamic | memory_clock_mhz | 9501 |
| 1 | Static | max_graphics_clock_mhz | 3105 |
| 1 | Static | max_memory_clock_mhz | 10001 |
| 1 | Dynamic | pcie_link_gen_current | 4 |
| 1 | Static | pcie_link_gen_max | 4 |
| 1 | Dynamic | pcie_link_width_current | 16 |
| 1 | Static | pcie_link_width_max | 16 |
| 1 | Static | persistence_mode | Disabled |
| 1 | Static | addressing_mode | Not set |
| 1 | Dynamic | fan_speed_percent | 55 |

### Python, Tools, and Runtime

| Item | Value |
| --- | --- |
| Python implementation | CPython |
| Full Python version | 3.11.14 (main, Feb  3 2026, 22:51:56) [Clang 21.1.4 ] |
| Python ABI / cache tag | cpython-311-x86_64-linux-gnu / cpython-311 |
| Python executable | /data/hongzefu/robomme_benchmark-restore-DataGen/.venv/bin/python3 |
| venv / prefix | /data/hongzefu/robomme_benchmark-restore-DataGen/.venv / /data/hongzefu/robomme_benchmark-restore-DataGen/.venv |
| base prefix | /home/hongzefu/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu |
| uv | uv 0.10.2 (/home/hongzefu/.local/bin/uv) |
| git | git version 2.43.0 (/usr/bin/git) |
| Torch | 2.9.1+cu128 |
| Torch compiled CUDA | 12.8 |
| cuDNN | 91002 |
| Torch CUDA available / visible count | True / 1 |

| Torch CUDA index | Name | Compute capability | Total memory |
| --- | --- | --- | --- |
| 0 | NVIDIA RTX 6000 Ada Generation | 8.9 | 47673769984 bytes (44.40 GiB) |

### Restricted Runtime Environment

| Category | Variable | Value |
| --- | --- | --- |
| Runtime | CUDA_VISIBLE_DEVICES | 0 |
| Runtime | OMP_NUM_THREADS | Not set |
| Runtime | MKL_NUM_THREADS | Not set |
| Runtime | CUDA_HOME | Not set |
| Runtime | PYTHONPATH | Not set |
| Slurm allocation | SLURM_JOB_ID | Not set |
| Slurm allocation | SLURM_JOB_GPUS | Not set |
| Slurm allocation | SLURM_GPUS_ON_NODE | Not set |
| Slurm allocation | SLURM_CPUS_PER_TASK | Not set |
| Slurm allocation | SLURM_CPUS_ON_NODE | Not set |
| Slurm allocation | SLURM_MEM_PER_NODE | Not set |
| Slurm allocation | SLURM_MEM_PER_CPU | Not set |
| Slurm allocation | SLURM_NNODES | Not set |
| Slurm allocation | SLURM_NODELIST | Not set |

### Dependencies

- Total distributions: 111.
- The complete distribution list is available in the same JSON at debug_environment.packages.distributions.

| Core dependency | Version |
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

## Parameters

    {
      "output_dir": "/data/hongzefu/robomme_benchmark-restore-DataGen/artifacts/generated/no-patch-full-16x100",
      "env": "all",
      "episodes": 100,
      "workers": 20,
      "requested_gpus": "0",
      "metadata_root": "/data/hongzefu/robomme_benchmark-restore-DataGen/src/robomme/env_metadata/train",
      "reference_root": "/data/hongzefu/robomme_benchmark-restore-DataGen/data/robomme_data_h5",
      "max_abs_diff": 1e-08,
      "seed_attempts_per_episode": 1,
      "save_video_for_recording": true,
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
      "gpus": [
        "0"
      ]
    }

## Scope

- Task count: 16
- Episodes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
- Expected trajectory count: 1600
- Complete 16x100: True

## Generation and Contract

- Successful workers: 1600
- Failed workers: 0
- Metadata errors: 0
- Generated HDF5 errors: 0
- Official HDF5 errors: 0
- Official final completions: 1600/1600
- Generated final completions: 1600/1600

## Element-wise joint_action Comparison

- Vectors: 761885
- Elements: 6095080
- Different elements: 217242
- Comparison errors: 10
- Maximum absolute difference: 0.007857919612339614
- Maximum-difference location: {'task': 'BinFill', 'episode': 99, 'timestep': 625, 'element_index': 5, 'reference_value': 2.8383087099151, 'generated_value': 2.8461666295274397}
- Maximum allowed absolute difference: 1e-08
- Within tolerance: False

## File Manifest

- Status: collected
- Official reference revision: a5e4e25ffe8af34f64944f9533d06455ce5f8337
- Generated files: 32
- Generated bytes: 510364369421
- Official reference HDF5 files: 16
- Official reference HDF5 bytes: 512595968744

## Conclusion

- Full acceptance passed: False

### Generated Artifact SHA-256

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| record_dataset_PickXtimes.h5 | 35656486880 | 4b6324ecc50f43cb669eb0e9054c5017ccfa1e7d00f4b2a3dfe97afd73b2d67d |
| record_dataset_PickXtimes_metadata.json | 11021 | 353fe6921410882439830bec69101be899c5fc72d174c5963115a26a2d8ab7cf |
| record_dataset_StopCube.h5 | 20981589288 | f96af78445b166f9837944cc53c404dc12a26e7f84d7b606da94b96cc55388d8 |
| record_dataset_StopCube_metadata.json | 10829 | c1bad3cb75430c4a48830e8f87657a792c8fc415cb9fc87cecd2e4fd940c336c |
| record_dataset_SwingXtimes.h5 | 28819337264 | 6803353a2b7c8627651668e0c4f8b657c37948b3bf49af35a633109935625220 |
| record_dataset_SwingXtimes_metadata.json | 11142 | f7f96712fbde14a8294b446dfe34faf7d227e9d19723991b75db90171dcc271f |
| record_dataset_BinFill.h5 | 39978558852 | 336f22e35989db37c6bb8603c9423655a6a5f22bcfaaad1a2b67d02e88e75faa |
| record_dataset_BinFill_metadata.json | 10748 | 922159218bae3b03a4bef49b95f7584a168d68b3d3ce8b0563f5f6022f0db9d7 |
| record_dataset_VideoUnmaskSwap.h5 | 23102930152 | 66159eccba258f5c1b1ececcd9364a638a6907ceb544abf36d60dcd4b8f31502 |
| record_dataset_VideoUnmaskSwap_metadata.json | 11566 | ee5eecc20191f0dbfb8fcba7823ba5386de4fba658df15a79881cfc942d49482 |
| record_dataset_VideoUnmask.h5 | 14381527488 | e6e57a5d1e4f2a58751a5d1a45025ad0a4791a49726b50c5e47da87abd87d62a |
| record_dataset_VideoUnmask_metadata.json | 11172 | 93673c9c9c7b6d2e6ec1ff5b1b467e93aa1d0aa7ef8f53d98c996a666c796c8c |
| record_dataset_ButtonUnmaskSwap.h5 | 26509802976 | cefbd9c9f494268c9c7c4aa821fda4c459b54d9f1aa60732a5ee3a929e2063ca |
| record_dataset_ButtonUnmaskSwap_metadata.json | 11687 | 291bed6c5b6aea558a414bd8d190de6f23177e0431d3b40c61025f617cd04446 |
| record_dataset_ButtonUnmask.h5 | 17672981320 | 5644c1cfdca075aa07619f8de5403027490b7e596831a97a9eafddb3744d55b5 |
| record_dataset_ButtonUnmask_metadata.json | 11293 | 03f2529b3f06103520a823b047b1320f9bc9356a30753f2c1725320ed1f22fe8 |
| record_dataset_VideoRepick.h5 | 45565342348 | f694c72079cb823081e888bd4802ec226234141e2bacc7f51bb4a0861cfff572 |
| record_dataset_VideoRepick_metadata.json | 11202 | 198db3951a11dee6e62a558f87b7f9d68da3e5c9517611944097c70721376a65 |
| record_dataset_VideoPlaceButton.h5 | 64075923312 | e77c21658ccad60b2000096c1fe0c669d117d0a9d25ed3a69878c375a21cf8d4 |
| record_dataset_VideoPlaceButton_metadata.json | 11717 | e8221e69d455e49595db0724da36318555b33bfaac573d90d4b68bb6d1a8073a |
| record_dataset_VideoPlaceOrder.h5 | 74602283824 | cfcd946c20606116b791b38307b08270ae3fc36ef5bebae6cf7bba30ea682c6d |
| record_dataset_VideoPlaceOrder_metadata.json | 11616 | d1966c8382844e0c6a4a7c80dc028cf5a855a5a86abb3c1dc793c3830559952c |
| record_dataset_PickHighlight.h5 | 22921958288 | 4ac341ceef34b5e52bde5f2b88e5db020073a9433d06ed17532369b759e1150f |
| record_dataset_PickHighlight_metadata.json | 11414 | 32f9a912c90be26626843ade4c5cea735552517664633ddfb2b35038e0a1b043 |
| record_dataset_InsertPeg.h5 | 31663595140 | 3cdf173944a3c1bc72fc1b07c3e3bddceff04840d6466b058571606a5ef6a7dc |
| record_dataset_InsertPeg_metadata.json | 11010 | fa3f65f35428c047604f0996577c6d5d43077a64103a5d81dcaeedc18b3f3b4f |
| record_dataset_MoveCube.h5 | 26142699936 | a04e997045e0bd53d07fb36cc223a3a801403994beb738332ce61b628c8d7848 |
| record_dataset_MoveCube_metadata.json | 10909 | 268637d8b30795ee930f453b0e199ad5e7e0a69e73d533078d077ecf42c6c0eb |
| record_dataset_PatternLock.h5 | 13792081904 | cb06e20b15ce45d3f5f1bddcce7e30913223606d424bf4c2a0445cc1ff079b85 |
| record_dataset_PatternLock_metadata.json | 11212 | 7a8e653e1612a2eec72a15dbe82a314d2f8395e785387d7e315da7622c3041c8 |
| record_dataset_RouteStick.h5 | 24497090800 | 39c06278b5f62a23cf2ae2978500f0fc89aa1d82fb542652c940564ef8ed59f2 |
| record_dataset_RouteStick_metadata.json | 11111 | 64d77aff1849e60492f9ee8b026adde618c64917ab75737869712573ee61ed17 |

## Error Summary

- BinFill/episode_11: timestep sets do not match
- BinFill/episode_94: timestep sets do not match
- VideoRepick/episode_39: timestep sets do not match
- VideoRepick/episode_59: timestep sets do not match
- VideoPlaceButton/episode_58: timestep sets do not match
- VideoPlaceButton/episode_83: timestep sets do not match
- VideoPlaceOrder/episode_46: timestep sets do not match
- VideoPlaceOrder/episode_50: timestep sets do not match
- PickHighlight/episode_3: timestep sets do not match
- PickHighlight/episode_38: timestep sets do not match
- DatasetGenerationError: Post-generation validation or joint_action comparison did not meet acceptance criteria
- See the co-located JSON for complete errors and the per-trajectory audit.

## Artifacts

- JSON: /data/hongzefu/robomme_benchmark-restore-DataGen/scripts/data-generation/reports/generation_report.json
- Markdown: /data/hongzefu/robomme_benchmark-restore-DataGen/scripts/data-generation/reports/generation_report.md
