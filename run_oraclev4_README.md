# oraclev4.py 运行脚本使用说明

## 功能特性

- ✅ 使用指定的 micromamba 环境：`/home/hongzefu/micromamba/envs/maniskillenv1228`
- ✅ 后台运行，SSH 断开后仍可继续运行
- ✅ 实时查看日志信息
- ✅ 自动管理进程和日志文件

## 使用方法

### 1. 启动脚本

```bash
cd /home/hongzefu/historybench-v5.6.1b7-videoStream
bash run_oraclev4.sh start
```

脚本会在后台运行，即使 SSH 断开也会继续执行。

### 2. 实时查看日志

```bash
bash run_oraclev4.sh view
```

这会实时显示日志输出（类似 `tail -f`），按 `Ctrl+C` 退出查看（不会停止脚本）。

### 3. 查看最后几行日志

```bash
bash run_oraclev4.sh tail
```

查看日志文件的最后 50 行。

### 4. 查看运行状态

```bash
bash run_oraclev4.sh status
```

显示：
- 是否正在运行
- 进程 PID
- 日志文件路径和大小
- 最后 5 行日志

### 5. 停止脚本

```bash
bash run_oraclev4.sh stop
```

## 日志文件位置

所有日志文件保存在：
```
/home/hongzefu/historybench-v5.6.1b5-changePatternLock/logs/
```

日志文件命名格式：`oraclev4_YYYYMMDD_HHMMSS.log`

## 注意事项

1. **环境路径**：脚本强制使用 `/home/hongzefu/micromamba/envs/maniskillenv1228` 环境
2. **环境检查**：启动前会自动检查环境路径和 Python 解释器是否存在
3. **查看日志**：使用 `view` 命令可以实时查看日志，不会影响脚本运行
4. **SSH 断开**：脚本使用 `nohup` 运行，SSH 断开后仍会继续执行
5. **重新连接后**：使用 `status` 命令检查脚本是否仍在运行

## 示例工作流

```bash
# 1. 启动脚本
bash run_oraclev4.sh start

# 2. 在另一个终端查看实时日志
bash run_oraclev4.sh view

# 3. SSH 断开后重新连接，检查状态
bash run_oraclev4.sh status

# 4. 查看最新日志
bash run_oraclev4.sh tail

# 5. 完成后停止脚本
bash run_oraclev4.sh stop
```

## 故障排查

如果遇到问题：

1. **检查 conda 环境**：
   ```bash
   conda env list | grep maniskillenv1228
   ```

2. **检查进程**：
   ```bash
   ps aux | grep oraclev4.py
   ```

3. **查看完整日志**：
   ```bash
   ls -lt logs/oraclev4_*.log | head -1
   cat <最新的日志文件>
   ```

