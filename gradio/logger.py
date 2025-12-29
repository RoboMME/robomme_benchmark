import json
import threading
import os
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import cv2
from PIL import Image

# 线程锁，防止多用户同时写入时文件损坏
lock = threading.Lock()
# 使用基于 logger.py 文件位置的绝对路径，确保日志文件始终保存在 gradio/data/ 目录下
BASE_DIR = Path(__file__).parent.absolute()
USER_ACTION_LOG_DIR = str(BASE_DIR / "data" / "user_action_logs")

def _get_current_attempt_index(f):
    """
    获取当前最新的 attempt 索引。
    
    Args:
        f: h5py.File 对象
    
    Returns:
        int: 当前 attempt 索引，如果没有则返回 -1
    """
    attempt_indices = []
    for key in f.keys():
        if key.startswith("attempt_"):
            try:
                idx = int(key.split("_")[1])
                attempt_indices.append(idx)
            except (ValueError, IndexError):
                pass
    
    if not attempt_indices:
        return -1
    return max(attempt_indices)

def _get_or_create_attempt(f, attempt_idx, username, env_id, episode_idx):
    """
    获取或创建指定索引的 attempt 组。
    
    Args:
        f: h5py.File 对象
        attempt_idx: attempt 索引（如果为 None，则创建新的）
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    
    Returns:
        h5py.Group: attempt 组对象
    """
    if attempt_idx is None:
        # 创建新的 attempt
        current_max = _get_current_attempt_index(f)
        attempt_idx = current_max + 1
    
    attempt_name = f"attempt_{attempt_idx}"
    
    if attempt_name not in f:
        # 创建新的 attempt 组
        attempt_group = f.create_group(attempt_name)
        
        # 创建 metadata 组（只创建组，option_list 将在第一次记录 action 时添加）
        metadata = attempt_group.create_group("metadata")
        
        # 创建 actions 组
        attempt_group.create_group("actions")
    else:
        # 获取现有的 attempt 组
        attempt_group = f[attempt_name]
    
    return attempt_group

def _ensure_hdf5_file(username, env_id, episode_idx, create_new_attempt=False):
    """
    确保 HDF5 文件存在并初始化必要的组结构。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        create_new_attempt: 是否创建新的 attempt（用于 refresh）
    
    Returns:
        tuple: (h5py.File 对象, h5py.Group attempt_group, int attempt_idx) 或 (None, None, -1)（如果出错）
    """
    if not username or not env_id or episode_idx is None:
        return None, None, -1
    
    # 构建文件路径
    user_dir = os.path.join(USER_ACTION_LOG_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    hdf5_file = os.path.join(user_dir, f"{env_id}_{episode_idx}.h5")
    
    try:
        # 检查文件是否存在
        file_exists = os.path.exists(hdf5_file)
        
        # 以追加模式打开（如果不存在则创建）
        f = h5py.File(hdf5_file, "a")
        
        # 获取或创建 attempt
        if create_new_attempt:
            # 创建新的 attempt
            attempt_group = _get_or_create_attempt(f, None, username, env_id, episode_idx)
            attempt_idx = _get_current_attempt_index(f)
        else:
            # 获取当前最新的 attempt，如果不存在则创建 attempt_0
            current_attempt_idx = _get_current_attempt_index(f)
            if current_attempt_idx == -1:
                # 文件存在但没有 attempt，创建 attempt_0
                attempt_group = _get_or_create_attempt(f, 0, username, env_id, episode_idx)
                attempt_idx = 0
            else:
                # 使用当前最新的 attempt
                attempt_group = _get_or_create_attempt(f, current_attempt_idx, username, env_id, episode_idx)
                attempt_idx = current_attempt_idx
        
        return f, attempt_group, attempt_idx
    except Exception as e:
        print(f"Error ensuring HDF5 file {hdf5_file}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, -1

def _normalize_image(image_array):
    """
    规范化图像格式为 uint8, [H, W, 3] RGB。
    
    Args:
        image_array: numpy array，可能是各种格式的图像
    
    Returns:
        numpy array: 规范化后的图像 [H, W, 3], uint8
    """
    if image_array is None:
        return None
    
    # 确保是 uint8 类型
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    # 确保是 RGB 格式 [H, W, 3]
    if len(image_array.shape) == 2:
        # 灰度图转 RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # RGBA 转 RGB
        image_array = image_array[:, :, :3]
    elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
        # 其他格式，尝试转换
        print(f"Warning: Unexpected image shape {image_array.shape}, attempting to convert")
        image_array = image_array[:, :, :3] if image_array.shape[2] > 3 else image_array
    
    return image_array

def _add_action_to_hdf5(attempt_group, action_index, action_data):
    """
    将操作记录添加到 HDF5 文件的 attempt 组中。
    
    新的数据格式（多帧时序交互结构）：
    每个 action 组包含扁平化的数组结构：
    - click_history_annotated_image: [T, H, W, 3] 图像序列（点击图像 + 最终执行图像，按时间排序，带坐标标记）
    - click_history: [N, 3] 点击数据 [x, y, img_idx]
    - click_timestamps: [N] 点击绝对时间戳（ISO格式）
    - option_history: [M] 结构化数组，包含选项索引和标签
    - option_timestamps: [M] 选项绝对时间戳（ISO格式）
    - final_choice: [3] 决策向量 [Type, Val1, Val2]，其中 Type=option_idx
    
    Args:
        attempt_group: h5py.Group 对象，表示 attempt 组
        action_index: 操作索引（用于生成唯一的 action 组名）
        action_data: 操作数据字典，包含：
            - option_idx: execute 时使用的选项索引（最后一次选择的）
            - option_label: execute 时使用的选项标签
            - final_coordinates: 最后执行的坐标 {"x": x, "y": y}（可选）
            - final_coords_str: 最后执行的坐标字符串（可选）
            - final_image_array: 最后执行时的图片数组（可选）
            - option_selects_before_execute: 列表，包含 execute 之前所有的选项选择
            - coordinate_clicks_before_execute: 列表，包含 execute 之前所有的坐标点击（每个元素包含 coordinates, coords_str, image_array, timestamp）
            - status: 执行状态
            - done: 是否完成
    """
    try:
        actions_group = attempt_group["actions"]
        action_name = f"action_{action_index}"
        
        # 创建 action 组
        action_group = actions_group.create_group(action_name)
        
        # 存储基本属性
        execute_timestamp = action_data.get("execute_timestamp", datetime.now().isoformat())
        action_group.create_dataset("execute_timestamp", data=execute_timestamp.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 获取数据
        coordinate_clicks = action_data.get("coordinate_clicks_before_execute", [])
        option_selects = action_data.get("option_selects_before_execute", [])
        final_image_array = action_data.get("final_image_array")
        final_coordinates = action_data.get("final_coordinates")
        option_idx = action_data.get("option_idx")
        
        # ========== 1. 收集和排序图像序列 ==========
        # 收集所有图像及其时间戳和来源信息
        image_entries = []  # [(timestamp, image_array, source_type, source_index, coordinates), ...]
        click_base_image = None  # 保存原始点击图像（不画圈的）
        
        # 收集点击图像
        for click_idx, click_data in enumerate(coordinate_clicks):
            click_image = click_data.get("image_array")
            click_timestamp = click_data.get("timestamp", execute_timestamp)
            click_coords = click_data.get("coordinates")
            if click_image is not None:
                normalized_image = _normalize_image(click_image)
                if normalized_image is not None:
                    # 保存第一个点击的原始图像作为 click_base_image
                    if click_base_image is None:
                        click_base_image = normalized_image.copy()
                    # 在图像上画圈标记坐标点
                    marked_image = normalized_image.copy()
                    if click_coords:
                        try:
                            # 确保数组在内存中是连续的
                            if not marked_image.flags['C_CONTIGUOUS']:
                                marked_image = np.ascontiguousarray(marked_image)
                            else:
                                marked_image = marked_image.copy()
                            
                            x = click_coords.get("x", 0)
                            y = click_coords.get("y", 0)
                            # 画红色圆圈: 中心点, 半径5, 颜色(255,0,0), 线宽2
                            cv2.circle(marked_image, (int(x), int(y)), 5, (255, 0, 0), 2)
                        except Exception as e:
                            print(f"Error drawing circle on click image: {e}")
                    
                    image_entries.append((click_timestamp, marked_image, "click", click_idx, click_coords))
        
        # 收集最终执行图像
        if final_image_array is not None:
            normalized_final = _normalize_image(final_image_array)
            if normalized_final is not None:
                # 如果最终执行有坐标，也在图像上画圈
                marked_final = normalized_final.copy()
                if final_coordinates:
                    try:
                        if not marked_final.flags['C_CONTIGUOUS']:
                            marked_final = np.ascontiguousarray(marked_final)
                        else:
                            marked_final = marked_final.copy()
                        
                        x = final_coordinates.get("x", 0)
                        y = final_coordinates.get("y", 0)
                        cv2.circle(marked_final, (int(x), int(y)), 5, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"Error drawing circle on final image: {e}")
                
                image_entries.append((execute_timestamp, marked_final, "final", None, final_coordinates))
        
        # 按时间戳排序图像
        image_entries.sort(key=lambda x: x[0])
        
        # 构建 images 数组 [T, H, W, 3]
        if image_entries:
            # 获取图像尺寸（假设所有图像尺寸相同）
            first_image = image_entries[0][1]
            H, W = first_image.shape[0], first_image.shape[1]
            T = len(image_entries)
            
            images_array = np.zeros((T, H, W, 3), dtype=np.uint8)
            for idx, (_, img, _, _, _) in enumerate(image_entries):
                # 确保图像尺寸一致（如果不一致则调整）
                if img.shape[0] != H or img.shape[1] != W:
                    # 使用 cv2 调整大小
                    img = cv2.resize(img, (W, H))
                images_array[idx] = img
            
            # 存储 click_history_annotated_image 数组（使用压缩）
            action_group.create_dataset(
                "click_history_annotated_image",
                data=images_array,
                compression="gzip",
                compression_opts=9,
                dtype=np.uint8
            )
        else:
            # 如果没有图像，创建空数组 [0, H, W, 3]
            # 使用默认尺寸 224x224（常见的图像尺寸）
            default_H, default_W = 224, 224
            images_array = np.zeros((0, default_H, default_W, 3), dtype=np.uint8)
            action_group.create_dataset("click_history_annotated_image", data=images_array, dtype=np.uint8)
        
        # 存储 click_base_image（原始图片，不画圈的）
        if click_base_image is not None:
            # 确保图像尺寸一致
            if image_entries:
                H, W = image_entries[0][1].shape[0], image_entries[0][1].shape[1]
                if click_base_image.shape[0] != H or click_base_image.shape[1] != W:
                    click_base_image = cv2.resize(click_base_image, (W, H))
            
            action_group.create_dataset(
                "click_base_image",
                data=click_base_image,
                compression="gzip",
                compression_opts=9,
                dtype=np.uint8
            )
        
        # ========== 2. 构建 click_history 和 click_timestamps ==========
        click_history_list = []
        click_timestamps_list = []
        
        # 创建图像到索引的映射（基于时间戳和来源）
        image_to_index = {}
        for idx, (ts, img, source_type, source_idx, _) in enumerate(image_entries):
            # 使用 (source_type, source_idx) 作为键来映射点击图像
            if source_type == "click":
                image_to_index[("click", source_idx)] = idx
        
        # 处理每个点击
        for click_idx, click_data in enumerate(coordinate_clicks):
            coordinates = click_data.get("coordinates")
            click_timestamp = click_data.get("timestamp", execute_timestamp)
            
            if coordinates:
                x = coordinates.get("x", 0)
                y = coordinates.get("y", 0)
                
                # 查找对应的图像索引
                img_idx = image_to_index.get(("click", click_idx), 0)
                if len(image_entries) == 0:
                    img_idx = 0  # 如果没有图像，使用 0
                elif img_idx >= len(image_entries):
                    img_idx = len(image_entries) - 1  # 确保索引有效
                
                click_history_list.append([x, y, img_idx])
                click_timestamps_list.append(click_timestamp)
        
        # 存储 click_history [N, 3]
        if click_history_list:
            click_history_array = np.array(click_history_list, dtype=np.int32)
            action_group.create_dataset("click_history", data=click_history_array, dtype=np.int32)
        else:
            click_history_array = np.zeros((0, 3), dtype=np.int32)
            action_group.create_dataset("click_history", data=click_history_array, dtype=np.int32)
        
        # 存储 click_timestamps [N]
        if click_timestamps_list:
            # 使用可变长度字符串数组
            click_timestamps_array = [ts.encode('utf-8') if isinstance(ts, str) else str(ts).encode('utf-8') for ts in click_timestamps_list]
            action_group.create_dataset(
                "click_timestamps",
                data=click_timestamps_array,
                dtype=h5py.string_dtype(encoding='utf-8')
            )
        else:
            # 创建空的可变长度字符串数组
            action_group.create_dataset(
                "click_timestamps",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding='utf-8')
            )
        
        # ========== 3. 构建 option_history 和 option_timestamps ==========
        option_history_list = []  # [(option_idx, option_label), ...]
        option_timestamps_list = []
        
        # 处理选项选择
        for select_data in option_selects:
            opt_idx = select_data.get("option_idx")
            opt_label = select_data.get("option_label", "")
            opt_timestamp = select_data.get("timestamp", execute_timestamp)
            
            if opt_idx is not None:
                option_history_list.append((opt_idx, opt_label))
                option_timestamps_list.append(opt_timestamp)
        
        # 存储 option_history [M] 结构化数组，包含选项索引和标签
        # 使用结构化数组存储 (idx, label) 对，满足 [M, 2] 的概念（虽然实际是结构化数组）
        if option_history_list:
            # 分离索引和标签
            option_indices = [item[0] for item in option_history_list]
            option_labels = [str(item[1]) if item[1] is not None else "" for item in option_history_list]
            
            # 使用结构化数组存储 [M] 个 (idx, label) 对
            dtype = [('idx', 'i4'), ('label', h5py.string_dtype(encoding='utf-8'))]
            structured_array = np.array([(idx, label) for idx, label in zip(option_indices, option_labels)], dtype=dtype)
            action_group.create_dataset("option_history", data=structured_array)
        else:
            # 创建空的结构化数组
            dtype = [('idx', 'i4'), ('label', h5py.string_dtype(encoding='utf-8'))]
            structured_array = np.array([], dtype=dtype)
            action_group.create_dataset("option_history", data=structured_array)
        
        # 存储 option_timestamps [M]
        if option_timestamps_list:
            option_timestamps_array = [ts.encode('utf-8') if isinstance(ts, str) else str(ts).encode('utf-8') for ts in option_timestamps_list]
            action_group.create_dataset(
                "option_timestamps",
                data=option_timestamps_array,
                dtype=h5py.string_dtype(encoding='utf-8')
            )
        else:
            # 创建空的可变长度字符串数组
            action_group.create_dataset(
                "option_timestamps",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding='utf-8')
            )
        
        # ========== 4. 构建 final_choice [3] ==========
        # Type=option_idx, 如果有坐标则 Val1=x, Val2=y，否则 Val1=0, Val2=0
        final_option_idx = option_idx if option_idx is not None else 0
        
        if final_coordinates:
            final_x = final_coordinates.get("x", 0)
            final_y = final_coordinates.get("y", 0)
            final_choice = np.array([final_option_idx, final_x, final_y], dtype=np.int32)
        else:
            final_choice = np.array([final_option_idx, 0, 0], dtype=np.int32)
        
        action_group.create_dataset("final_choice", data=final_choice, dtype=np.int32)
        
        # ========== 5. 存储其他可选信息 ==========
        if "done" in action_data:
            action_group.create_dataset("done", data=np.bool_(bool(action_data["done"])))
        
    except Exception as e:
        print(f"Error adding action to HDF5: {e}")
        import traceback
        traceback.print_exc()

def log_user_action_hdf5(username, env_id, episode_idx, action_data, option_list=None,
                         status=None, difficulty=None, language_goal=None, seed=None):
    """
    记录用户的详细操作到 HDF5 文件中。
    
    新的数据格式：
    每个 action 组记录一次 execute action，包含 execute 之前所有的 click 和选择的 option。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        action_data: 操作数据字典，包含：
            - option_idx: execute 时使用的选项索引（最后一次选择的）
            - option_label: execute 时使用的选项标签
            - final_coordinates: 最后执行的坐标 {"x": x, "y": y}（可选）
            - final_coords_str: 最后执行的坐标字符串（可选）
            - final_image_array: 最后执行时的图片数组（可选）
            - option_selects_before_execute: 列表，包含 execute 之前所有的选项选择
            - coordinate_clicks_before_execute: 列表，包含 execute 之前所有的坐标点击（每个元素包含 coordinates, coords_str, image_array, timestamp）
            - status: 执行状态
            - done: 是否完成
        option_list: 可选的选项列表，格式为 List[dict]，每个 dict 包含 "label" 和 "available" 字段
                    如果提供且是第一次记录 action，将存储到 metadata 中
        status: 任务状态字符串（可选），如果提供且是第一次记录 action，将存储到 metadata 中
        difficulty: 难度字符串（可选），如果提供且是第一次记录 action，将存储到 metadata 中
        language_goal: 语言目标字符串（可选），如果提供且是第一次记录 action，将存储到 metadata 中
        seed: 随机种子整数（可选），如果提供且是第一次记录 action，将存储到 metadata 中
    
    文件路径: data/user_action_logs/{username}/{env_id}_{episode_idx}.h5
    文件格式: HDF5，包含 attempt_N 组，每个 attempt 包含 metadata 和 actions 组
    """
    if not username or not env_id or episode_idx is None:
        print(f"Warning: Missing required parameters for log_user_action_hdf5: username={username}, env_id={env_id}, episode_idx={episode_idx}")
        return
    
    # 使用线程锁确保并发安全
    with lock:
        f, attempt_group, attempt_idx = _ensure_hdf5_file(username, env_id, episode_idx, create_new_attempt=False)
        if f is None or attempt_group is None:
            print(f"Error: Failed to open HDF5 file for {username}/{env_id}_{episode_idx}")
            return
        
        try:
            # 获取当前 attempt 的 actions 组
            actions_group = attempt_group["actions"]
            # 计算现有 action 的数量
            action_index = len(actions_group)  # type: ignore
            
            # 如果是第一次记录 action，存储 metadata 信息
            if action_index == 0:
                metadata_group = attempt_group["metadata"]
                
                # 存储基本字段（总是存储，因为这些是必需的）
                if "username" not in metadata_group:
                    metadata_group.create_dataset("username", data=username.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
                if "env_id" not in metadata_group:
                    metadata_group.create_dataset("env_id", data=env_id.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
                if "episode_idx" not in metadata_group:
                    metadata_group.create_dataset("episode_idx", data=np.int32(episode_idx))
                
                # 存储可选字段（如果提供）
                if status is not None and "status" not in metadata_group:
                    metadata_group.create_dataset("status", data=status.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
                if difficulty is not None and "difficulty" not in metadata_group:
                    metadata_group.create_dataset("difficulty", data=difficulty.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
                if language_goal is not None and "language_goal" not in metadata_group:
                    metadata_group.create_dataset("language_goal", data=language_goal.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
                if seed is not None and "seed" not in metadata_group:
                    metadata_group.create_dataset("seed", data=np.int32(seed))
                
                # 如果提供了 option_list，将其存储到 metadata 中
                if option_list is not None:
                    # 构建 option_list 二维数组：[[option_label, needs_coordinate], ...]
                    option_list_array = []
                    for opt in option_list:
                        option_label = opt.get("label", "")
                        needs_coordinate = bool(opt.get("available", False))
                        option_list_array.append([option_label, needs_coordinate])
                    
                    # 存储为二维数组
                    # 使用结构化数组存储，因为 HDF5 不支持混合类型的二维数组
                    # 格式: [N] 结构化数组，每个元素包含 (label, needs_coordinate)
                    if option_list_array:
                        # 创建结构化数组
                        dtype = [('label', h5py.string_dtype(encoding='utf-8')), ('needs_coordinate', np.bool_)]
                        structured_data = np.array(
                            [(item[0], bool(item[1])) for item in option_list_array],
                            dtype=dtype
                        )
                        metadata_group.create_dataset("option_list", data=structured_data)
            else:
                # 如果不是第一次记录，更新 status（如果提供且与当前不同）
                if status is not None:
                    metadata_group = attempt_group["metadata"]
                    if "status" in metadata_group:
                        # 更新现有的 status
                        del metadata_group["status"]
                    metadata_group.create_dataset("status", data=status.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
            
            # 添加操作记录到当前 attempt
            _add_action_to_hdf5(attempt_group, action_index, action_data)
            
            # 强制刷新以确保所有数据都被写入
            f.flush()
        except Exception as e:
            print(f"Error writing to HDF5 file: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if f is not None:
                try:
                    f.flush()  # 再次刷新以确保所有数据被写入
                except:
                    pass
                f.close()  # 关闭文件，这会自动保存所有更改

def has_existing_actions(username, env_id, episode_idx):
    """
    检查指定任务是否已有 actions 记录。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    
    Returns:
        bool: 如果存在 actions 则返回 True，否则返回 False
    """
    if not username or not env_id or episode_idx is None:
        return False
    
    user_dir = os.path.join(USER_ACTION_LOG_DIR, username)
    hdf5_file = os.path.join(user_dir, f"{env_id}_{episode_idx}.h5")
    
    if not os.path.exists(hdf5_file):
        return False
    
    try:
        with h5py.File(hdf5_file, "r") as f:
            current_attempt_idx = _get_current_attempt_index(f)
            if current_attempt_idx == -1:
                return False
            
            attempt_name = f"attempt_{current_attempt_idx}"
            if attempt_name not in f:
                return False
            
            attempt_group = f[attempt_name]
            if "actions" not in attempt_group:
                return False
            
            actions_group = attempt_group["actions"]
            return len(actions_group) > 0
    except Exception as e:
        print(f"Error checking existing actions: {e}")
        return False

def create_new_attempt(username, env_id, episode_idx):
    """
    为指定的任务创建新的 attempt。
    在 refresh 时调用此函数来创建新的 attempt。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    
    Returns:
        int: 新创建的 attempt 索引，如果失败返回 -1
    """
    if not username or not env_id or episode_idx is None:
        return -1
    
    with lock:
        f, attempt_group, attempt_idx = _ensure_hdf5_file(username, env_id, episode_idx, create_new_attempt=True)
        if f is None or attempt_group is None:
            print(f"Error: Failed to create new attempt for {username}/{env_id}_{episode_idx}")
            return -1
        
        try:
            f.flush()
            print(f"Created new attempt_{attempt_idx} for {username}/{env_id}_{episode_idx}")
            return attempt_idx
        except Exception as e:
            print(f"Error creating new attempt: {e}")
            import traceback
            traceback.print_exc()
            return -1
        finally:
            if f is not None:
                try:
                    f.flush()
                except:
                    pass
                f.close()

def log_user_action(username, env_id, episode_idx, action_data, option_list=None,
                     status=None, difficulty=None, language_goal=None, seed=None):
    """
    记录用户的详细操作到 HDF5 文件中。
    
    新的数据格式：
    每个 action 组记录一次 execute action，包含 execute 之前所有的 click 和选择的 option。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        action_data: 操作数据字典，包含：
            - option_idx: execute 时使用的选项索引（最后一次选择的）
            - option_label: execute 时使用的选项标签
            - final_coordinates: 最后执行的坐标 {"x": x, "y": y}（可选）
            - final_coords_str: 最后执行的坐标字符串（可选）
            - final_image_array: 最后执行时的图片数组（可选）
            - option_selects_before_execute: 列表，包含 execute 之前所有的选项选择
            - coordinate_clicks_before_execute: 列表，包含 execute 之前所有的坐标点击（每个元素包含 coordinates, coords_str, image_array, timestamp）
            - status: 执行状态
            - done: 是否完成
        option_list: 可选的选项列表，格式为 List[dict]，每个 dict 包含 "label" 和 "available" 字段
        status: 任务状态字符串（可选），如果提供且是第一次记录 action，将存储到 metadata 中
        difficulty: 难度字符串（可选），如果提供且是第一次记录 action，将存储到 metadata 中
        language_goal: 语言目标字符串（可选），如果提供且是第一次记录 action，将存储到 metadata 中
        seed: 随机种子整数（可选），如果提供且是第一次记录 action，将存储到 metadata 中
    
    文件路径: data/user_action_logs/{username}/{env_id}_{episode_idx}.h5
    文件格式: HDF5，包含 actions 组，每个 action 记录一次 execute，包含 execute 之前所有的 option_select 和 coordinate_clicks
    """
    # 直接调用 HDF5 版本
    log_user_action_hdf5(username, env_id, episode_idx, action_data, option_list=option_list,
                         status=status, difficulty=difficulty, language_goal=language_goal, seed=seed)
