from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


#HistoryBench
import matplotlib.pyplot as plt
import h5py
import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from historybench.HistoryBench_env.util import *


def sequential_task_check(self, tasks,allow_subgoal_change_this_timestep):
    """
    带任务名称和demonstration标志的序列任务检查函数

    Args:
        tasks: 任务列表，每个元素是包含 "func"、"name"、"demonstration"、可选
               "failure_func"、"solve" 键的字典，或旧格式的元组
               (task_func, task_name[, demonstration[, failure_func[, solve]]])

    Returns:
        tuple: (all_completed: bool, current_task_name: str, task_failed: bool)
            - all_completed: 如果所有任务都完成返回True，否则返回False
            - current_task_name: 当前任务的名称
            - task_failed: 当前任务是否触发失败条件

    使用示例:
        tasks = [
            {
                "func": lambda: is_obj_pickup(self, obj=self.cube_0),
                "name": "Pick up cube 0",
                "demonstration": True,
                "failure_func": lambda: self.some_failure_condition(),
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.cube_0),
            },
            {
                "func": lambda: is_obj_pickup(self, obj=self.cube_1),
                "name": "Pick up cube 1",
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.cube_1),
            },
            {
                "func": lambda: is_obj_at_location(self, obj=self.cube_1, location=self.target),
                "name": "Place cube 1",
                "demonstration": True,
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, obj=self.cube_1, target=self.target),
            },
        ]
        all_completed, current_task, task_failed = sequential_task_check(self, tasks)
    """
    # 将任务条目标准化为字典格式，兼容旧的二元组/三元组定义
    if not hasattr(self, '_timelimit_deadlines'):
        self._timelimit_deadlines = {}

    normalized_tasks = []
    for task in tasks:
        if isinstance(task, dict):
            # 拷贝以避免在原始数据上产生副作用
            task_entry = dict(task)
            func = task_entry.get("func") or task_entry.get("task_func")
            if func is None:
                raise KeyError("Task dictionary must contain a 'func' callable")
            name = task_entry.get("name") or task_entry.get("task_name") or "Unknown"
            demonstration = task_entry.get("demonstration")
            if demonstration is None:
                demonstration = task_entry.get("demo", False)
            failure_func = task_entry.get("failure_func") or task_entry.get("failure")
            solve_callable = task_entry.get("solve")
            segment=task_entry.get("segment")
            subgoal_segment=task_entry.get("subgoal_segment")

            task_entry["func"] = func
            task_entry["name"] = name
            task_entry["demonstration"] = bool(demonstration)
            task_entry["failure_func"] = failure_func
            task_entry["solve"] = solve_callable
            task_entry['segment']=segment
            task_entry['subgoal_segment']=subgoal_segment
            normalized_tasks.append(task_entry)



        # else:
        #     if len(task) == 2:
        #         func, name = task
        #         demonstration = False
        #         failure_func = None
        #         solve_callable = None
        #     elif len(task) == 3:
        #         func, name, demonstration = task
        #         failure_func = None
        #         solve_callable = None
        #     elif len(task) == 4:
        #         func, name, demonstration, failure_func = task
        #         solve_callable = None
        #     else:
        #         func, name, demonstration, failure_func, solve_callable = task[:5]
        #     if len(task) < 2:
        #         raise ValueError("Task entries must be dicts or tuples/lists with at least 2 items")
        #     normalized_tasks.append({
        #         "func": func,
        #         "name": name,
        #         "demonstration": bool(demonstration),
        #         "failure_func": failure_func,
        #         "solve": solve_callable,
        #     })

    # 获取任务数量
    num_tasks = len(normalized_tasks)

    # 如果没有任务，直接返回True
    if num_tasks == 0:
        # 设置当前任务信息为空
        self.current_task_index = -1
        self.current_task_name = "No tasks"
        self.current_task_demonstration = False
        self.current_task_failure = False
        self.current_task_solve = None
        return True, "No tasks", False,None

    # 初始化timestep（如果不存在）
    if not hasattr(self, 'timestep'):
        self.timestep = 0

    # 确保timestep不超过任务数量
    if self.timestep >= num_tasks:
        # 所有任务已完成
        self.current_task_index = num_tasks
        self.current_task_name = "All tasks completed"
        self.current_task_demonstration = False
        self.current_task_failure = False
        self.current_task_solve = None
        return True, "All tasks completed", False,None

    # 获取当前任务
    task_entry = normalized_tasks[self.timestep]
    current_task_func = task_entry["func"]
    current_task_name = task_entry.get("name", "Unknown")
    current_demonstration = task_entry.get("demonstration", False)
    current_failure_func = task_entry.get("failure_func")
    current_task_specialflag=task_entry.get("specialflag", None)
    current_segment=task_entry.get("segment",None)
    current_subgoal_segment=task_entry.get("subgoal_segment",None)
    # 设置当前任务信息，供RecordWrapper使用
    if allow_subgoal_change_this_timestep==True:
        self.current_task_index = self.timestep
        self.current_task_name = current_task_name
        self.current_task_demonstration = current_demonstration
        self.current_task_failure = False
        self.current_task_solve = task_entry.get("solve")
        self.current_segment=current_segment
        self.current_subgoal_segment=current_subgoal_segment
    self.current_task_name_online = current_task_name#实时subgoal
    self.current_subgoal_segment_online=current_subgoal_segment#实时subgoalsegment
    self.current_segment_online=current_segment#实时segmentonline

    self.current_task_specialflag=current_task_specialflag

    # 如果切换到了新的任务，重置静态检查相关状态
    last_task_index = getattr(self, "_last_task_index", None)
    if last_task_index != self.timestep:
        if hasattr(self, "first_timestep"):
            delattr(self, "first_timestep")
        _clear_timelimit_deadline(self, last_task_index)
    self._last_task_index = self.timestep

    # 先检查失败条件
    failure_triggered = False
    task_idx = self.timestep
    if current_failure_func is not None:
        if callable(current_failure_func):
            try:
                failure_result = current_failure_func()
            except Exception as exc:  # pragma: no cover - 防御性
                display_index = self.timestep + 1
                print(f"Task {display_index} failure check raised exception: {exc}")
                failure_triggered = True
            else:
                failure_triggered = _coerce_failure_result(failure_result)
        else:
            failure_triggered = _coerce_failure_result(current_failure_func)

    if failure_triggered:
        self.current_task_failure = True
        _clear_timelimit_deadline(self, task_idx)
        display_index = self.timestep + 1
        print(f"Task {display_index} failed: {current_task_name}")
        return False, current_task_name, True,current_task_specialflag

    # 执行当前任务检查
    if current_task_func():
        display_index = self.timestep + 1
        print(f"Task {display_index} completed: {current_task_name}")
        _clear_timelimit_deadline(self, task_idx)

        # 检查是否是最后一个任务
        if self.timestep == num_tasks - 1:
            # 所有任务完成，确保timestep超出范围以避免重复检查
            self.timestep = num_tasks
            self.current_task_index = num_tasks
            self.current_task_name = "All tasks completed"
            self.current_task_demonstration = False
            print(f"All {num_tasks} tasks completed successfully!")
            return True, "All tasks completed", False,None
        else:
            # 进入下一个timestep
            self.timestep += 1
            # 获取下一个任务的名称
            next_task_name = normalized_tasks[self.timestep].get("name", "Unknown")
            return False, next_task_name, False,None  # 还有后续任务

    return False, current_task_name, False,current_task_specialflag  # 当前任务未完成

def _coerce_failure_result(value):
    """Normalize various failure_func return types into a boolean."""
    if isinstance(value, (list, tuple, set)):
        return any(_coerce_failure_result(item) for item in value)
    if isinstance(value, dict):
        return any(_coerce_failure_result(item) for item in value.values())
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.detach().cpu().bool().any().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.any(value))
    try:
        return bool(value)
    except ValueError:
        try:
            iterator = iter(value)
        except TypeError:
            raise
        return any(_coerce_failure_result(item) for item in iterator)

def _clear_timelimit_deadline(self, task_index):
    if task_index is None:
        return
    deadlines = getattr(self, '_timelimit_deadlines', None)
    if isinstance(deadlines, dict):
        deadlines.pop(task_index, None)

def timewindow(self, func, timewindow_timer,min_steps=300, max_steps=500):
    """
    包装任意函数，使其只在指定的时间窗口内（min_steps到max_steps之间）才能返回True
    从第一次调用时开始计数

    Args:
        func: 要包装的函数（例如 lambda: is_button_pressed(self, obj=self.button)）
        min_steps: 时间窗口起始步数（默认300）
        max_steps: 时间窗口结束步数（默认500）
        timewindow_timer: 计时器编号，用于区分不同的时间窗口（默认0）

    Returns:
        bool: 如果在时间窗口内且func返回True则返回True，否则返回False
    """
    if not hasattr(self, '_timewindow_timers'):
        self._timewindow_timers = {}

    current_step = int(getattr(self, "elapsed_steps", 0))

    # 如果计时器不存在，则开始计数
    if timewindow_timer not in self._timewindow_timers:
        self._timewindow_timers[timewindow_timer] = current_step
        print(f"Timewindow timer {timewindow_timer} started at step {current_step}")

    # 获取起始步数（继续之前的计数）
    start_step = self._timewindow_timers[timewindow_timer]
    elapsed = current_step - start_step

    # 如果还没到时间窗口，返回False
    if elapsed < min_steps:
        return False

    # 如果超过时间窗口，返回False（任务失败）
    if elapsed > max_steps:
        return False

    # 在时间窗口内，调用被包装的函数
    return func()


def in_demonstration(self):
    if self.use_demonstrationwrapper==True:
        return True
    else:
        return False


def check_block_away_gripper(self,obj):
    gripper_open_flag=False
    away_flag=False

    qpos=self.agent.robot.get_qpos().tolist()[0]
    last_two = qpos[-2:]

    if  all(x > 0.02 for x in last_two):
        gripper_open_flag=True

    gripper_pos = self.agent.tcp.pose.p.tolist()[0]
    obj_pos= obj.pose.p.tolist()[0]
    gripper_pos=torch.as_tensor(gripper_pos, dtype=torch.float32).flatten()[:3]
    obj_pos=torch.as_tensor(obj_pos, dtype=torch.float32).flatten()[:3]

    distance = np.linalg.norm(obj_pos - gripper_pos)
    if distance>0.02:
        away_flag=True

    flag=gripper_open_flag and away_flag
    return flag

def is_obj_pickup(self, obj, goal_pos=None):
    # if in_demonstration(self):
    #     obj_lifted = obj.pose.p[:, 2] > 0.05
    #     return obj_lifted

    # else:

        # 检查物体的z坐标是否大于0.05
        obj_lifted = obj.pose.p[:, 2] > 0.05

        # 检查robot是否真正抓住了物体
        is_grasping = self.agent.is_grasping(obj)

        result = obj_lifted & is_grasping

        return result    

def is_any_obj_pickup_flag_currentpickup(self, objects):
    # 仅记录当前被抓起的方块引用，不在此处更新计数；计数统一在环境 step 中处理
    for obj in objects:
        if is_obj_pickup(self,obj):
            self.currentpickup=obj
            print(f"currentpickup={obj}")
            return True
    return False


def is_obj_dropped(self, obj):
    # 获取物体和目标的位置
    obj_pos = obj.pose.p[0]  # [x, y, z]

    # 检查物体是否未被抓取
    is_grasping = self.agent.is_grasping(obj)

    gripper_pos = self.agent.tcp.pose.p.tolist()[0]


    if in_demonstration(self):
        # 只有当物体在目标位置附近且未被抓取时才返回True
        if obj_pos[2] <=0.035 and not is_grasping:
            return True
    else:
        if obj_pos[2] <= 0.2 and not is_grasping and gripper_pos[2] > 0.05:
            return True
    return False

def is_obj_dropped_currentpickup(self,list):
    current_obj = getattr(self, "currentpickup", None)
    if current_obj is None:
        return False

    if not is_obj_dropped(self, current_obj):
        return False

    # 这里只负责清空 currentpickup，真正的放下计数在环境 step 中完成
    self.currentpickup = None
    return True

def is_bin_putdown(self, obj, goal_pos=None):
    # 检查物体的z坐标是否大于0.3

    is_grasping = self.agent.is_grasping(obj)
    gripper_pos = self.agent.tcp.pose.p.tolist()[0]
    if obj.pose.p[:, 2] <= 0.07 and not is_grasping and gripper_pos[2] > 0.05:
        return True
    return False
def is_reset(self):
    gripper = self.agent.tcp.pose.p

    if isinstance(gripper, torch.Tensor):
        gripper_pos = gripper.detach().cpu().numpy()
    else:
        gripper_pos = np.asarray(gripper, dtype=np.float32)

    if gripper_pos.ndim > 1:
        gripper_pos = gripper_pos[0]

    target_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
    distance = np.linalg.norm(gripper_pos - target_pos)
    return float(distance) < 0.1

def is_bin_pickup(self, obj,):
    # 检查物体的z坐标是否大于0.3
    is_bin_pickup = obj.pose.p[:, 2] > 0.15
     # 检查robot是否真正抓住了物体
    return is_bin_pickup 

def is_any_bin_pickup(self, objects):
    """Return True if any object in the iterable is picked up."""
    for obj in objects:
        if is_bin_pickup(self, obj=obj):
            return True
    return False


def is_A_pickup_notB(self, A, B):
    # 检查物体A的z坐标是否大于0.1
    is_obj_pickup = A.pose.p[:, 2] > 0.1
    
    # 获取gripper位置
    gripper_pos = self.agent.tcp.pose.p
    
    # 计算A和gripper的距离
    dist_A_gripper = np.linalg.norm(A.pose.p - gripper_pos, axis=-1)
    
    # 计算B和gripper的距离
    dist_B_gripper = np.linalg.norm(B.pose.p - gripper_pos, axis=-1)
    
    # A距离gripper更近
    is_A_closer = dist_A_gripper < dist_B_gripper
    
    # 两个条件都满足

    return is_obj_pickup & is_A_closer
def is_A_insert_notB(self, A, B,box,direction=None,mark_end_flag=False,threashold=0.05):
    """Check peg insertion with optional direction constraint."""
    def _to_np(vec):
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    A_pos = _to_np(A.pose.p)
    B_pos = _to_np(B.pose.p)
    box_pos = _to_np(box.pose.p)

    is_obj_insert = np.linalg.norm(A_pos - box_pos, axis=-1) < threashold
    dist_A_gripper = np.linalg.norm(A_pos - box_pos, axis=-1)
    dist_B_gripper = np.linalg.norm(B_pos - box_pos, axis=-1)
    is_A_closer = dist_A_gripper < dist_B_gripper

    direction_ok = True
    if direction is not None:
        gripper_pos = _to_np(self.agent.tcp.pose.p)
        side_indicator = gripper_pos[1] - box_pos[1]
        if abs(side_indicator) < 1e-3:
            side_indicator = B_pos[1] - box_pos[1]
        direction_ok = side_indicator * direction < 0

    success = bool(is_obj_insert and is_A_closer and direction_ok)
    if success and mark_end_flag:
        print("marked end step!",self.elapsed_steps+3)
        self.end_steps=int(getattr(self, "elapsed_steps", 0))
    return success




def restore_finish(self):
    peg_pos = np.array(self.peg.pose.p.tolist()[0])
    init_pos = np.array(self.peg_init_pose.p.tolist()[0])
    flag=np.linalg.norm(peg_pos-init_pos)<0.05

    return flag




def is_any_obj_pickup(self, objects):
    """Return True if any object in the iterable is picked up."""
    for obj in objects:
        if is_obj_pickup(self, obj=obj):
            return True
    return False

def correct_timestep(self, time_range=None, stop_timestep=None):
    """
    Failure helper for timing-based tasks.

    Returns True (failure) when the recorded stop timestep falls outside the
    allowed time window or when we have already exceeded the window without
    recording a stop timestep.
    """

    min_step, max_step = time_range
    current_step = int(getattr(self, "elapsed_steps", 0))



    if min_step <= stop_timestep <= max_step:
        return True
    
    return False


def is_obj_stopped_onto(self, obj, target, stop):
    # 获取物体和目标的位置
    obj_pos = obj.pose.p[0]  # [x, y, z]
    target_pos = target.pose.p[0]  # [x, y, z]

    # 计算水平距离（忽略z轴）
    horizontal_distance = torch.sqrt(
        (obj_pos[0] - target_pos[0])**2 +
        (obj_pos[1] - target_pos[1])**2
    )

    # 设置距离阈值
    distance_threshold = self.cube_half_size*(2.5)

    # 只有当物体在目标位置附近且已经停止移动时才返回True
    stop_ok = stop 
    #print("stop_ok",stop_ok,"horizontal_distance",horizontal_distance <= distance_threshold)
    if horizontal_distance <= distance_threshold and stop_ok:
        if getattr(self, "stop_timestep", None) is None:
            self.stop_timestep = int(getattr(self, "elapsed_steps", 0))
        return True

    return False

def is_all_obj_dropped(self, objects):
    return all(is_obj_dropped(self, obj) for obj in objects)

def is_obj_swing_onto(self, obj, target, achieved_list=None,distance_threshold=0.01,z_threshold=0.1):
    
    # 获取物体和目标的位置
    obj_pos = obj.pose.p[0]  # [x, y, z]
    target_pos = target.pose.p[0]  # [x, y, z]

    # 计算水平距离（忽略z轴）
    horizontal_distance = torch.sqrt(
        (obj_pos[0] - target_pos[0])**2 +
        (obj_pos[1] - target_pos[1])**2
    )

    # 设置距离阈值0.01
         #再小会检测不到第一个摆动目标

    z_flag=obj_pos[2]<z_threshold

    if horizontal_distance <= distance_threshold and z_flag:

        return True

    return False


def is_obj_dropped_onto(self, obj, target):
    # 获取物体和目标的位置
    obj_pos = obj.pose.p[0]  # [x, y, z]
    target_pos = target.pose.p[0]  # [x, y, z]

    # 计算水平距离（忽略z轴）
    horizontal_distance = torch.sqrt(
        (obj_pos[0] - target_pos[0])**2 +
        (obj_pos[1] - target_pos[1])**2
    )

    # 设置距离阈值
    distance_threshold = 0.05
    # 只有当物体在目标位置附近且未被抓取时才返回True
    if horizontal_distance <= distance_threshold and is_obj_dropped(self,obj):
        return True

    return False


def is_obj_pushed_onto(self, obj, target,distance_threshold=None,must_gripper_open=False):


    if must_gripper_open==True:
        qpos=self.agent.robot.get_qpos().tolist()[0]
        last_two = qpos[-2:]

        if  not(all(x > 0.02 for x in last_two)):
            return False

    # 获取物体和目标的位置
    obj_pos = obj.pose.p[0]  # [x, y, z]
    target_pos = target.pose.p[0]  # [x, y, z]

    # 计算水平距离（忽略z轴）
    horizontal_distance = torch.sqrt(
        (obj_pos[0] - target_pos[0])**2 +
        (obj_pos[1] - target_pos[1])**2
    )

    # 设置距离阈值
    if distance_threshold is None:
        distance_threshold = self.cube_half_size * 2 * 1.2

    # 只有当物体在目标位置附近且未被抓取时才返回True
    if horizontal_distance <= distance_threshold:
        return True

    return False


def gripper_direction_correct(self,target,direction):
    if direction==-1:
        gripper_pos = self.agent.tcp.pose.p[0]
        target_pos = target.pose.p[0]
        print(gripper_pos[1]>target_pos[1])#y>y on the right side
        return gripper_pos[1]>target_pos[1]
    else:
        gripper_pos = self.agent.tcp.pose.p[0]
        target_pos = target.pose.p[0]
        print(gripper_pos[1]<target_pos[1])#y<y on the left side
        return gripper_pos[1]<target_pos[1]
    
    
def is_obj_pushed_onto_byAnotB_wDirection(self, obj, target, A, B,direction=None):
    """
    Check if object is pushed onto target by A (not B).
    A must be closer to obj than B.

    Args:
        self: environment instance
        obj: the object being pushed
        target: the target position
        A: the pusher that should be closer (e.g., robot TCP)
        B: the pusher that should be farther (e.g., another object)

    Returns:
        bool: True if obj is at target and A is closer to obj than B
    """
    # 首先检查物体是否在目标位置
    if not is_obj_pushed_onto(self, obj, target,distance_threshold=self.cube_half_size * 2 * 1.2):
        return False

    # 获取位置
    obj_pos = obj.pose.p[0]  # [x, y, z]
    A_pos = A.pose.p[0]  # [x, y, z]
    B_pos = B.pose.p[0]  # [x, y, z]

    # 计算A到obj的距离
    distance_A_to_obj = torch.sqrt(
        (obj_pos[0] - A_pos[0])**2 +
        (obj_pos[1] - A_pos[1])**2 +
        (obj_pos[2] - A_pos[2])**2
    )

    # 计算B到obj的距离
    distance_B_to_obj = torch.sqrt(
        (obj_pos[0] - B_pos[0])**2 +
        (obj_pos[1] - B_pos[1])**2 +
        (obj_pos[2] - B_pos[2])**2
    )

    # A必须比B更靠近obj
    if distance_A_to_obj < distance_B_to_obj:
        #if gripper_direction_correct(self,target,direction):
            return True

    return False

def is_obj_swing_onto_any(self, obj, targets):
    """Check if object swings onto any of the targets in the list."""
    for target in targets:
        if is_obj_swing_onto(self, obj=obj, target=target):
            print(f"failure:swing onto {target}") 
            return True
    return False

def too_many_swings(self):
    # 读取环境里设置的 swing_over_limit 标志；True 表示摆动次数超出允许上限
    return getattr(self, "swing_over_limit", False)

def is_any_obj_dropped_onto_delete(self, objects, target):
    for obj in objects:
        if is_obj_dropped_onto_delete(self, obj, target):
            if obj in self.red_cubes:
                self.red_cubes_in_bin+=1
            elif obj in self.blue_cubes:
                self.blue_cubes_in_bin+=1
            elif obj in self.green_cubes:
                self.green_cubes_in_bin+=1
            print(f"red_cubes_in_bin={self.red_cubes_in_bin},blue_cubes_in_bin={self.blue_cubes_in_bin},green_cubes_in_bin={self.green_cubes_in_bin}")
            return True


    return False

def is_obj_dropped_onto_delete(self, obj, target):
    # 如果物体在目标附近且高度足够低，删除物体

    if is_obj_dropped_onto(self,obj,target) and check_block_away_gripper(self,obj):
        # 删除物体：将其移动到远离场景的位置
        with torch.no_grad():
            # 移动到场景外的位置
            obj.set_pose(sapien.Pose(p=[10.0, 10.0, 0.0]))
        return True
    return False

def is_obj_dropped_onto_any(self, obj, target):
    """Check if object is dropped onto any of the targets in the list."""
    for t in target:
        if is_obj_dropped_onto(self, obj=obj, target=t):
            return True
    return False

def is_static(self, threshold: float = 0.2):
    qvel = self.agent.robot.get_qvel()[..., :-2]
    return torch.max(torch.abs(qvel), 1)[0] <= threshold


def reset_check(self,gripper=None,target_qpos=None):
    if target_qpos==None:
        target_qpos=reset_panda.get_reset_panda_param("qpos",gripper=gripper)
    current_qpos=self.agent.robot.qpos
    if torch.max(torch.abs(current_qpos - target_qpos)) < 0.01:
        return True
    return False

def button_hover(self,button,distance_threshold=0.03,z_threshold=0.2):
     # 获取物体和目标的位置
    obj_pos =self.agent.tcp.pose.p[0]
    target_pos = button.pose.p[0]

    # 计算水平距离（忽略z轴）
    horizontal_distance = torch.sqrt(
        (obj_pos[0] - target_pos[0])**2 +
        (obj_pos[1] - target_pos[1])**2
    )

    # 设置距离阈值0.01

    z_flag=obj_pos[2]<z_threshold

    if horizontal_distance <= distance_threshold and z_flag:
        return True

    return False

def before_absTimestep(self,absTimestep):
    if int(getattr(self, "elapsed_steps", 0))<absTimestep:
        return False
    else:
        return True

def static_check(self, timestep, static_steps=10):
    """
    静态检查函数，记录第一次调用的timestep，当达到指定静止步数后返回成功
    如果is_static返回False，则重新开始计数

    Args:
        timestep: 当前timestep
        static_steps: 需要保持静止的步数，默认为10

    Returns:
        bool: 如果timestep达到初始记录的timestep+static_steps则返回True，否则返回False
    """
    # 检查机器人是否静止
    if not is_static(self):
        # 如果不静止，重新开始计数
        self.first_timestep = timestep
        #print(f"Robot not static, restarting count at timestep: {timestep}")
        return False

    # 初始化first_timestep（如果不存在）
    if not hasattr(self, 'first_timestep'):
        self.first_timestep = timestep
        print(f"Static check initialized at timestep: {timestep}")

    # 检查是否达到目标timestep（初始timestep + static_steps）
    target_timestep = self.first_timestep + static_steps
    current_progress = timestep - self.first_timestep

    if current_progress >= static_steps:
        setattr(self, "_static_deadline", None)
        return True
    else:
        return False


def get_button_depth(self,obj):
    """返回按钮按下深度（米），0=未按，越大表示按得越深。支持向量化并行环境。"""
    assert hasattr(self, "button"), "还没有创建按钮 (_build_button)"
    qpos = obj.get_qpos()  # 形状通常是 (B, 1) 或 (1,)
    depth = -(qpos[..., 0])  # 我们把 [-travel, 0] 取负号变成 [0, travel]
    return depth

def is_button_pressed(self, obj):
    flag=False
    depth = get_button_depth(self,obj=obj)#0.015
    #print(depth)
    if depth > 0.005:
        flag=True

    return flag


def is_any_button_pressed_removelist(self, button_list):
    """
    Return True if any button in the provided list is pressed and remove those buttons from the list.

    Args:
        button_list (MutableSequence): sequence of button objects to check.

    Returns:
        bool: True if at least one button was pressed during this call.
    """
    if not button_list:
        return False

    pressed_found = False
    # Iterate over a copy so we can safely mutate the original list.
    for button in list(button_list):
        if is_button_pressed(self, button):
            pressed_found = True
            button_list.remove(button)

    return pressed_found

def check_in_bin_number(self, in_bin_list, total_number_list):
    """
    检查两个列表中的元素是否一一对应相等

    Args:
        in_bin_list: 包含当前bin中的数量列表，例如 [self.red_cubes_in_bin, self.blue_cubes_in_bin, self.green_cubes_in_bin]
        total_number_list: 包含目标数量列表，例如 [self.red_cubes_target_number, self.blue_cubes_target_number, self.green_cubes_target_number]

    Returns:
        bool: 如果所有元素一一对应相等返回True，否则返回False
    """
    # 检查列表长度是否相同
    if len(in_bin_list) != len(total_number_list):
        return False

    # 检查每个元素是否一一对应相等
    for in_bin, target in zip(in_bin_list, total_number_list):
        if in_bin != target:
            print(f"in_bin={in_bin},target={target}")
            return False

    return True


def direction(current_target, last_target,direction=8):
    """
    Return the closest compass direction from last_target to current_target
    using the xy plane (up, down, left, right and four diagonals).
    """

    def _extract_xy(target):
        if not hasattr(target, "pose"):
            raise ValueError("Target must have pose information to compute direction.")
        pos = target.pose.p
        if isinstance(pos, torch.Tensor):
            coords = pos.detach().cpu().numpy()
        else:
            coords = np.asarray(pos, dtype=np.float32)
        if coords.ndim > 1:
            coords = coords[0]
        if coords.shape[0] < 2:
            raise ValueError("Pose must provide at least x and y coordinates.")
        return coords[:2]

    current_xy = _extract_xy(current_target)
    last_xy = _extract_xy(last_target)
    delta = current_xy - last_xy
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        return "same"

    delta /= norm
    diag = np.sqrt(2.0)
    if direction ==8:
        direction_vectors = {
            "forward": np.array([1.0, 0.0]),
            "backward": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]),
            "right": np.array([0.0, -1.0]),
            "forward-left": np.array([1.0, 1.0]) / diag,
            "forward-right": np.array([1.0, -1.0]) / diag,
            "backward-left": np.array([-1.0, 1.0]) / diag,
            "backward-right": np.array([-1.0, -1.0]) / diag,
        }
    elif direction ==4:
        direction_vectors = {
            "forward": np.array([1.0, 0.0]),
            "backward": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]),
            "right": np.array([0.0, -1.0]),
        }


    best_direction = max(
        direction_vectors.items(), key=lambda item: float(np.dot(delta, item[1]))
    )[0]
    return best_direction
