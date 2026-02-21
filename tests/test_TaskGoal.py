"""
test_TaskGoal.py

测试 task_goal.get_language_goal 的所有 env_id 分支，每种情况各打印一次。
运行: uv run python -m pytest tests/test_TaskGoal.py -s

每个 env_id 测试情况与传入参数概要：
- BinFill: 1/2/3 种颜色有数量（0 种不测）→ unwrapped: red/blue/green_cubes_target_number
- PickXtimes: 1 次 / 多次 → unwrapped: num_repeats, target_color_name
- SwingXtimes: 1 次 / 多次 → unwrapped: num_repeats, target_color_name
- VideoUnmask: pick=1 / pick>1 → unwrapped: color_names, configs；self.difficulty
- VideoUnmaskSwap: pick_times=1 / 2 → unwrapped: color_names；self.pick_times
- ButtonUnmask: pick=1 / pick>1 → unwrapped: color_names, configs；self.difficulty
- ButtonUnmaskSwap: pick_times=1 / 2 → unwrapped: color_names；self.pick_times
- VideoPlaceButton: 单一 → self.target_color_name, target_target_language
- VideoPlaceOrder: 单一 → self.target_color_name, which_in_subset
- PickHighlight: 单一 → 无
- VideoRepick: 1 次 / 多次 → self.num_repeats
- StopCube: 单一 → unwrapped: stop_time
- InsertPeg / MoveCube / PatternLock / RouteStick: 单一 → 无
"""
from pathlib import Path
import importlib.util
import types


def _load_task_goal_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "robomme" / "robomme_env" / "utils" / "task_goal.py"
    spec = importlib.util.spec_from_file_location("task_goal_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_task_goal_module()
get_language_goal = mod.get_language_goal


class _Unwrapped:
    """模拟 env.unwrapped，可按需设置任意属性。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Env:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped


class _Self:
    """模拟被调用方法的 self 对象。"""
    def __init__(self, env, **kwargs):
        self.env = env
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_self(unwrapped_attrs=None, **self_attrs):
    unwrapped = _Unwrapped(**(unwrapped_attrs or {}))
    env = _Env(unwrapped)
    return _Self(env, **self_attrs)


def _call(env_id, mock_self):
    result = get_language_goal(mock_self, env_id)
    print()
    for idx, goal in enumerate(result, start=1):
        print(f"[{env_id}] goal{idx}: {goal}")
    return result


def test_unknown_env_returns_single_goal_when_equal():
    """未知 env_id 走默认空字符串分支，两个 goal 相同时返回单元素列表。"""
    s = _make_self()
    result = _call("UnknownEnv", s)
    assert result == [""]


def test_movecube_still_returns_two_goals():
    """已定义分支且两条文案不同，仍返回双元素列表。"""
    s = _make_self()
    result = _call("MoveCube", s)
    assert len(result) == 2


# ── BinFill: 3 branches（0 种颜色不测）──
# 测试情况：按「有目标数量的颜色种类」分 1/2/3 种，决定句子是单色/两色并列/三色并列。
# 传入：env.unwrapped 的 red_cubes_target_number, blue_cubes_target_number, green_cubes_target_number。

def test_binfill_one_color():
    """情况：1 种颜色有数量 → 单色句。传入：仅 red=3，蓝绿为 0。"""
    s = _make_self(unwrapped_attrs=dict(
        red_cubes_target_number=3,
        blue_cubes_target_number=0,
        green_cubes_target_number=0,
    ))
    result = _call("BinFill", s)
    assert "three red cubes" in result[0]
    assert " and " not in result[0]
    assert "three red cubes" in result[1]


def test_binfill_two_colors():
    """情况：2 种颜色有数量 → 「A and B」句式。传入：red=1, blue=2, green=0。"""
    s = _make_self(unwrapped_attrs=dict(
        red_cubes_target_number=1,
        blue_cubes_target_number=2,
        green_cubes_target_number=0,
    ))
    result = _call("BinFill", s)
    assert "one red cube" in result[0]
    assert "two blue cubes" in result[0]
    assert " and " in result[0]
    assert "one red cube" in result[1]
    assert "two blue cubes" in result[1]


def test_binfill_three_colors():
    """情况：3 种颜色有数量 → 「A, B and C」句式。传入：red=2, blue=3, green=1。"""
    s = _make_self(unwrapped_attrs=dict(
        red_cubes_target_number=2,
        blue_cubes_target_number=3,
        green_cubes_target_number=1,
    ))
    result = _call("BinFill", s)
    assert "two red cubes" in result[0]
    assert "three blue cubes" in result[0]
    assert "one green cube" in result[0]
    assert "two red cubes" in result[1]
    assert "three blue cubes" in result[1]
    assert "one green cube" in result[1]


# ── PickXtimes: 2 branches ──
# 测试情况：重复次数 1 次 vs 多次，决定是否出现「repeating this action N times」。
# 传入：env.unwrapped 的 num_repeats, target_color_name。

def test_pickxtimes_once():
    """情况：只捡 1 次 → 无「repeating」。传入：num_repeats=1, target_color_name=red。"""
    s = _make_self(unwrapped_attrs=dict(num_repeats=1, target_color_name="red"))
    result = _call("PickXtimes", s)
    assert "repeating" not in result[0]
    assert "red cube" in result[0]
    if len(result) > 1:
        assert "repeating" not in result[1]
        assert "red cube" in result[1]


def test_pickxtimes_multiple():
    """情况：捡多次 → 有「repeating this action N times」。传入：num_repeats=3, target_color_name=blue。"""
    s = _make_self(unwrapped_attrs=dict(num_repeats=3, target_color_name="blue"))
    result = _call("PickXtimes", s)
    assert "repeating this action three times" in result[0]
    assert "blue cube" in result[0]
    assert "pick-and-place action three times" in result[1]
    assert "blue cube" in result[1]


# ── SwingXtimes: 2 branches ──
# 测试情况：摆动 1 次 vs 多次，1 次用「put it down on the left-side target」，多次用「repeating this back and forth motion N times」。
# 传入：env.unwrapped 的 num_repeats, target_color_name。

def test_swingxtimes_once():
    """情况：摆动 1 次 → 左目标放下句式，无 repeating。传入：num_repeats=1, target_color_name=green。"""
    s = _make_self(unwrapped_attrs=dict(num_repeats=1, target_color_name="green"))
    result = _call("SwingXtimes", s)
    assert "put it down on the left-side target" in result[0]
    assert "repeating" not in result[0]
    assert "left-side target" in result[1]
    assert "repeating" not in result[1]


def test_swingxtimes_multiple():
    """情况：摆动多次 → repeating back and forth N times。传入：num_repeats=5, target_color_name=red。"""
    s = _make_self(unwrapped_attrs=dict(num_repeats=5, target_color_name="red"))
    result = _call("SwingXtimes", s)
    assert "repeating this back and forth motion five times" in result[0]
    assert "right-to-left swing motion five times" in result[1]


# ── VideoUnmask: 2 branches ──
# 测试情况：按难度 config 里 pick 为 1 或 >1，决定只描述第一个容器还是「再捡第二个容器」。
# 传入：env.unwrapped 的 color_names, configs；self.difficulty（用于 configs[difficulty]['pick']）。

def test_videounmask_pick_one():
    """情况：pick=1 → 只描述捡第一个容器。传入：color_names, configs[easy][pick]=1, difficulty=easy。"""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["red", "blue", "green"],
            configs={"easy": {"pick": 1}},
        ),
        difficulty="easy",
    )
    result = _call("VideoUnmask", s)
    assert "red cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container" not in g


def test_videounmask_pick_two():
    """情况：pick>1 → 描述捡两个容器（第一色 + 第二色）。传入：configs[hard][pick]=2, difficulty=hard。"""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["red", "blue", "green"],
            configs={"hard": {"pick": 2}},
        ),
        difficulty="hard",
    )
    result = _call("VideoUnmask", s)
    assert "red cube" in result[0]
    assert "another container hiding the blue cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container hiding the blue cube" in g


# ── VideoUnmaskSwap: 2 branches ──
# 测试情况：self.pick_times 为 1 或 2，决定只描述第一个容器还是两个容器。
# 传入：env.unwrapped 的 color_names；self.pick_times。

def test_videounmaskswap_pick_one():
    """情况：pick_times=1 → 只描述第一个容器。传入：color_names, pick_times=1。"""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["red", "blue", "green"]),
        pick_times=1,
    )
    result = _call("VideoUnmaskSwap", s)
    assert "red cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container" not in g


def test_videounmaskswap_pick_two():
    """情况：pick_times=2 → 描述捡两个容器。传入：color_names, pick_times=2。"""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["red", "blue", "green"]),
        pick_times=2,
    )
    result = _call("VideoUnmaskSwap", s)
    assert "red cube" in result[0]
    assert "another container hiding the blue cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container hiding the blue cube" in g


# ── ButtonUnmask: 2 branches ──
# 测试情况：与 VideoUnmask 类似，按 configs[difficulty]['pick'] 为 1 或 >1 分两种句式（先按按钮再捡容器）。
# 传入：env.unwrapped 的 color_names, configs；self.difficulty。

def test_buttonunmask_pick_one():
    """情况：pick=1 → 先按按钮再捡一个容器。传入：color_names, configs[easy][pick]=1, difficulty=easy。"""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["green", "red", "blue"],
            configs={"easy": {"pick": 1}},
        ),
        difficulty="easy",
    )
    result = _call("ButtonUnmask", s)
    assert "press the button" in result[0]
    assert "green cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press the button" in g
    assert "green cube" in g
    assert "another container" not in g


def test_buttonunmask_pick_two():
    """情况：pick>1 → 先按按钮再捡两个容器。传入：configs[hard][pick]=2, difficulty=hard。"""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["green", "red", "blue"],
            configs={"hard": {"pick": 2}},
        ),
        difficulty="hard",
    )
    result = _call("ButtonUnmask", s)
    assert "press the button" in result[0]
    assert "another container hiding the red cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press the button" in g
    assert "another container hiding the red cube" in g


# ── ButtonUnmaskSwap: 2 branches ──
# 测试情况：self.pick_times 为 1 或 2，句式为先按两个按钮再捡容器（一个或两个）。
# 传入：env.unwrapped 的 color_names；self.pick_times。

def test_buttonunmaskswap_pick_one():
    """情况：pick_times=1 → 先按两按钮再捡一个容器。传入：color_names, pick_times=1。"""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["blue", "green", "red"]),
        pick_times=1,
    )
    result = _call("ButtonUnmaskSwap", s)
    assert "press both buttons" in result[0]
    assert "blue cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press both buttons" in g
    assert "blue cube" in g
    assert "another container" not in g


def test_buttonunmaskswap_pick_two():
    """情况：pick_times=2 → 先按两按钮再捡两个容器。传入：color_names, pick_times=2。"""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["blue", "green", "red"]),
        pick_times=2,
    )
    result = _call("ButtonUnmaskSwap", s)
    assert "press both buttons" in result[0]
    assert "another container hiding the green cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press both buttons" in g
    assert "another container hiding the green cube" in g


# ── VideoPlaceButton: 1 branch ──
# 测试情况：唯一句式「把某色方块放到按钮被按后对应的目标上」。
# 传入：self.target_color_name, self.target_target_language（如 "after" → "right after the button was pressed"）。

def test_videoplacebutton():
    """情况：按视频放方块到「按钮按下后」的目标。传入：target_color_name=red, target_target_language=after。"""
    s = _make_self(
        target_color_name="red",
        target_target_language="after",
    )
    result = _call("VideoPlaceButton", s)
    assert "red cube" in result[0]
    assert "right after the button was pressed" in result[0]
    assert "red cube" in result[1]
    assert "where it was placed immediately after the button was pressed" in result[1]


# ── VideoPlaceOrder: 1 branch ──
# 测试情况：唯一句式「把某色方块放到第 N 次放置的目标上」（N 用序数词，如 third）。
# 传入：self.target_color_name, self.which_in_subset（序数，如 3 → "third"）。

def test_videoplaceorder():
    """情况：按顺序放到第 N 个目标。传入：target_color_name=blue, which_in_subset=3。"""
    s = _make_self(
        target_color_name="blue",
        which_in_subset=3,
    )
    result = _call("VideoPlaceOrder", s)
    assert "blue cube" in result[0]
    assert "third target" in result[0]
    assert "blue cube" in result[1]
    assert "third target" in result[1]
    assert "where it was placed" in result[1]


# ── PickHighlight: 1 branch ──
# 测试情况：唯一句式「先按按钮，再捡起桌上被高亮的所有方块」；无额外传入参数。

def test_pickhighlight():
    """情况：按按钮后捡所有高亮方块。传入：无（仅需 env）。"""
    s = _make_self()
    result = _call("PickHighlight", s)
    assert "press the button" in result[0]
    assert "highlighteted" in result[0]
    assert "highlighted cubes" in result[1]
    assert "press the button again to stop" in result[1]


# ── VideoRepick: 2 branches ──
# 测试情况：self.num_repeats 为 1 或 >1，1 次为「再捡同一块再放下」，多次为「重复捡放 N 次」。
# 传入：self.num_repeats。

def test_videorepick_once():
    """情况：只再捡放 1 次。传入：num_repeats=1。"""
    s = _make_self(num_repeats=1)
    result = _call("VideoRepick", s)
    assert "pick up the same block" in result[0]
    assert "repeatedly" not in result[0]
    assert "pick up the same cube" in result[1]
    assert "repeatedly" not in result[1]


def test_videorepick_multiple():
    """情况：重复捡放多次。传入：num_repeats=4。"""
    s = _make_self(num_repeats=4)
    result = _call("VideoRepick", s)
    assert "repeatedly pick up and put down" in result[0]
    assert "four times" in result[0]
    assert "same cube" in result[1]
    assert "four times" in result[1]


# ── StopCube: 1 branch ──
# 测试情况：唯一句式「在方块第 N 次到达目标时按按钮停下」，N 为序数词。
# 传入：env.unwrapped 的 stop_time（如 2 → "second time"）。

def test_stopcube():
    """情况：第 N 次到达时按按钮停下。传入：stop_time=2。"""
    s = _make_self(unwrapped_attrs=dict(stop_time=2))
    result = _call("StopCube", s)
    assert "second time" in result[0]
    assert "second visit" in result[1]


# ── InsertPeg: 1 branch ──
# 测试情况：唯一句式「抓住同一端、同一根 peg 插入同一侧」；无额外传入。

def test_insertpeg():
    """情况：同一 peg 同一端插入同一侧。传入：无。"""
    s = _make_self()
    result = _call("InsertPeg", s)
    assert "grasp the same end" in result[0]
    assert "grasp the same peg at the same end" in result[1]
    assert "as in the video" in result[1]


# ── MoveCube: 1 branch ──
# 测试情况：唯一句式「按之前的方式把方块移到目标」；无额外传入。

def test_movecube():
    """情况：按之前方式移动方块到目标。传入：无。"""
    s = _make_self()
    result = _call("MoveCube", s)
    assert "move the cube to the target" in result[0]
    assert "shown in the video" in result[1]


# ── PatternLock: 1 branch ──
# 测试情况：唯一句式「用杆重画相同图案」；无额外传入。

def test_patternlock():
    """情况：用杆重画相同图案。传入：无。"""
    s = _make_self()
    result = _call("PatternLock", s)
    assert "retrace the same pattern" in result[0]
    assert "retrace the same pattern shown in the video" in result[1]


# ── RouteStick: 1 branch ──
# 测试情况：唯一句式「用杆沿桌上棍子走相同路径」；无额外传入。

def test_routestick():
    """情况：用杆沿相同路径绕开棍子。传入：无。"""
    s = _make_self()
    result = _call("RouteStick", s)
    assert "navigate around the sticks" in result[0]
    assert "following the same path shown in the video" in result[1]
