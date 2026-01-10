"""
UI布局模块
定义Gradio界面组件、CSS和JS

================================================================================
模块架构说明
================================================================================

本模块是整个应用的视图层（View Layer），负责：
1. 定义所有 Gradio UI 组件的布局和结构
2. 定义 CSS 样式，控制界面外观和动画效果
3. 定义 JavaScript 代码，实现前端交互逻辑和验证
4. 绑定 UI 事件到后端回调函数（gradio_callbacks.py）

模块依赖关系：
- gradio_callbacks.py: 提供所有事件回调函数（控制层）
- user_manager.py: 提供用户管理和任务分配功能
- config.py: 提供配置常量（字体大小、视图高度、缩放比例等）
- note_content.py: 提供任务提示内容

================================================================================
界面流程说明
================================================================================

用户界面采用多阶段显示模式，通过控制组件的 visible 属性实现界面切换：

1. 【初始阶段】loading_group 显示
   └─> 应用启动时显示加载提示

2. 【登录阶段】login_group 显示
   └─> 用户选择用户名并点击 Login 按钮

3. 【主界面阶段】main_interface 显示
   ├─> demo_video_group 和 combined_view_group 同时显示（左右并排）
   └─> operation_zone_group 显示（执行任务）
       └─> 用户选择动作、点击关键点、执行任务

================================================================================
组件交互关系说明
================================================================================

【状态管理】
- uid_state: 用户会话唯一标识符，在所有回调函数间传递，用于标识当前会话
- username_state: 用户名状态，用于记录当前登录用户

【数据流】
1. 用户操作 → UI组件事件 → gradio_callbacks.py 回调函数
2. 回调函数处理业务逻辑 → 更新状态 → 返回更新后的组件值
3. Gradio 自动更新 UI 组件显示

【组件可见性控制】
- 通过设置 Group 组件的 visible 属性控制界面阶段切换
- demo_video_group 和 combined_view_group 同时显示（左右并排）
- operation_zone_group 在登录成功后立即显示

【实时数据流】
- combined_display: HTML 组件，通过 MJPEG 流实时显示机器人执行画面
- img_display: 图像组件，显示当前观察图像，支持点击选择关键点
- log_output: 文本组件，实时显示系统日志和任务进度

================================================================================
界面组件详细说明
================================================================================

【主要界面组】
1. loading_group - 加载界面（初始显示）
   - 用途：应用启动时显示加载提示信息
   - 可见性：初始为 True，登录后隐藏

2. login_group - 登录界面
   - 用途：用户选择用户名并登录
   - 可见性：初始为 False，加载完成后显示

3. main_interface - 主界面（执行界面）
   - 用途：任务执行的主要工作区域
   - 可见性：初始为 False，登录成功后显示
   - 布局：demo_video_group 和 combined_view_group 同时并排显示，operation_zone_group 在下方显示

【登录界面 (login_group)】
- username_input - 用户名下拉选择框
  - 数据源：从 user_manager.user_tasks 获取可用用户名列表
  - 功能：用户选择自己的用户名
- login_btn - 登录按钮
  - 功能：执行登录和任务加载
  - 事件：点击后调用 login_and_load_task() 函数
- login_msg - 登录消息显示
  - 功能：显示登录成功/失败消息

【主界面 (main_interface) - 顶部容器：参考区域 (Reference Zone)】
├─ 左侧列 (30% 宽度，scale=3)：
│  ├─ Progress Tracker 组
│  │  ├─ task_info_box - 当前任务信息框
│  │  │  - 显示：当前任务编号、环境ID等信息
│  │  │  - 更新时机：任务加载时、执行步骤后
│  │  └─ progress_info_box - 进度信息框
│  │     - 显示：任务完成进度（已完成/总数）
│  │     - 更新时机：执行步骤后
│  │
│  ├─ Task Goal 组
│  │  └─ goal_box - 任务目标/指令框
│  │     - 显示：当前任务的详细指令和目标
│  │     - 更新时机：任务加载时
│  │
│  └─ System Log 组
│     └─ log_output - 系统日志输出
│        - 显示：系统运行日志、错误信息、执行结果
│        - 样式：紧凑日志样式（.compact-log），等宽字体
│        - 更新时机：执行步骤时、系统事件发生时
│
└─ 右侧列 (70% 宽度，scale=7)：
   ├─ demo_video_group - 演示视频组（始终显示）
   │  ├─ video_display - 演示视频播放器
   │  │  - 功能：播放任务演示视频
   │  │  - 控制：禁用用户控制，只能通过按钮播放
   │  │  - 限制：如果 RESTRICT_VIDEO_PLAYBACK=True，禁用所有鼠标交互
   │  └─ play_video_btn - 播放演示视频按钮
   │     - 功能：触发视频播放（唯一播放方式）
   │     - 事件：调用 play_demo_video() 函数
   │
   └─ combined_view_group - 组合视图组（始终显示）
      └─ combined_display - 执行实时流显示（HTML组件）
         - 功能：通过 MJPEG 流实时显示机器人执行画面
         - 数据源：/video_feed/{uid} 端点
         - 更新：实时流式传输，无需手动刷新

【主界面 (main_interface) - 底部容器：操作区域 (Operation Zone)】
├─ 左侧列：动作选择 (Action Selection, scale=ACTION_SCALE)
│  └─ options_radio - 动作单选按钮组
│     - 功能：显示可选的动作列表，用户选择要执行的动作
│     - 数据源：从任务配置中获取可用动作列表
│     - 样式：每个选项占满一行（通过 CSS 实现）
│     - 事件：选择改变时调用 on_option_select() 函数
│
├─ 中间列：关键点选择 (Keypoint Selection, scale=LIVE_OBSERVATION_SCALE)
│  └─ img_display - 实时观察图像显示（可点击）
│     - 功能：显示当前观察图像，用户点击选择关键点坐标
│     - 交互：支持点击选择（select 事件）
│     - 事件：点击后调用 on_map_click() 函数
│     - 高亮：当需要选择坐标时，显示蓝色闪烁边框提示
│
└─ 右侧列：控制面板 (Control Panel, scale=CONTROL_SCALE)
   ├─ coords_group - 坐标组（条件显示）
   │  ├─ 可见性：根据选择的动作类型动态显示/隐藏
   │  │  - 需要坐标的动作：显示
   │  │  - 不需要坐标的动作：隐藏
   │  ├─ 高亮：当需要选择坐标且未选择时，显示蓝色闪烁边框
   │  └─ coords_box - 坐标文本框
   │     - 显示：用户点击图像后选择的坐标值
   │     - 初始值："please click the keypoint selection image"
   │     - 验证：执行前检查是否已选择坐标（通过 JavaScript）
   │
   ├─ exec_btn - 执行按钮
   │  - 功能：执行用户选择的动作和坐标
   │  - 样式：红色停止按钮样式（variant="stop"）
   │  - 事件：点击后调用 execute_step() 函数
   │  - 验证：执行前通过 JavaScript 检查坐标是否已选择
   │
   ├─ next_task_btn - 下一个任务按钮
   │  - 功能：加载下一个任务
   │  - 初始状态：禁用（任务完成后启用）
   │  - 事件：点击后调用 load_next_task_wrapper() 函数
   │  - 特殊处理：对于 user_test 用户，会跳转回环境选择界面
   │
【全屏加载遮罩层 (loading_overlay)】
- 组件类型：HTML 组件
- 功能：在加载任务时显示全屏半透明遮罩层，防止用户操作
- 显示方式：通过更新组件内容为包含 .loading-overlay 的 HTML 字符串
- 隐藏方式：通过更新组件内容为空字符串 ""
- 使用场景：
  - 用户点击 Login 按钮时
  - 用户点击 Next Task 按钮时

================================================================================
状态变量说明
================================================================================

- uid_state (gr.State)
  - 类型：字符串或 None
  - 用途：用户会话唯一标识符，在所有回调函数间传递
  - 生成时机：用户登录时由 user_manager 生成
  - 使用场景：标识当前会话，用于获取会话数据、视频流等

- username_state (gr.State)
  - 类型：字符串
  - 用途：记录当前登录的用户名
  - 更新时机：用户登录成功后
  - 使用场景：在回调函数中获取当前用户信息

================================================================================
CSS样式类说明
================================================================================

- .ref-zone - 参考区域样式
  - 用途：为顶部参考区域添加底部边框，与操作区域分隔
  - 应用：main_interface 的顶部 Row 容器

- .compact-log - 紧凑日志样式
  - 用途：设置日志文本框为紧凑显示，限制最大高度
  - 应用：log_output 组件
  - 样式：等宽字体、最大高度 120px

- .coords-group-highlight - 坐标组高亮样式
  - 用途：当需要选择坐标时，为坐标组添加蓝色闪烁边框
  - 应用：coords_group 组件（通过 JavaScript 动态添加类）
  - 效果：蓝色边框、圆角、闪烁动画

- .live-obs-highlight - 实时观察高亮样式
  - 用途：当需要选择坐标时，为实时观察图像添加蓝色闪烁边框
  - 应用：img_display 组件（通过 JavaScript 动态添加类）
  - 效果：蓝色边框、圆角、闪烁动画

- .btn-counting / .btn-persistence / .btn-reference / .btn-behavior
  - 用途：为不同类别的环境按钮定义颜色分类
  - 应用：环境选择按钮（通过 elem_classes 参数）
  - 效果：不同颜色背景和文字，便于视觉区分

================================================================================
JavaScript功能说明
================================================================================

1. 视频播放控制
   - 功能：确保演示视频只能通过按钮播放，禁用自动播放和用户控制
   - 实现：setupVideoAutoplay() 函数设置视频属性，禁用所有交互
   - 监听：initPlayVideoButtonListener() 监听播放按钮点击事件

2. 坐标选择检查
   - 功能：在执行动作前检查是否已选择坐标
   - 实现：checkCoordsBeforeExecute() 检查坐标框值
   - 监听：initExecuteButtonListener() 为 EXECUTE 按钮添加检查

3. LeaseLost 错误处理
   - 功能：检测用户在其他地方登录导致的租约丢失错误
   - 实现：initLeaseLostHandler() 监听多种错误来源
   - 处理：显示提示信息，建议用户刷新页面

4. 高亮动画
   - 功能：当需要选择坐标时，为坐标组和实时观察图像添加闪烁提示
   - 实现：applyCoordsGroupHighlight() 定期检查并应用高亮样式
   - 效果：蓝色边框闪烁动画，引导用户操作

================================================================================
事件绑定流程说明
================================================================================

【事件链模式】
大多数重要操作采用事件链模式（.then()），确保操作顺序执行：
1. 第一步：显示加载遮罩层（show_loading_info）
2. 第二步：执行实际操作（如登录、加载任务等）
3. 第三步：隐藏遮罩层（返回空字符串到 loading_overlay）

【主要事件绑定】
1. login_btn.click() - 登录按钮
   - 链式调用：显示遮罩层 → 登录和加载任务 → 更新界面

2. next_task_btn.click() - 下一个任务按钮
   - 链式调用：显示遮罩层 → 加载下一个任务 → 更新界面

4. play_video_btn.click() - 播放视频按钮
   - 功能：启用视频播放

6. img_display.select() - 图像点击事件
   - 功能：处理用户点击图像选择关键点坐标

7. options_radio.change() - 动作选择改变事件
   - 功能：根据选择的动作类型显示/隐藏坐标组

8. exec_btn.click() - 执行按钮
   - 功能：执行用户选择的动作和坐标，更新任务进度

9. demo.load() - 应用加载事件
   - 功能：应用启动时自动初始化，显示登录界面

================================================================================
"""
import gradio as gr
from user_manager import user_manager
from config import RESTRICT_VIDEO_PLAYBACK, REFERENCE_VIEW_HEIGHT, LIVE_OBSERVATION_SCALE, ACTION_SCALE, CONTROL_SCALE, ENV_IDS, FONT_SIZE, TEXT_INFO_SCALE, COMBINED_VIEW_SCALE, DEMO_VIDEO_SCALE, REFERENCE_ZONE_HEIGHT, OPERATION_ZONE_HEIGHT, DEMO_VIDEO_HEIGHT
from note_content import get_task_hint
from gradio_callbacks import (
    login_and_load_task,
    load_next_task_wrapper,
    on_map_click,
    on_option_select,
    execute_step,
    init_app,
    play_demo_video,
    show_loading_info,  # 【新增导入】显示加载环境提示信息的函数
    on_video_end # Added for video end event handling
)

SYNC_JS = """
/**
 * ================================================================================
 * 同步 JavaScript 代码
 * ================================================================================
 * 
 * 本 JavaScript 代码在页面加载时执行，实现前端交互逻辑和验证功能。
 * 主要功能包括：
 * 1. 视频播放控制：确保演示视频只能通过按钮播放，禁用自动播放和用户控制
 * 2. 坐标选择验证：在执行动作前检查是否已选择坐标
 * 3. LeaseLost 错误处理：检测并处理用户在其他地方登录导致的租约丢失错误
 * 4. 高亮动画提示：当需要选择坐标时，为相关组件添加闪烁提示
 * 
 * 代码采用立即执行函数（IIFE）模式，避免污染全局作用域。
 * ================================================================================
 */
(function() {
    // ============================================================================
    // Execute 按钮条件控制状态变量
    // ============================================================================
    // 目的：跟踪视频播放状态和播放按钮点击状态，用于控制 execute 按钮的启用/禁用
    let isVideoPlaying = false;      // 视频是否正在播放
    let isPlayButtonClicked = false;  // 播放按钮是否已被点击
    
    // ============================================================================
    // 视频播放控制功能
    // ============================================================================
    // 策略：不再自动播放视频，只有点击按钮才播放
    // 目的：确保用户主动观看演示视频，而不是被动播放
    
    /**
     * 设置视频自动播放属性
     * 
     * 功能说明：
     * - 配置视频元素的所有播放相关属性，确保视频不会自动播放
     * - 禁用所有用户控制（鼠标点击、键盘控制、右键菜单等）
     * - 使用 MutationObserver 监听 muted 属性变化，确保始终静音
     * 
     * @param {HTMLVideoElement} v - 视频元素对象
     */
    function setupVideoAutoplay(v) {
        // 确保视频始终静音（muted 属性）
        // 这是浏览器自动播放策略的要求：只有静音的视频才能自动播放
        v.muted = true;
        v.setAttribute('muted', 'true');
        
        // 禁用自动播放，等待用户通过按钮主动触发播放
        v.autoplay = false;
        v.setAttribute('autoplay', 'false');
        
        // 启用循环播放，确保视频播放完成后自动重新开始
        v.loop = true;
        v.setAttribute('loop', 'true');
        
        // 禁用视频控制栏（播放/暂停按钮、进度条等）
        // 用户只能通过我们提供的"播放视频"按钮来控制播放
        // [DEBUG] 暂时启用 controls 以调试视频加载问题
        v.controls = true;
        v.setAttribute('controls', 'true');
        
        // 启用内联播放（在移动设备上重要）
        // 防止视频在移动设备上全屏播放
        v.playsInline = true;
        v.setAttribute('playsinline', 'true');
        
        // 禁用所有鼠标交互（点击、悬停等）
        // 这是最严格的限制，确保用户无法直接操作视频元素
        // [DEBUG] 暂时启用交互
        v.style.pointerEvents = 'auto';
        
        // 阻止用户通过键盘控制视频（空格键暂停/播放、方向键快进/快退等）
        // 使用捕获阶段（true）确保在事件到达视频元素之前就被拦截
        v.addEventListener('keydown', (e) => {
            e.preventDefault();      // 阻止默认行为
            e.stopPropagation();     // 阻止事件冒泡
        }, true);
        
        // 阻止右键菜单（防止用户通过右键菜单控制视频）
        v.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopPropagation();
        }, true);
        
        // 使用 MutationObserver 监听 muted 属性的变化
        // 防止其他代码或浏览器行为修改 muted 属性
        // 如果检测到 muted 被设置为 false，立即恢复为 true
        if (!v.dataset.mutedObserverAttached) {
            const mutedObserver = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    // 只处理属性变化类型的突变
                    if (mutation.type === 'attributes' && mutation.attributeName === 'muted') {
                        // 如果 muted 属性被设置为 false，立即恢复为 true
                        if (!v.muted) {
                            v.muted = true;
                            v.setAttribute('muted', 'true');
                        }
                    }
                });
            });
            
            // 开始观察视频元素的属性变化
            mutedObserver.observe(v, {
                attributes: true,              // 观察属性变化
                attributeFilter: ['muted']     // 只观察 muted 属性
            });
            
            // 标记已附加观察器，避免重复附加
            v.dataset.mutedObserverAttached = 'true';
        }
    }
    
    /**
     * 确保演示视频的自动播放设置
     * 
     * 功能说明：
     * - 查找页面中的演示视频元素
     * - 为每个视频元素应用自动播放限制设置
     * - 确保视频始终静音
     * 
     * 注意：此函数在当前实现中未被直接调用，但保留以备将来使用
     */
    function ensureDemoVideoAutoplay() {
        // 查找演示视频容器（通过 ID 选择器）
        const videoWrapper = document.getElementById('demo_video');
        if (!videoWrapper) return;  // 如果容器不存在，直接返回
        
        // 查找容器内的所有 video 元素
        const vids = videoWrapper.querySelectorAll('video');
        if (vids.length === 0) return;  // 如果没有视频元素，直接返回
        
        // 为每个视频元素应用设置
        vids.forEach((v) => {
            if (!v.dataset.autoplaySetup) {
                // 首次设置：应用完整的自动播放限制
                setupVideoAutoplay(v);
                v.dataset.autoplaySetup = 'true';  // 标记已设置
            } else {
                // 已设置过：只确保 muted 属性正确
                if (!v.muted) {
                    v.muted = true;
                    v.setAttribute('muted', 'true');
                }
            }
        });
    }

    // ============================================================================
    // 坐标选择验证功能
    // ============================================================================
    // 目的：在执行动作前检查用户是否已选择坐标（对于需要坐标的动作）
    
    /**
     * 查找坐标输入框元素
     * 
     * 功能说明：
     * - 尝试多种选择器策略查找坐标输入框
     * - 通过检查输入框的值来判断是否为坐标输入框
     * - 坐标输入框的初始值为 "please click the keypoint selection image"
     * 
     * @returns {HTMLTextAreaElement|null} 坐标输入框元素，如果未找到则返回 null
     */
    function findCoordsBox() {
        // 尝试多种选择器策略，因为 Gradio 可能使用不同的 DOM 结构
        const selectors = [
            '#coords_box textarea',              // 精确 ID 选择器
            '[id*="coords_box"] textarea',       // 包含 "coords_box" 的 ID 选择器
            'textarea[data-testid*="coords"]',   // 通过 data-testid 属性选择
            'textarea'                           // 通用 textarea 选择器（最后尝试）
        ];
        
        // 按顺序尝试每个选择器
        for (const selector of selectors) {
            const elements = document.querySelectorAll(selector);
            for (const el of elements) {
                const value = el.value || '';
                // 如果输入框的值是初始提示文本，说明这就是坐标输入框
                if (value.trim() === 'please click the keypoint selection image') {
                    return el;
                }
            }
        }
        return null;  // 未找到坐标输入框
    }
    
    /**
     * 在执行动作前检查坐标是否已选择
     * 
     * 功能说明：
     * - 查找坐标输入框
     * - 检查输入框的值是否为初始提示文本
     * - 如果是初始提示文本，说明用户尚未选择坐标，阻止执行并显示提示
     * 
     * @returns {boolean} 如果坐标已选择返回 true，否则返回 false
     */
    function checkCoordsBeforeExecute() {
        const coordsBox = findCoordsBox();
        if (coordsBox) {
            const coordsValue = coordsBox.value || '';
            // 如果值是初始提示文本，说明需要坐标但用户没有点击图像选择
            if (coordsValue.trim() === 'please click the keypoint selection image') {
                // 显示提示信息，要求用户先选择坐标
                alert('please click the keypoint selection image before execute!');
                return false;  // 返回 false 表示验证失败，阻止执行
            }
        }
        return true;  // 坐标已选择或不需要坐标，允许执行
    }
    
    /**
     * 为 EXECUTE 按钮添加坐标检查监听器
     * 
     * 功能说明：
     * - 为执行按钮添加点击事件监听器
     * - 在点击时先检查 execute 按钮条件（视频状态和播放按钮点击状态）
     * - 然后检查坐标是否已选择
     * - 如果任一检查失败，阻止按钮的默认行为（阻止执行）
     * 
     * @param {HTMLButtonElement} btn - 执行按钮元素
     */
    function attachCoordsCheckToButton(btn) {
        // 使用 dataset 标记避免重复附加监听器
        if (!btn.dataset.coordsCheckAttached) {
            // 在捕获阶段（true）添加监听器，确保在其他监听器之前执行
            btn.addEventListener('click', function(e) {
                // 首先检查 execute 按钮条件（视频状态和播放按钮点击状态）
                const hasVideo = hasDemoVideo();
                if (hasVideo && (isVideoPlaying || !isPlayButtonClicked)) {
                    // 如果有视频但条件不满足，阻止执行
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    alert('Please wait for the demonstration video to finish playing and make sure you have clicked "Start Demonstration Video" button before executing.');
                    return false;
                }
                
                // 执行坐标检查
                if (!checkCoordsBeforeExecute()) {
                    // 如果检查失败，阻止所有后续操作
                    e.preventDefault();              // 阻止默认行为
                    e.stopPropagation();             // 阻止事件冒泡
                    e.stopImmediatePropagation();   // 阻止同一元素上的其他监听器执行
                    return false;
                }
            }, true);
            // 标记已附加监听器
            btn.dataset.coordsCheckAttached = 'true';
        }
    }
    
    /**
     * 播放演示视频的函数（只在点击按钮时调用）
     * 
     * 功能说明：
     * - 查找页面中的演示视频元素
     * - 确保视频设置正确（静音、循环）
     * - 根据视频加载状态选择播放策略
     * - 处理视频播放的 Promise（现代浏览器的 play() 方法返回 Promise）
     * 
     * 播放策略：
     * - 如果视频已加载（readyState >= 2），直接播放
     * - 如果视频未加载，等待加载完成后再播放
     */
    function playDemoVideo() {
        const videoWrapper = document.getElementById('demo_video');
        if (videoWrapper) {
            const vids = videoWrapper.querySelectorAll('video');
            vids.forEach(v => {
                // 确保视频设置正确（这些设置可能在页面更新时被重置）
                v.muted = true;
                v.setAttribute('muted', 'true');
                v.loop = true;
                v.setAttribute('loop', 'true');
                
                // 尝试播放视频
                // readyState 值说明：
                // 0 = HAVE_NOTHING（没有信息）
                // 1 = HAVE_METADATA（已获取元数据）
                // 2 = HAVE_CURRENT_DATA（已获取当前帧数据，可以显示第一帧）
                // 3 = HAVE_FUTURE_DATA（已获取未来数据，可以播放但可能卡顿）
                // 4 = HAVE_ENOUGH_DATA（已获取足够数据，可以流畅播放）
                if (v.readyState >= 2) {
                    // 视频已加载到可以播放的状态，直接播放
                    const playPromise = v.play();
                    // 现代浏览器的 play() 方法返回 Promise
                    // 如果播放失败（例如用户未交互），Promise 会被拒绝
                    // 这里静默处理错误，避免控制台报错
                    if (playPromise && playPromise.catch) {
                        playPromise.catch(() => {});
                    }
                } else {
                    // 视频还未准备好，等待加载完成后再播放
                    // 监听 loadeddata 事件：视频的第一帧数据已加载
                    v.addEventListener('loadeddata', function() {
                        const playPromise = v.play();
                        if (playPromise && playPromise.catch) {
                            playPromise.catch(() => {});
                        }
                    }, { once: true });  // once: true 表示只执行一次
                    
                    // 也监听 canplay 事件作为备选（视频可以开始播放）
                    v.addEventListener('canplay', function() {
                        const playPromise = v.play();
                        if (playPromise && playPromise.catch) {
                            playPromise.catch(() => {});
                        }
                    }, { once: true });
                }
            });
        }
    }
    
    /**
     * 初始化视频播放状态监听器
     * 
     * 功能说明：
     * - 查找演示视频元素并监听播放状态变化
     * - 监听 play、pause、ended 事件，更新 isVideoPlaying 状态
     * - 当视频状态变化时，更新 execute 按钮状态
     * - 当视频元素被移除时，重置状态
     * 
     * 实现策略：
     * - 使用 MutationObserver 等待视频元素加载
     * - 为每个视频元素添加事件监听器
     * - 处理视频循环播放的情况（ended 后可能自动重新播放）
     */
    function initVideoPlayStateListener() {
        let lastVideoCount = 0;
        let lastVideoSrc = ""; // 记录上一次的视频源
        
        /**
         * 尝试为视频元素附加播放状态监听器
         * 内部函数，在 MutationObserver 回调中调用
         */
        function attachToVideoElements() {
            const videoWrapper = document.getElementById('demo_video');
            if (!videoWrapper) {
                // 如果容器不存在，检查是否需要重置状态
                if (lastVideoCount > 0) {
                    // 之前有视频，现在没有了，重置状态
                    resetExecuteButtonState();
                    // 移除演示视频组高亮（容器不存在，可能任务切换）
                    removeDemoVideoGroupHighlight();
                    lastVideoCount = 0;
                    lastVideoSrc = "";
                }
                return;
            }
            
            const vids = videoWrapper.querySelectorAll('video');
            const currentVideoCount = vids.length;
            const currentVideoSrc = currentVideoCount > 0 ? vids[0].src : "";
            
            // 如果视频数量变化（从有到无），重置状态
            if (lastVideoCount > 0 && currentVideoCount === 0) {
                resetExecuteButtonState();
                // 移除演示视频组高亮（任务切换，新任务没有视频）
                removeDemoVideoGroupHighlight();
            }
            
            // 如果视频数量变化（从无到有）或者视频源发生变化（任务切换），应用高亮
            const isNewVideo = (lastVideoCount === 0 && currentVideoCount > 0);
            const isSrcChanged = (lastVideoSrc !== "" && currentVideoSrc !== "" && lastVideoSrc !== currentVideoSrc);
            
            if (isNewVideo || isSrcChanged) {
                applyDemoVideoGroupHighlight();
                // 重置播放状态，因为是新视频
                resetExecuteButtonState();
            }
            
            lastVideoCount = currentVideoCount;
            lastVideoSrc = currentVideoSrc;
            
            if (vids.length === 0) {
                // 如果没有视频元素，移除高亮（如果存在）
                removeDemoVideoGroupHighlight();
                return;
            }
            
            vids.forEach(v => {
                // 检查是否已经附加了监听器
                if (v.dataset.playStateListenerAttached) return;
                
                // 监听播放开始事件
                v.addEventListener('play', function() {
                    isVideoPlaying = true;
                    updateExecuteButtonState();
                }, { once: false });
                
                // 监听暂停事件
                v.addEventListener('pause', function() {
                    isVideoPlaying = false;
                    updateExecuteButtonState();
                }, { once: false });
                
                // 监听播放结束事件
                v.addEventListener('ended', function() {
                    isVideoPlaying = false;
                    updateExecuteButtonState();
                    // 移除演示视频组高亮（视频播放完毕）
                    removeDemoVideoGroupHighlight();
                    // 注意：如果视频设置了 loop=true，ended 后会自动重新播放
                    // 但我们需要等待实际的 play 事件来更新状态
                }, { once: false });
                
                // 检查当前播放状态（处理页面加载时视频已经在播放的情况）
                if (!v.paused && !v.ended) {
                    isVideoPlaying = true;
                } else {
                    isVideoPlaying = false;
                }
                
                // 标记已附加监听器
                v.dataset.playStateListenerAttached = 'true';
            });
            
            // 初始化后立即更新 execute 按钮状态
            updateExecuteButtonState();
        }
        
        // 使用 MutationObserver 等待视频元素加载
        const observer = new MutationObserver(function(mutations) {
            attachToVideoElements();
        });
        
        // 开始观察整个文档的变化
        observer.observe(document.body, {
            childList: true,    // 观察子节点的添加和删除
            subtree: true       // 观察所有后代节点
        });
        
        // 立即执行一次，处理视频已经存在的情况
        setTimeout(attachToVideoElements, 2000);
        
        // 定期检查视频状态（处理视频元素被替换的情况）
        setInterval(function() {
            attachToVideoElements();
        }, 1000);
    }
    
    /**
     * 初始化播放视频按钮监听器
     * 
     * 功能说明：
     * - 查找"播放视频"按钮并为其添加点击事件监听器
     * - 使用 MutationObserver 等待 Gradio 动态加载按钮
     * - 这是唯一触发视频播放的方式（用户点击按钮）
     * 
     * 实现策略：
     * - 由于 Gradio 是动态加载组件，按钮可能在页面加载后才出现
     * - 使用 MutationObserver 监听 DOM 变化，确保按钮出现后立即附加监听器
     * - 同时使用 setTimeout 立即尝试一次（处理按钮已存在的情况）
     */
    function initPlayVideoButtonListener() {
        /**
         * 尝试为播放视频按钮附加监听器
         * 内部函数，在 MutationObserver 回调中调用
         */
        function attachToPlayVideoButton() {
            const playBtn = document.getElementById('play_video_btn');
            if (playBtn && !playBtn.dataset.playVideoAttached) {
                // 按钮存在且尚未附加监听器
                playBtn.addEventListener('click', function(e) {
                    // 检查按钮是否可交互（可能被禁用）
                    if (playBtn.disabled || playBtn.hasAttribute('disabled')) {
                        return;  // 按钮被禁用，不执行任何操作
                    }
                    // 标记播放按钮已被点击
                    isPlayButtonClicked = true;
                    // 按钮可交互，点击后立即播放视频
                    playDemoVideo();
                    // 更新 execute 按钮状态（视频可能立即开始播放）
                    // 注意：实际的视频播放状态会在视频的 play 事件中更新
                    updateExecuteButtonState();
                });
                // 标记已附加监听器，避免重复附加
                playBtn.dataset.playVideoAttached = 'true';
            }
        }
        
        // 使用 MutationObserver 等待 Gradio 加载完成
        // MutationObserver 可以监听 DOM 树的变化，当按钮被添加到页面时触发
        const observer = new MutationObserver(function(mutations) {
            attachToPlayVideoButton();
        });
        
        // 开始观察整个文档的变化
        observer.observe(document.body, {
            childList: true,    // 观察子节点的添加和删除
            subtree: true       // 观察所有后代节点，不仅仅是直接子节点
        });
        
        // 立即执行一次，处理按钮已经存在的情况（页面已加载完成）
        setTimeout(attachToPlayVideoButton, 2000);
    }
    
    /**
     * 检查是否存在演示视频
     * 
     * @returns {boolean} 如果存在演示视频返回 true，否则返回 false
     */
    function hasDemoVideo() {
        const videoWrapper = document.getElementById('demo_video');
        if (!videoWrapper) return false;
        const vids = videoWrapper.querySelectorAll('video');
        return vids.length > 0;
    }
    
    /**
     * 应用演示视频组高亮样式
     * 
     * 功能说明：
     * - 为 demo_video_group 添加 demo-video-group-highlight 类
     * - 使用 CSS 类和内联样式双重保障，确保高亮效果显示
     */
    function applyDemoVideoGroupHighlight() {
        
        // 尝试多种方式查找demo_video_group元素
        let demoVideoGroup = document.getElementById('demo_video_group');
        
        // 如果直接通过ID找不到，尝试通过querySelector查找包含demo_video的组
        if (!demoVideoGroup) {
            const videoWrapper = document.getElementById('demo_video');
            if (videoWrapper) {
                // 查找包含videoWrapper的组元素
                demoVideoGroup = videoWrapper.closest('.gr-group');
            }
        }
        
        // 如果还是找不到，遍历所有组查找包含demo_video的组
        if (!demoVideoGroup) {
            const allGroups = document.querySelectorAll('.gr-group');
            for (let group of allGroups) {
                const videoWrapperInGroup = group.querySelector('#demo_video');
                if (videoWrapperInGroup) {
                    demoVideoGroup = group;
                    break;
                }
            }
        }
        
        
        if (demoVideoGroup) {
            // 确保组有正确的ID
            if (!demoVideoGroup.id || !demoVideoGroup.id.includes('demo_video_group')) {
                demoVideoGroup.id = 'demo_video_group';
            }
            demoVideoGroup.classList.add('demo-video-group-highlight');
            
            // 应用内联样式确保样式生效（类似坐标组的处理方式）
            demoVideoGroup.style.setProperty('border', '3px solid #3b82f6', 'important');
            demoVideoGroup.style.setProperty('border-radius', '8px', 'important');
            demoVideoGroup.style.setProperty('padding', '15px', 'important');
            demoVideoGroup.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
        }
    }
    
    /**
     * 移除演示视频组高亮样式
     * 
     * 功能说明：
     * - 从 demo_video_group 移除 demo-video-group-highlight 类
     * - 清理所有高亮相关的样式
     */
    function removeDemoVideoGroupHighlight() {
        // 尝试多种方式查找demo_video_group元素
        let demoVideoGroup = document.getElementById('demo_video_group');
        
        // 如果直接通过ID找不到，尝试通过querySelector查找包含demo_video的组
        if (!demoVideoGroup) {
            const videoWrapper = document.getElementById('demo_video');
            if (videoWrapper) {
                demoVideoGroup = videoWrapper.closest('.gr-group');
            }
        }
        
        // 如果还是找不到，遍历所有组查找包含demo_video的组
        if (!demoVideoGroup) {
            const allGroups = document.querySelectorAll('.gr-group');
            for (let group of allGroups) {
                const videoWrapperInGroup = group.querySelector('#demo_video');
                if (videoWrapperInGroup) {
                    demoVideoGroup = group;
                    break;
                }
            }
        }
        
        if (demoVideoGroup) {
            demoVideoGroup.classList.remove('demo-video-group-highlight');
            // 移除内联样式
            demoVideoGroup.style.removeProperty('border');
            demoVideoGroup.style.removeProperty('border-radius');
            demoVideoGroup.style.removeProperty('padding');
            demoVideoGroup.style.removeProperty('animation');
        }
    }
    
    /**
     * 更新演示视频组高亮状态
     * 
     * 功能说明：
     * - 根据是否有演示视频来应用或移除高亮
     * - 如果有演示视频，应用高亮；否则移除高亮
     */
    function updateDemoVideoGroupHighlight() {
        const hasVideo = hasDemoVideo();
        if (hasVideo) {
            applyDemoVideoGroupHighlight();
        } else {
            removeDemoVideoGroupHighlight();
        }
    }
    
    /**
     * 重置 Execute 按钮条件控制状态
     * 
     * 功能说明：
     * - 在任务切换或页面重置时调用
     * - 重置 isPlayButtonClicked 为 false
     * - 重置 isVideoPlaying 为 false
     * - 更新 execute 按钮状态
     */
    function resetExecuteButtonState() {
        isPlayButtonClicked = false;
        isVideoPlaying = false;
        updateExecuteButtonState();
    }
    
    /**
     * 创建操作区域遮罩层
     * 
     * 功能说明：
     * - 查找操作区域组（operation_zone_group）
     * - 创建遮罩层 DOM 元素并添加到操作区域组中
     * - 遮罩层用于在 execute 按钮被禁用时覆盖整个操作区域
     * 
     * @returns {HTMLElement|null} 遮罩层元素，如果操作区域不存在则返回 null
     */
    function createOperationZoneOverlay() {
        // 查找操作区域组（通过 ID 选择器）
        const operationZone = document.querySelector('[id*="operation_zone_group"]');
        if (!operationZone) return null;  // 如果操作区域不存在，直接返回
        
        // 检查是否已经存在遮罩层
        let overlay = operationZone.querySelector('.operation-zone-overlay');
        if (overlay) {
            return overlay;  // 如果已存在，直接返回
        }
        
        // 创建遮罩层元素
        overlay = document.createElement('div');
        overlay.className = 'operation-zone-overlay';
        overlay.id = 'operation_zone_overlay';
        
        // 将遮罩层添加到操作区域组中（作为第一个子元素，确保在最上层）
        operationZone.insertBefore(overlay, operationZone.firstChild);
        
        return overlay;
    }
    
    /**
     * 更新操作区域遮罩状态
     * 
     * 功能说明：
     * - 根据 execute 按钮的状态显示或隐藏遮罩
     * - 当 execute 按钮被禁用时显示遮罩，启用时隐藏遮罩
     * - 遮罩状态与 execute 按钮状态完全同步
     */
    function updateOperationZoneOverlay() {
        // 检查是否存在演示视频
        const hasVideo = hasDemoVideo();
        
        // 判断 execute 按钮是否应该被启用
        // 条件：如果没有视频，或者（视频不在播放且播放按钮已被点击）
        const shouldEnableExecute = !hasVideo || (!isVideoPlaying && isPlayButtonClicked);
        
        // 查找操作区域组
        const operationZone = document.querySelector('[id*="operation_zone_group"]');
        if (!operationZone) return;  // 如果操作区域不存在，直接返回
        
        // 查找或创建遮罩层
        let overlay = operationZone.querySelector('.operation-zone-overlay');
        if (!overlay) {
            // 如果遮罩层不存在，尝试创建
            overlay = createOperationZoneOverlay();
            if (!overlay) return;  // 如果创建失败，直接返回
        }
        
        // 根据 execute 按钮状态更新遮罩
        if (shouldEnableExecute) {
            // Execute 按钮启用，隐藏遮罩
            overlay.classList.remove('active');
        } else {
            // Execute 按钮禁用，显示遮罩
            overlay.classList.add('active');
        }
    }
    
    /**
     * 更新 Execute 按钮状态
     * 
     * 功能说明：
     * - 根据视频播放状态和播放按钮点击状态，启用或禁用 execute 按钮
     * - 条件：只有当视频不在播放且播放按钮已被点击时，才启用 execute 按钮
     * - 如果没有演示视频，则始终启用 execute 按钮
     * - 查找页面中所有的 EXECUTE 按钮并更新其状态
     * - 同时更新操作区域遮罩状态
     * 
     * 实现策略：
     * - 搜索所有按钮，通过按钮文本内容识别 EXECUTE 按钮
     * - 根据条件设置按钮的 disabled 属性
     * - 调用 updateOperationZoneOverlay() 更新遮罩状态
     */
    function updateExecuteButtonState() {
        // 检查是否存在演示视频
        const hasVideo = hasDemoVideo();
        
        // 检查 episode 是否已完成（通过检查状态消息）
        let episodeFinished = false;
        try {
            // 查找系统日志输出元素（使用 elem_id="log_output"）
            const logOutput = document.getElementById('log_output');
            if (logOutput) {
                const text = logOutput.textContent || logOutput.innerText || '';
                if (text.includes('episode failed') || text.includes('episode success')) {
                    episodeFinished = true;
                }
            }
        } catch (e) {
            // 如果检查失败，忽略错误
        }
        
        // 查找页面中的所有按钮
        const buttons = document.querySelectorAll('button');
        for (const btn of buttons) {
            // 获取按钮文本内容（支持多种方式）
            const btnText = btn.textContent || btn.innerText || '';
            // 如果按钮文本包含 "EXECUTE"，说明这是执行按钮
            if (btnText.trim().includes('EXECUTE')) {
                // 如果 episode 已完成，禁用按钮
                if (episodeFinished) {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                    btn.style.cursor = 'not-allowed';
                    continue;
                }
                
                // 如果没有演示视频，始终启用 execute 按钮
                // 如果有演示视频，条件：!isVideoPlaying && isPlayButtonClicked
                const shouldEnable = !hasVideo || (!isVideoPlaying && isPlayButtonClicked);
                btn.disabled = !shouldEnable;
                
                // 如果按钮被禁用，添加视觉提示（可选）
                if (!shouldEnable) {
                    btn.style.opacity = '0.5';
                    btn.style.cursor = 'not-allowed';
                } else {
                btn.style.opacity = '1';
                btn.style.cursor = 'pointer';
            }
        }
        
        // 更新操作区域遮罩状态（与 execute 按钮状态同步）
        updateOperationZoneOverlay();
    }
    }
    
    /**
     * 初始化执行按钮监听器
     * 
     * 功能说明：
     * - 查找页面中所有的 EXECUTE 按钮
     * - 为每个按钮添加坐标检查监听器
     * - 使用 MutationObserver 处理动态加载的按钮
     * 
     * 实现策略：
     * - 搜索所有按钮，通过按钮文本内容识别 EXECUTE 按钮
     * - 使用 MutationObserver 确保新添加的按钮也能被处理
     */
    function initExecuteButtonListener() {
        /**
         * 尝试为所有 EXECUTE 按钮附加坐标检查监听器
         * 内部函数，在 MutationObserver 回调中调用
         */
        function attachToExecuteButtons() {
            // 查找页面中的所有按钮
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                // 获取按钮文本内容（支持多种方式）
                const btnText = btn.textContent || btn.innerText || '';
                // 如果按钮文本包含 "EXECUTE"，说明这是执行按钮
                if (btnText.trim().includes('EXECUTE')) {
                    // 为执行按钮附加坐标检查监听器
                    attachCoordsCheckToButton(btn);
                }
            }
            // 更新 execute 按钮状态（确保新添加的按钮状态正确）
            updateExecuteButtonState();
        }
        
        // 使用 MutationObserver 等待 Gradio 加载完成
        const observer = new MutationObserver(function(mutations) {
            attachToExecuteButtons();
        });
        
        // 开始观察整个文档的变化
        observer.observe(document.body, {
            childList: true,    // 观察子节点的添加和删除
            subtree: true       // 观察所有后代节点
        });
        
        // 立即执行一次，处理已经加载的按钮
        setTimeout(attachToExecuteButtons, 2000);
    }
    
    // ============================================================================
    // LeaseLost 错误处理功能
    // ============================================================================
    // 目的：检测并处理用户在其他地方登录导致的租约丢失错误
    
    /**
     * 初始化 LeaseLost 错误处理器
     * 
     * 功能说明：
     * - LeaseLost 错误发生在用户在同一账号的多个标签页/窗口中登录时
     * - 当用户在一个标签页登录后，其他标签页的会话会失效
     * - 本函数通过多种方式检测 LeaseLost 错误，并提示用户刷新页面
     * 
     * 检测策略（三重保障）：
     * 1. 监听全局错误事件（window.error）
     * 2. 监听 DOM 变化，查找 Gradio 显示的错误消息
     * 3. 拦截 fetch 请求，检查 API 响应中的错误
     */
    function initLeaseLostHandler() {
        // ========================================================================
        // 策略 1: 监听全局错误事件
        // ========================================================================
        // 当 JavaScript 抛出未捕获的错误时触发
        window.addEventListener('error', function(e) {
            // 获取错误消息（支持多种错误对象格式）
            const errorMsg = e.message || e.error?.message || '';
            // 检查错误消息中是否包含 LeaseLost 相关关键词
            if (errorMsg.includes('LeaseLost') || errorMsg.includes('lease lost')) {
                e.preventDefault();  // 阻止默认错误处理
                // 显示用户友好的提示信息
                alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                // 可选: 自动刷新页面（当前已注释，让用户手动刷新）
                // window.location.reload();
            }
        });
        
        // ========================================================================
        // 策略 2: 监听 DOM 变化，查找 Gradio 显示的错误消息
        // ========================================================================
        // Gradio 使用 toast 通知显示错误消息，这些消息会被添加到 DOM 中
        // 通过 MutationObserver 监听 DOM 变化，查找包含错误消息的新元素
        const errorObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                // 遍历所有新添加的节点
                mutation.addedNodes.forEach(function(node) {
                    // 只处理元素节点（nodeType === 1）
                    if (node.nodeType === 1) {
                        // 获取节点的文本内容
                        const text = node.textContent || node.innerText || '';
                        // 检查文本中是否包含 LeaseLost 相关关键词
                        if (text.includes('LeaseLost') || text.includes('lease lost') || 
                            text.includes('logged in elsewhere') || text.includes('no longer valid')) {
                            // 使用 setTimeout 延迟显示，确保 DOM 更新完成
                            setTimeout(() => {
                                alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                            }, 100);
                        }
                    }
                });
            });
        });
        
        // 开始观察整个文档的变化
        errorObserver.observe(document.body, {
            childList: true,    // 观察子节点的添加和删除
            subtree: true       // 观察所有后代节点
        });
        
        // ========================================================================
        // 策略 3: 拦截 fetch 请求，检查 API 响应中的错误
        // ========================================================================
        // Gradio 使用 fetch API 与后端通信
        // 拦截所有 fetch 请求，检查响应中是否包含 LeaseLost 错误
        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            // 调用原始的 fetch 函数
            return originalFetch.apply(this, args).then(function(response) {
                // 检查响应是否成功
                if (response.ok) {
                    // 克隆响应（因为响应只能读取一次）
                    return response.clone().json().then(function(data) {
                        // Gradio API 返回的数据结构通常是对象
                        if (data && typeof data === 'object') {
                            // 将数据转换为字符串，检查是否包含错误信息
                            const dataStr = JSON.stringify(data);
                            if (dataStr.includes('LeaseLost') || dataStr.includes('lease lost')) {
                                // 检测到错误，显示提示
                                setTimeout(() => {
                                    alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                                }, 100);
                            }
                        }
                        return response;  // 返回原始响应
                    }).catch(function() {
                        // JSON 解析失败，返回原始响应（不影响正常流程）
                        return response;
                    });
                }
                return response;  // 响应不成功，直接返回
            });
        };
    }
    
    // ============================================================================
    // 高亮动画提示功能
    // ============================================================================
    // 目的：当需要选择坐标时，为坐标组和实时观察图像添加蓝色闪烁边框提示
    
    /**
     * 应用坐标组高亮动画
     * 
     * 功能说明：
     * - 定期检查是否需要显示高亮提示
     * - 当坐标组可见且坐标未选择时，显示蓝色闪烁边框
     * - 同时为实时观察图像（live_obs）添加高亮提示
     * - 当坐标已选择或坐标组隐藏时，移除高亮
     * 
     * 实现策略：
     * - 使用 setInterval 定期检查（每 500ms）
     * - 通过检查坐标输入框的值判断是否需要高亮
     * - 使用 CSS 类和内联样式双重保障，确保高亮效果显示
     */
    function applyCoordsGroupHighlight() {
        /**
         * 移除坐标组的高亮样式
         * 
         * @param {HTMLElement} group - 坐标组元素
         */
        function removeHighlightStyles(group) {
            // 移除所有内联样式
            group.style.removeProperty('border');
            group.style.removeProperty('border-radius');
            group.style.removeProperty('padding');
            group.style.removeProperty('animation');
            // 移除 CSS 类
            group.classList.remove('coords-group-highlight');
            // 如果 ID 是 coords_group，移除它（恢复默认状态）
            if (group.id === 'coords_group') {
                group.removeAttribute('id');
            }
        }
        
        /**
         * 移除实时观察图像的高亮样式
         * 
         * @param {HTMLElement} liveObs - 实时观察图像元素
         */
        function removeLiveObsHighlight(liveObs) {
            if (liveObs) {
                // 移除所有内联样式
                liveObs.style.removeProperty('border');
                liveObs.style.removeProperty('border-radius');
                liveObs.style.removeProperty('animation');
                // 移除 CSS 类
                liveObs.classList.remove('live-obs-highlight');
            }
        }
        
        /**
         * 应用实时观察图像的高亮样式
         * 
         * @param {HTMLElement} liveObs - 实时观察图像元素
         */
        function applyLiveObsHighlight(liveObs) {
            if (liveObs) {
                // 获取计算后的样式，检查是否需要应用样式
                const computedStyle = window.getComputedStyle(liveObs);
                const needsStyle = (
                    computedStyle.borderColor !== 'rgb(59, 130, 246)' ||  // 不是蓝色边框
                    computedStyle.borderWidth !== '3px' ||                 // 不是 3px 宽度
                    computedStyle.animationName === 'none' ||              // 没有动画
                    !computedStyle.animationName.includes('bluePulse')     // 不是蓝色脉冲动画
                );
                
                // 如果需要应用样式
                if (needsStyle) {
                    // 添加 CSS 类（触发 CSS 规则）
                    liveObs.classList.add('live-obs-highlight');
                    // 应用内联样式（确保样式生效，使用 !important）
                    liveObs.style.setProperty('border', '3px solid #3b82f6', 'important');
                    liveObs.style.setProperty('border-radius', '8px', 'important');
                    liveObs.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
                }
            }
        }
        
        /**
         * 检查并应用/移除坐标组高亮
         * 
         * 这是核心函数，定期执行，检查当前状态并应用相应的高亮效果
         */
        function checkForCoordsGroup() {
            // 查找坐标输入框和实时观察图像
            const coordsBox = document.querySelector('[id*="coords_box"]');
            const liveObs = document.querySelector('#live_obs');
            let targetGroup = null;              // 需要高亮的坐标组
            let shouldHighlightLiveObs = false;  // 是否需要高亮实时观察图像
            
            if (coordsBox) {
                // 检查坐标是否已选择
                // 坐标输入框可能是 textarea 元素本身，也可能包含 textarea 子元素
                const coordsTextarea = coordsBox.querySelector('textarea') || coordsBox;
                const coordsValue = (coordsTextarea.value || '').trim();
                // 如果值不是初始提示文本，说明坐标已选择
                const isCoordsSelected = coordsValue !== 'please click the keypoint selection image';
                
                // 查找包含坐标输入框的父组（Gradio 使用 .gr-group 类）
                let parentGroup = coordsBox.parentElement;
                // 向上遍历 DOM 树，找到 .gr-group 元素
                while (parentGroup && !parentGroup.classList.contains('gr-group')) {
                    parentGroup = parentGroup.parentElement;
                }
                
                // 如果执行按钮也在同一个组中，需要找到更具体的父组
                // 因为坐标组和执行按钮可能在不同的组中
                if (parentGroup && parentGroup.querySelector('[id*="exec_btn"]') !== null) {
                    // 查找所有组，找到包含坐标输入框但不包含执行按钮的组
                    const allGroups = document.querySelectorAll('.gr-group');
                    for (let group of allGroups) {
                        if (group.contains(coordsBox) && !group.querySelector('[id*="exec_btn"]')) {
                            parentGroup = group;
                            break;
                        }
                    }
                }
                
                // 如果找到了坐标组且不包含执行按钮
                if (parentGroup && !parentGroup.querySelector('[id*="exec_btn"]')) {
                    const computedStyle = window.getComputedStyle(parentGroup);
                    // 检查组是否可见
                    const isVisible = parentGroup.offsetParent !== null && computedStyle.display !== 'none';
                    // 检查是否已有 CSS 动画
                    const hasCSSAnimation = computedStyle.animationName !== 'none' && computedStyle.animationName.includes('bluePulse');
                    
                    // 不需要高亮的情况：
                    // 1. 坐标组隐藏（不可见）
                    // 2. 坐标已选择（值不是初始提示文本）
                    if (!isVisible || isCoordsSelected) {
                        // 移除高亮样式（如果存在）
                        const hasBluePulseAnimation = parentGroup.style.animation && parentGroup.style.animation.includes('bluePulse');
                        if (hasBluePulseAnimation || hasCSSAnimation) {
                            removeHighlightStyles(parentGroup);
                            // 同时移除 CSS 类
                            parentGroup.classList.remove('coords-group-highlight');
                        }
                        // 移除实时观察图像的高亮
                        removeLiveObsHighlight(liveObs);
                    } else {
                        // 需要高亮：坐标组可见且坐标未选择
                        targetGroup = parentGroup;
                        shouldHighlightLiveObs = true;  // 同时高亮实时观察图像
                        
                        // 确保组有正确的 ID
                        if (!parentGroup.id || !parentGroup.id.includes('coords_group')) {
                            parentGroup.id = 'coords_group';
                        }
                        
                        // 添加 CSS 类（触发 CSS 规则）
                        parentGroup.classList.add('coords-group-highlight');
                        
                        // 检查是否需要应用内联样式
                        const needsStyle = (
                            computedStyle.borderColor === 'rgb(0, 0, 0)' ||      // 黑色边框（默认）
                            computedStyle.borderWidth === '0px' ||              // 无边框
                            computedStyle.borderStyle === 'none' ||             // 无边框样式
                            computedStyle.animationName === 'none' ||            // 无动画
                            !computedStyle.animationName.includes('bluePulse')    // 不是蓝色脉冲动画
                        );
                        
                        // 如果需要，应用内联样式（确保样式生效）
                        if (needsStyle) {
                            parentGroup.style.setProperty('border', '3px solid #3b82f6', 'important');
                            parentGroup.style.setProperty('border-radius', '8px', 'important');
                            parentGroup.style.setProperty('padding', '15px', 'important');
                            parentGroup.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
                        }
                    }
                }
            }
            
            // 根据 shouldHighlightLiveObs 标志应用或移除实时观察图像的高亮
            if (shouldHighlightLiveObs && liveObs) {
                const liveObsComputedStyle = window.getComputedStyle(liveObs);
                const isLiveObsVisible = liveObs.offsetParent !== null && liveObsComputedStyle.display !== 'none';
                if (isLiveObsVisible) {
                    // 实时观察图像可见，应用高亮
                    applyLiveObsHighlight(liveObs);
                } else {
                    // 实时观察图像不可见，移除高亮
                    removeLiveObsHighlight(liveObs);
                }
            } else {
                // 不需要高亮实时观察图像，移除高亮
                removeLiveObsHighlight(liveObs);
            }
            
            // 清理其他不应该有高亮的组
            // 遍历所有组，移除不应该有高亮的组的高亮样式
            const allGroups = document.querySelectorAll('.gr-group');
            allGroups.forEach(group => {
                if (group === targetGroup) return;  // 跳过目标组（应该高亮的组）
                
                // 检查组的特征
                const hasExecBtn = group.querySelector('[id*="exec_btn"]') !== null;      // 包含执行按钮
                const hasCoordsBox = group.querySelector('[id*="coords_box"]') !== null;  // 包含坐标输入框
                const hasBluePulseAnimation = group.style.animation && group.style.animation.includes('bluePulse');
                const hasHighlightClass = group.classList.contains('coords-group-highlight');
                const computedStyle = window.getComputedStyle(group);
                const hasCSSAnimation = computedStyle.animationName !== 'none' && computedStyle.animationName.includes('bluePulse');
                
                // 如果组包含执行按钮或不包含坐标输入框，且当前有高亮样式，则移除
                if ((hasExecBtn || !hasCoordsBox) && (hasBluePulseAnimation || hasHighlightClass || hasCSSAnimation)) {
                    removeHighlightStyles(group);
                }
            });
        }
        
        // 每 500 毫秒执行一次检查
        // 这个频率既能及时响应状态变化，又不会造成性能问题
        setInterval(checkForCoordsGroup, 500);
    }
    
    // ============================================================================
    // 页面初始化
    // ============================================================================
    // 在页面加载完成后初始化所有功能
    
    /**
     * 初始化操作区域遮罩
     * 
     * 功能说明：
     * - 使用 MutationObserver 监听操作区域的加载
     * - 当操作区域加载完成后，创建遮罩层
     * - 确保遮罩层在操作区域动态加载时也能正确创建
     */
    function initOperationZoneOverlay() {
        /**
         * 尝试创建遮罩层
         * 内部函数，在 MutationObserver 回调中调用
         */
        function tryCreateOverlay() {
            const overlay = createOperationZoneOverlay();
            if (overlay) {
                // 遮罩层创建成功，立即更新状态
                updateOperationZoneOverlay();
            }
        }
        
        // 使用 MutationObserver 等待操作区域加载
        const observer = new MutationObserver(function(mutations) {
            tryCreateOverlay();
        });
        
        // 开始观察整个文档的变化
        observer.observe(document.body, {
            childList: true,    // 观察子节点的添加和删除
            subtree: true       // 观察所有后代节点
        });
        
        // 立即执行一次，处理操作区域已经存在的情况
        setTimeout(tryCreateOverlay, 2000);
        
        // 定期检查遮罩层（处理操作区域被替换的情况）
        setInterval(function() {
            tryCreateOverlay();
        }, 1000);
    }
    
    /**
     * 强制视频组件互斥显示
     * 
     * 功能说明：
     * - 确保 video_display 和 no_video_display 不会同时显示
     * - 直接读取当前任务 ID (env_id)，根据配置决定是否显示视频
     * - 这是一个强制性的前端修正，确保 UI 状态与任务配置一致
     */
    function enforceVideoMutex() {
        // 配置：应该显示视频的环境 ID 列表 (对应 config.py 中的 DEMO_VIDEO_ENV_IDS)
        const DEMO_VIDEO_ENV_IDS = [
            "VideoPlaceOrder",
            "VideoUnmaskSwap",
            "VideoUnmask",
            "VideoRepick",
            "VideoPlaceButton",
            "InsertPeg",
            "MoveCube",
            "PatternLock",
            "RouteStick"
        ];

        const videoDisplay = document.querySelector('#demo_video'); // 视频容器
        const noVideoDisplay = document.querySelector('#no_video_display'); // 无视频提示
        
        // 尝试获取任务信息框
        const taskInfoTextarea = document.querySelector('#task_info_box textarea');
        const taskInfoDiv = document.querySelector('#task_info_box');
        
        if (!videoDisplay || !noVideoDisplay) return;
        
        // 1. 获取当前环境 ID
        let currentEnvId = null;
        let text = '';
        
        if (taskInfoTextarea) {
            text = taskInfoTextarea.value;
        } else if (taskInfoDiv) {
            text = taskInfoDiv.innerText || taskInfoDiv.textContent;
        }
        
        // 解析环境 ID: 预期格式 "Task: {EnvId} (Ep ...)"
        if (text) {
            const match = text.match(/Task:\s*([a-zA-Z0-9_]+)/);
            if (match && match[1]) {
                currentEnvId = match[1];
            }
        }
        
        // 2. 决定是否应该显示视频
        let shouldShowVideo = false;
        
        if (currentEnvId) {
            // 如果成功获取到环境 ID，根据配置判断
            shouldShowVideo = DEMO_VIDEO_ENV_IDS.includes(currentEnvId);
        } else {
            // 如果无法获取环境 ID (例如加载中)，回退到基于 DOM 可见性的判断
            const isVideoVisible = videoDisplay.style.display !== 'none' && !videoDisplay.classList.contains('hidden');
            shouldShowVideo = isVideoVisible;
        }
        
        // 3. 执行显示/隐藏逻辑
        if (shouldShowVideo) {
            // 必须显示视频
            if (videoDisplay.classList.contains('hidden')) videoDisplay.classList.remove('hidden');
            if (videoDisplay.style.display === 'none') videoDisplay.style.removeProperty('display');
            
            noVideoDisplay.style.display = 'none';
            noVideoDisplay.classList.add('hidden');
        } else {
            // 必须隐藏视频
            videoDisplay.style.display = 'none';
            videoDisplay.classList.add('hidden');
            
            if (noVideoDisplay.classList.contains('hidden')) noVideoDisplay.classList.remove('hidden');
            if (noVideoDisplay.style.display === 'none') noVideoDisplay.style.removeProperty('display');
        }
    }

    /**
     * 初始化所有功能
     * 
     * 功能说明：
     * - 初始化执行按钮监听器（坐标检查）
     * - 初始化 LeaseLost 错误处理器
     * - 初始化播放视频按钮监听器
     * - 初始化视频播放状态监听器（用于控制 execute 按钮）
     * - 初始化操作区域遮罩（用于禁用操作区域交互）
     * - 延迟启动坐标组高亮功能（等待 Gradio 完全加载）
     * - 启动视频组件互斥检查
     */
    function initializeAll() {
        // 初始化执行按钮监听器（为 EXECUTE 按钮添加坐标检查）
        initExecuteButtonListener();
        
        // 初始化 LeaseLost 错误处理器（检测并处理租约丢失错误）
        initLeaseLostHandler();
        
        // 初始化播放视频按钮监听器（为播放按钮添加点击事件）
        initPlayVideoButtonListener();
        
        // 初始化视频播放状态监听器（监听视频播放状态，用于控制 execute 按钮）
        initVideoPlayStateListener();
        
        // 初始化操作区域遮罩（用于在 execute 按钮禁用时禁用操作区域）
        initOperationZoneOverlay();
        
        // 启动视频互斥检查（定期执行）
        setInterval(enforceVideoMutex, 200);
        
        // 延迟启动坐标组高亮功能和演示视频组高亮功能
        // 等待 2 秒确保 Gradio 组件完全加载和渲染
        setTimeout(() => {
            applyCoordsGroupHighlight();
            // 更新演示视频组高亮状态（任务加载时，如果有演示视频）
            updateDemoVideoGroupHighlight();
            // 确保 execute 按钮状态正确（在组件加载后）
            updateExecuteButtonState();
        }, 2000);
    }
    
    // 根据文档加载状态选择初始化时机
    if (document.readyState === 'loading') {
        // 文档还在加载中，等待 DOMContentLoaded 事件
        document.addEventListener('DOMContentLoaded', function() {
            initializeAll();
        });
    } else {
        // 文档已加载完成，立即初始化
        initializeAll();
    }
})();
"""

CSS = f"""
/**
 * ================================================================================
 * CSS 样式定义
 * ================================================================================
 * 
 * 本 CSS 样式表定义了整个 UI 界面的外观和布局样式。
 * 使用 f-string 格式，支持动态插入配置值（如字体大小、视图高度等）。
 * 
 * 样式分类：
 * 1. 全局字体大小配置
 * 2. 组件布局样式
 * 3. 环境按钮颜色分类
 * 4. 高亮动画效果
 * 5. 按钮状态样式
 * 6. 加载遮罩层样式
 * ================================================================================
 */

/* ============================================================================
   全局字体大小配置
   ============================================================================
   目的：统一整个应用的字体大小，确保界面一致性
   策略：使用多个选择器确保覆盖所有 Gradio 元素，使用 !important 确保优先级
   ============================================================================ */

/* 基础元素字体大小 */
body, html {{
    font-size: {FONT_SIZE} !important;
}}

/* Gradio 容器字体大小 */
.gradio-container, #gradio-app {{
    font-size: {FONT_SIZE} !important;
}}

/* 所有文本元素的字体大小（全面覆盖） */
.gradio-container *, #gradio-app *, button, input, textarea, select, label, p, span, div, h1, h2, h3, h4, h5, h6, .gr-button, .gr-textbox, .gr-dropdown, .gr-radio, .gr-checkbox, .gr-markdown {{
    font-size: {FONT_SIZE} !important;
}}

/* 紧凑日志样式 */
/* 用途：限制日志显示区域的高度，使用等宽字体便于阅读，支持 HTML 格式 */
/* 注意：需要同时设置容器和内部所有元素，确保动态插入的内容也应用正确的字体大小 */
.compact-log,
#log_output {{
    max-height: 200px !important;        /* 最大高度 120px，超出部分滚动 */
    overflow-y: auto !important;         /* 垂直滚动 */
    font-family: monospace !important;    /* 等宽字体，便于对齐和阅读 */
    font-size: calc({FONT_SIZE} * 0.8) !important;     /* 使用稍小的字体大小，使等宽字体看起来与其他字体一致 */
    padding: 8px !important;             /* 内边距 */
    border: 1px solid rgba(204, 204, 204, 0.3) !important;   /* 半透明边框 */
    border-radius: 4px !important;       /* 圆角 */
    background-color: transparent !important; /* 透明背景 */
    line-height: 1.4 !important;        /* 行高 */
}}

/* 确保动态插入到 log_output 中的所有元素都使用正确的字体大小 */
.compact-log *,
#log_output *,
.compact-log div,
#log_output div {{
    font-size: calc({FONT_SIZE} * 0.8) !important;     /* 强制所有内部元素使用相同的字体大小 */
    font-family: monospace !important;    /* 确保等宽字体 */
}}

/* ============================================================================
   暗色模式样式 - System Log
   ============================================================================
   用途：当 Gradio 处于暗色模式时，将 System Log 字体颜色设置为白色
   ============================================================================ */
/* 检测多种暗色模式标识方式，确保兼容性 */
.dark .compact-log,
.dark #log_output,
.gradio-container.dark .compact-log,
.gradio-container.dark #log_output,
[data-theme="dark"] .compact-log,
[data-theme="dark"] #log_output,
.dark .compact-log *,
.dark #log_output *,
.gradio-container.dark .compact-log *,
.gradio-container.dark #log_output *,
[data-theme="dark"] .compact-log *,
[data-theme="dark"] #log_output *,
.dark .compact-log div,
.dark #log_output div,
.gradio-container.dark .compact-log div,
.gradio-container.dark #log_output div,
[data-theme="dark"] .compact-log div,
[data-theme="dark"] #log_output div {{
    color: #ffffff !important;           /* 白色字体 */
}}

/* 暗色模式下边框颜色适配 */
.dark .compact-log,
.dark #log_output,
.gradio-container.dark .compact-log,
.gradio-container.dark #log_output,
[data-theme="dark"] .compact-log,
[data-theme="dark"] #log_output {{
    border-color: rgba(255, 255, 255, 0.2) !important;  /* 浅色半透明边框 */
}}

/* ============================================================================
   组件布局样式
   ============================================================================
   定义各个 UI 组件的布局、边框、间距等样式
   ============================================================================ */

/* 实时观察图像容器（预留样式，当前为空） */
#live_obs {{ }}

/* 控制面板样式 */
/* 用途：为控制面板添加边框、内边距和背景色，使其与周围内容区分 */
#control_panel {{
    border: 1px solid #e5e7eb;      /* 浅灰色边框 */
    padding: 15px;                   /* 内边距 15px */
    border-radius: 8px;              /* 圆角边框，半径 8px */
    background-color: #f9fafb;       /* 浅灰色背景 */
}}

/* 参考区域样式 */
/* 用途：为顶部参考区域添加底部边框，与操作区域分隔 */
.ref-zone {{
    border-bottom: 2px solid #e5e7eb;  /* 底部边框，2px 宽度，浅灰色 */
    padding-bottom: 10px;              /* 底部内边距 10px */
    margin-bottom: 10px;               /* 底部外边距 10px */
    height: {REFERENCE_ZONE_HEIGHT} !important;  /* 固定高度（从配置读取） */
    overflow-y: auto !important;       /* 内容超出时显示垂直滚动条 */
}}

/* 组合视图 HTML 容器样式 */
/* 用途：移除组合视图容器的边框，使其与页面融合 */
#combined_view_html {{
    border: none !important;  /* 移除边框 */
}}

/* 组合视图中的图像样式 */
/* 用途：控制实时流图像的显示方式，确保图像居中且保持宽高比 */
#combined_view_html img {{
    max-width: 100%;                    /* 最大宽度 100%，响应式 */
    height: {REFERENCE_VIEW_HEIGHT};   /* 固定高度（从配置读取） */
    width: auto;                        /* 宽度自动，保持宽高比 */
    margin: 0 auto;                     /* 水平居中 */
    display: block;                     /* 块级元素 */
    border: none !important;            /* 移除边框 */
    border-radius: 8px;                 /* 圆角边框 */
    object-fit: contain;                 /* 保持宽高比，完整显示图像 */
}}

/* 演示视频容器样式 */
/* 用途：移除演示视频容器的边框并设置固定高度 */
#demo_video {{
    border: none !important;
    height: {DEMO_VIDEO_HEIGHT} !important;  /* 固定高度（从配置读取） */
}}

/* 修复：当组件被隐藏时，强制覆盖高度设置，确保不占据空间 */
#demo_video.hidden, 
#demo_video[style*="display: none"],
div.hidden #demo_video,
.gr-box.hidden #demo_video,
#no_video_display.hidden,
#no_video_display[style*="display: none"],
div.hidden #no_video_display {{
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    border: none !important;
}}

/* 演示视频元素样式 */
/* 用途：移除视频元素本身的边框并设置固定高度 */
#demo_video video {{
    border: none !important;
    height: {DEMO_VIDEO_HEIGHT} !important;  /* 固定高度（从配置读取） */
    width: 100% !important;                   /* 宽度自适应 */
}}

/* ============================================================================
   动作选择单选按钮样式
   ============================================================================
   目的：让每个动作选项占满一行，便于用户选择和阅读
   ============================================================================ */

/* 动作单选按钮组 - 每个选项占满一行 */
#action_radio .form-radio {{
    display: block !important;      /* 块级显示，每个选项占一行 */
    width: 100% !important;         /* 宽度 100% */
    margin-bottom: 8px !important;   /* 底部外边距 8px，选项之间间距 */
}}

/* 动作单选按钮标签样式 */
#action_radio .form-radio label {{
    width: 100% !important;         /* 标签宽度 100% */
    display: block !important;      /* 块级显示 */
}}

/* 动作单选按钮标签（备用选择器） */
#action_radio label {{
    display: block !important;      /* 块级显示 */
    width: 100% !important;         /* 宽度 100% */
    margin-bottom: 8px !important;  /* 底部外边距 8px */
}}
/* ============================================================================
   环境选择按钮颜色分类样式
   ============================================================================
   目的：为不同类别的环境按钮定义不同的颜色，通过视觉区分任务类型
   应用：通过 elem_classes 参数应用到对应的环境选择按钮上
   
   颜色方案说明：
   - Counting (计数任务): 蓝色系 - 浅蓝色背景，深蓝色文字，蓝色边框
   - Persistence (恒常任务): 绿色系 - 浅绿色背景，深绿色文字，绿色边框
   - Reference (参考任务): 黄色系 - 浅黄色背景，深黄色文字，黄色边框
   - Behavior (行为任务): 红色系 - 浅红色背景，深红色文字，红色边框
   ============================================================================ */

/* Counting 类别按钮样式 - 蓝色系 */
.btn-counting {{
    background-color: #dbeafe !important;  /* 浅蓝色背景 (#dbeafe) */
    color: #1e40af !important;              /* 深蓝色文字 (#1e40af) */
    border: 1px solid #bfdbfe !important;   /* 蓝色边框 (#bfdbfe) */
}}

/* Persistence 类别按钮样式 - 绿色系 */
.btn-persistence {{
    background-color: #dcfce7 !important;  /* 浅绿色背景 (#dcfce7) */
    color: #166534 !important;              /* 深绿色文字 (#166534) */
    border: 1px solid #bbf7d0 !important;   /* 绿色边框 (#bbf7d0) */
}}

/* Reference 类别按钮样式 - 黄色系 */
.btn-reference {{
    background-color: #fef9c3 !important;  /* 浅黄色背景 (#fef9c3) */
    color: #854d0e !important;              /* 深黄色文字 (#854d0e) */
    border: 1px solid #fde047 !important;   /* 黄色边框 (#fde047) */
}}

/* Behavior 类别按钮样式 - 红色系 */
.btn-behavior {{
    background-color: #fee2e2 !important;  /* 浅红色背景 (#fee2e2) */
    color: #991b1b !important;              /* 深红色文字 (#991b1b) */
    border: 1px solid #fecaca !important;   /* 红色边框 (#fecaca) */
}}

/* ============================================================================
   高亮动画效果样式
   ============================================================================
   目的：当需要用户选择坐标时，为坐标组和实时观察图像添加蓝色闪烁边框提示
   触发：通过 JavaScript 动态添加 CSS 类（coords-group-highlight, live-obs-highlight）
   ============================================================================ */

/* 坐标组高亮样式 */
/* 选择器说明：
   - #coords_group.coords-group-highlight: 通过 ID 和类选择
   - .gr-group.coords-group-highlight:has([id*="coords_box"]):not(:has([id*="exec_btn"])):
     通过类选择，且包含坐标输入框但不包含执行按钮的组
   注意：只有当 highlight 类存在时才应用样式（由 JavaScript 控制）
 */
#coords_group.coords-group-highlight,
.gr-group.coords-group-highlight:has([id*="coords_box"]):not(:has([id*="exec_btn"])) {{
    border: 3px solid #3b82f6 !important;        /* 蓝色边框，3px 宽度 */
    border-radius: 8px;                         /* 圆角边框 */
    padding: 15px;                              /* 内边距 15px */
    animation: bluePulse 1s ease-in-out infinite; /* 蓝色脉冲动画，1秒循环，缓入缓出 */
}}

/* 实时观察图像高亮样式 */
/* 注意：只有当 live-obs-highlight 类存在时才应用样式（由 JavaScript 控制） */
#live_obs.live-obs-highlight {{
    border: 3px solid #3b82f6 !important;        /* 蓝色边框，3px 宽度 */
    border-radius: 8px;                         /* 圆角边框 */
    animation: bluePulse 1s ease-in-out infinite; /* 蓝色脉冲动画 */
}}

/* 演示视频组高亮样式 */
/* 注意：只有当 demo-video-group-highlight 类存在时才应用样式（由 JavaScript 控制） */
#demo_video_group.demo-video-group-highlight {{
    border: 3px solid #3b82f6 !important;        /* 蓝色边框，3px 宽度 */
    border-radius: 8px;                         /* 圆角边框 */
    padding: 15px;                              /* 内边距 15px */
    animation: bluePulse 1s ease-in-out infinite; /* 蓝色脉冲动画，1秒循环，缓入缓出 */
}}

/* 蓝色脉冲动画关键帧定义 */
/* 功能：创建蓝色边框闪烁和阴影脉冲效果，吸引用户注意力 */
@keyframes bluePulse {{
    /* 起始和结束状态（0% 和 100%） */
    0%, 100% {{
        border-color: #3b82f6;                              /* 蓝色边框 (#3b82f6) */
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.8);        /* 无阴影（阴影大小为 0） */
    }}
    /* 中间状态（50%） */
    50% {{
        border-color: #2563eb;                              /* 更深的蓝色边框 (#2563eb) */
        box-shadow: 0 0 20px 8px rgba(59, 130, 246, 0.6);  /* 蓝色阴影，向外扩散 20px，模糊 8px */
    }}
}}
/* ============================================================================
   按钮状态样式
   ============================================================================
   定义各种按钮在不同状态下的样式（正常、禁用、悬停等）
   ============================================================================ */

/* Next Task 按钮 - 禁用状态样式 */
/* 用途：当按钮被禁用时（任务未完成），降低不透明度，表示不可点击 */
#next_task_btn:disabled,
#next_task_btn[disabled],
#next_task_btn.disabled {{
    opacity: 0.5 !important;  /* 不透明度 50%，视觉上变灰 */
}}

/* Play Video 按钮 - 禁用状态样式 */
/* 用途：当按钮被禁用时，降低不透明度并显示禁止光标 */
#play_video_btn:disabled,
#play_video_btn[disabled],
#play_video_btn.disabled {{
    opacity: 0.5 !important;           /* 不透明度 50% */
    cursor: not-allowed !important;   /* 禁止光标（表示不可点击） */
}}

/* ============================================
   Loading Overlay 全屏加载遮罩层样式
   ============================================
   功能说明：
   - 在用户点击登录/加载任务等按钮时，显示全屏半透明遮罩层
   - 遮罩层包含加载提示信息，防止用户在加载过程中进行其他操作
   - 加载完成后，通过更新 HTML 组件内容为空字符串来隐藏遮罩层
   
   样式说明：
   - .loading-overlay: 全屏遮罩层容器，固定在视口顶部，覆盖整个屏幕
   - .loading-content: 遮罩层中央的白色卡片，包含加载提示文本
   - .loading-spinner: 旋转动画类（预留，可用于添加旋转图标）
   ============================================ */
/* 全屏遮罩层容器样式 */
.loading-overlay {{
    position: fixed;        /* 固定定位，不随页面滚动 */
    top: 0;                 /* 从页面顶部开始 */
    left: 0;                /* 从页面左侧开始 */
    width: 100vw;           /* 宽度占满整个视口宽度 */
    height: 100vh;          /* 高度占满整个视口高度 */
    background: rgba(0, 0, 0, 0.5);  /* 半透明黑色背景，透明度50% */
    display: flex;          /* 使用 Flexbox 布局 */
    justify-content: center;  /* 水平居中 */
    align-items: center;    /* 垂直居中 */
    z-index: 9999;          /* 设置极高的层级，确保遮罩层显示在所有内容之上 */
}}
/* 遮罩层中央的白色卡片样式 */
.loading-content {{
    position: relative;     /* 相对定位，确保 z-index 生效 */
    background: #ffffff;    /* 纯白色背景，完全不透明 */
    padding: 30px 50px;      /* 内边距：上下30px，左右50px */
    border-radius: 10px;    /* 圆角边框，半径10px */
    text-align: center;      /* 文本居中对齐 */
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);  /* 阴影效果，增加立体感 */
    opacity: 1;             /* 确保卡片完全不透明 */
    z-index: 10000;         /* 设置更高的层级，确保显示在最上层（高于父元素的 9999） */
}}
/* 旋转动画样式（预留，可用于添加旋转加载图标） */
.loading-spinner {{
    animation: spin 1s linear infinite;  /* 旋转动画：1秒一圈，线性，无限循环 */
}}
/* 旋转动画关键帧定义 */
@keyframes spin {{
    from {{ transform: rotate(0deg); }}    /* 起始角度：0度 */
    to {{ transform: rotate(360deg); }}    /* 结束角度：360度（完整旋转一圈） */
}}
/* ============================================================================
   操作区域提示文字样式
   ============================================================================
   用途：为页面顶部的操作提示文字定义样式，使其显示在标题右侧
   ============================================================================ */

/* Operation Zone 提示文字样式 - 左对齐 */
#operation_hint {{
    text-align: left !important;              /* 文本左对齐 */
    font-size: {FONT_SIZE} !important;        /* 使用统一的字体大小 */
    padding: 0 !important;                    /* 无内边距 */
    margin: 0 !important;                     /* 无外边距 */
    font-weight: 500 !important;              /* 中等字重（500） */
    display: block !important;                 /* Block 布局，允许内部元素控制布局 */
    flex: 1 !important;                       /* 占据剩余空间 */
}}

/* ============================================================================
   操作区域遮罩样式
   ============================================================================
   目的：当 execute 按钮被禁用时，为操作区域添加灰色遮罩并禁用所有交互
   ============================================================================ */

/* 操作区域组容器样式 - 为遮罩提供定位上下文 */
[id*="operation_zone_group"] {{
    position: relative !important;  /* 为遮罩层提供定位上下文 */
    height: {OPERATION_ZONE_HEIGHT} !important;  /* 固定高度（从配置读取） */
    overflow-y: auto !important;    /* 内容超出时显示垂直滚动条 */
}}

/* 操作区域遮罩层样式 */
.operation-zone-overlay {{
    position: absolute !important;        /* 绝对定位，相对于 operation_zone_group */
    top: 0 !important;                     /* 从顶部开始 */
    left: 0 !important;                     /* 从左侧开始 */
    width: 100% !important;                 /* 宽度 100%，覆盖整个区域 */
    height: 100% !important;                /* 高度 100%，覆盖整个区域 */
    background-color: rgba(128, 128, 128, 0.6) !important;  /* 灰色半透明背景（60% 透明度） */
    z-index: 9999 !important;              /* 极高的层级，确保遮罩显示在最上层 */
    pointer-events: auto !important;        /* 启用鼠标交互，阻止点击穿透到下层 */
    transition: opacity 0.3s ease-in-out !important;  /* 过渡动画效果 */
    display: none !important;              /* 默认隐藏，通过 JavaScript 控制显示 */
    cursor: not-allowed !important;        /* 显示禁止光标 */
}}

/* 遮罩层显示状态 */
.operation-zone-overlay.active {{
    display: block !important;            /* 显示遮罩 */
}}

/* 当遮罩层激活时，禁用操作区域组下所有子元素的交互 */
[id*="operation_zone_group"]:has(.operation-zone-overlay.active) * {{
    pointer-events: none !important;      /* 禁用所有子元素的交互 */
    user-select: none !important;          /* 禁用文本选择 */
    -webkit-user-select: none !important;  /* Safari 浏览器 */
    -moz-user-select: none !important;     /* Firefox 浏览器 */
    -ms-user-select: none !important;      /* IE/Edge 浏览器 */
}}"""
if RESTRICT_VIDEO_PLAYBACK:
    CSS += """
    #demo_video {
        pointer-events: none;
    }
    """


def create_ui_blocks():
    """
    创建并返回 Gradio Blocks 对象
    
    功能说明：
    - 定义整个应用的 UI 布局结构
    - 创建所有界面组件（加载界面、模式选择、登录、主界面等）
    - 绑定所有事件处理器到对应的回调函数
    - 配置组件的可见性、样式和交互属性
    
    布局结构：
    1. 页面标题和操作提示（顶部）
    2. 全屏加载遮罩层（覆盖整个页面）
    3. 状态变量（uid_state, username_state）
    4. 加载界面组（loading_group）
    5. 登录界面组（login_group）
    6. 主界面组（main_interface）
       - 参考区域（顶部）
       - 操作区域（底部）
    7. 事件绑定（所有按钮和交互组件的事件处理）
    
    Returns:
        demo: Gradio Blocks 对象，包含完整的 UI 布局和事件绑定
    """
    # 注意：在 Gradio 6.0+ 中，css 和 js 参数应该传递给 launch() 或 mount_gradio_app()，
    # 而不是 Blocks 构造函数。CSS 和 JS 代码会在应用启动时自动注入。
    
    with gr.Blocks(title="Oracle Planner Interface") as demo:
        # ========================================================================
        # 页面标题和操作提示区域
        # ========================================================================
        # 用途：显示应用标题和用户操作提示
        # 布局：一行，标题和提示合并在一起
        with gr.Row():
            # 应用标题和操作提示合并
            gr.Markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                    <h2 style="margin: 0;">HistoryBench Human Evaluation 🚀🚀🚀</h2>
                    <h2 style="margin: 0;">Read the Task Goal 🏅, then select action in Action Selection 🧠 (and Keypoint Selection 🎯) to finish the task</h2>
                </div>
                """, 
                elem_id="operation_hint"
            )
        
        # ============================================
        # Loading Overlay 全屏加载遮罩层组件
        # ============================================
        # 功能说明：
        # - 这是一个 HTML 组件，用于显示全屏加载遮罩层
        # - 初始值为空字符串（隐藏状态）
        # - 当用户点击登录/加载任务等按钮时，show_loading_info() 函数会返回包含遮罩层 HTML 的字符串
        # - 加载完成后，回调函数返回空字符串，清空组件内容，遮罩层自动隐藏
        # 
        # 工作原理：
        # 1. 显示：通过更新组件内容为包含 .loading-overlay 的 HTML 字符串来显示遮罩层
        # 2. 隐藏：通过更新组件内容为空字符串 "" 来隐藏遮罩层
        # 
        # 使用场景：
        # - 用户点击 Login 按钮时
        # - 用户点击 Next Task 按钮时
        # ============================================
        loading_overlay = gr.HTML(value="", elem_id="loading_overlay")
        
        # ========================================================================
        # 状态变量
        # ========================================================================
        # 用途：在 UI 组件和回调函数之间传递状态信息
        # 注意：State 组件不会在界面上显示，只用于存储和传递数据
        
        # 用户会话唯一标识符
        # 类型：字符串或 None
        # 用途：标识当前用户会话，用于获取会话数据、视频流等
        # 生成时机：用户登录时由 user_manager 生成
        # 使用场景：标识当前会话，用于获取会话数据、视频流等
        uid_state = gr.State(value=None)
        
        # 用户名状态
        # 类型：字符串
        # 用途：记录当前登录的用户名
        # 更新时机：用户登录成功后
        # 使用场景：在回调函数中获取当前用户信息
        username_state = gr.State(value="")
        
        # ========================================================================
        # 加载界面组 (Loading Section)
        # ========================================================================
        # 用途：应用启动时显示加载提示信息
        # 可见性：初始为 True（显示），登录后隐藏
        with gr.Group(visible=True) as loading_group:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        # ========================================================================
        # 登录界面组 (Login Section)
        # ========================================================================
        # 用途：用户选择用户名并登录
        # 可见性：初始为 False（隐藏），加载完成后显示
        # 显示时机：应用初始化完成后（init_app 函数控制）
        with gr.Group(visible=False) as login_group:
            gr.Markdown("### User Login")
            with gr.Row():
                # 用户名下拉选择框
                # 数据源：从 user_manager.user_tasks 获取可用用户名列表
                # 功能：用户选择自己的用户名
                # 交互：可下拉选择，不可输入
                available_users = list(user_manager.user_tasks.keys())
                username_input = gr.Dropdown(
                    choices=available_users,  # 可用用户名列表
                    label="Username",
                    value=None                # 初始值：无选择
                )
                
                # 登录按钮
                # 功能：执行登录和任务加载
                # 事件：点击后调用 login_and_load_task() 函数
                # 样式：主要按钮样式（蓝色背景）
                login_btn = gr.Button("Login", variant="primary")
            
            # 登录消息显示
            # 功能：显示登录成功/失败消息
            # 更新时机：登录操作完成后
            login_msg = gr.Markdown("")

        # ========================================================================
        # 主界面组 (Main Interface)
        # ========================================================================
        # 用途：任务执行的主要工作区域
        # 可见性：初始为 False（隐藏），登录成功后显示
        # 结构：分为两个容器
        #   1. 顶部容器：参考区域（Reference Zone）- 显示任务信息和参考视图
        #   2. 底部容器：操作区域（Operation Zone）- 用户交互和操作控制
        with gr.Group(visible=False) as main_interface:
            # ====================================================================
            # 顶部容器：参考区域 (Reference Zone)
            # ====================================================================
            # 用途：显示任务信息、目标和参考视图（演示视频或实时流）
            # 布局：一行三列，左侧文本信息，中间执行实时流，右侧演示视频
            # 样式：通过 elem_classes="ref-zone" 应用底部边框样式
            with gr.Row(elem_classes="ref-zone"):
                # ================================================================
                # 左侧列：文本信息区域
                # ================================================================
                # 用途：显示任务进度、目标和系统日志
                # 布局：垂直排列三个组
                with gr.Column(scale=TEXT_INFO_SCALE):
                    # 进度跟踪组
                    with gr.Group():
                        gr.Markdown("### 1. Progress Tracker 📊🥳")
                        with gr.Row():
                            # 当前任务信息框
                            # 功能：显示当前任务编号、环境ID等信息
                            # 更新时机：任务加载时、执行步骤后
                            # 布局：占 2/3 宽度（scale=2）
                            task_info_box = gr.Textbox(
                                label="Current Task", 
                                interactive=False,  # 不可编辑
                                show_label=False,   # 不显示标签
                                scale=2,            # 宽度比例 2
                                elem_id="task_info_box"
                            )
                            
                            # 进度信息框
                            # 功能：显示任务完成进度（已完成/总数）
                            # 更新时机：执行步骤后
                            # 布局：占 1/3 宽度（scale=1）
                            progress_info_box = gr.Textbox(
                                label="Progress", 
                                interactive=False, 
                                show_label=False, 
                                scale=1
                            )
                    
                    # 任务目标组
                    with gr.Group():
                        gr.Markdown("### 2. Task Goal 🏅")
                        # 任务目标/指令框
                        # 功能：显示当前任务的详细指令和目标
                        # 更新时机：任务加载时
                        # 行数：3 行（可滚动）
                        goal_box = gr.Textbox(
                            label="Instruction", 
                            lines=3,              # 3 行高度
                            interactive=False,    # 不可编辑
                            show_label=False      # 不显示标签
                        )
                    
                    # 系统日志组
                    with gr.Group():
                        gr.Markdown("### 3. System Log 📝")
                        # 系统日志输出框
                        # 功能：显示系统运行日志、错误信息、执行结果
                        # 更新时机：执行步骤时、系统事件发生时
                        # 样式：紧凑日志样式（.compact-log），等宽字体，最大高度 120px
                        # 支持彩色显示：成功消息绿色，错误消息红色
                        log_output = gr.HTML(
                            value="",                      # 初始值为空
                            elem_classes="compact-log",    # 紧凑日志样式类
                            elem_id="log_output"           # ID 用于 CSS 定位
                        )

                # ================================================================
                # 中间列：执行实时流区域
                # ================================================================
                # 用途：显示执行阶段的实时流（机器人执行画面）
                with gr.Column(scale=COMBINED_VIEW_SCALE):
                    with gr.Group(visible=True) as combined_view_group:
                        # 标题行
                        gr.Markdown("### Execution LiveStream 🦾")
                        
                        # 内容行 - 单列布局
                        with gr.Column(scale=1):
                            # 执行实时流显示（HTML 组件）
                            # 功能：通过 MJPEG 流实时显示机器人执行画面
                            # 数据源：/video_feed/{uid} 端点（由 streaming_service.py 提供）
                            # 更新：实时流式传输，无需手动刷新
                            # 实现：使用 HTML img 标签，src 指向 MJPEG 流端点
                            combined_display = gr.HTML(
                                value="<div id='combined_view_html'><p>Waiting for video stream...</p></div>",
                                elem_id="combined_view_html"
                            )

                # ================================================================
                # 右侧列：演示视频区域
                # ================================================================
                # 用途：显示任务演示视频（如果有的话）
                with gr.Column(scale=DEMO_VIDEO_SCALE):
                    with gr.Group(visible=True, elem_id="demo_video_group") as demo_video_group:
                        # 标题行
                        gr.Markdown("### Watch video and remember robot actions 👀✍️")
                        
                        # 内容行 - 单列布局
                        with gr.Column(scale=1):
                            # 视频元素 ID 配置
                            # 如果限制视频播放控制，使用 "demo_video" ID（用于 JavaScript 控制）
                            video_elem_id = "demo_video" if RESTRICT_VIDEO_PLAYBACK else None
                            video_autoplay = False  # 不自动播放，等待用户点击按钮
                            
                            # 演示视频播放器
                            # 功能：播放任务演示视频
                            # 控制：禁用用户控制，只能通过按钮播放
                            # 限制：如果 RESTRICT_VIDEO_PLAYBACK=True，禁用所有鼠标交互
                            # 高度：通过 CSS 控制（DEMO_VIDEO_HEIGHT 配置）
                            video_display = gr.Video(
                                label="Demonstration Video", 
                                interactive=False,      # 禁用用户控制（播放/暂停等）
                                elem_id=video_elem_id,  # 元素 ID（用于 JavaScript 控制）
                                autoplay=video_autoplay, # 不自动播放
                                show_label=False,       # 不显示标签
                                visible=False           # 初始隐藏，只有在有视频时才显示
                            )
                            
                            # 无视频提示显示
                            # 功能：当没有演示视频时，显示"No video"黑色文字
                            # 初始状态：隐藏（当有视频时隐藏）
                            # 高度：占用和演示视频相同的高度（DEMO_VIDEO_HEIGHT = 30vh）
                            no_video_display = gr.HTML(
                                value=f"<div style='color: black; font-size: 20px; text-align: center; height: {DEMO_VIDEO_HEIGHT}; display: flex; align-items: center; justify-content: center;'>No video</div>",
                                visible=False,  # 初始隐藏
                                elem_id="no_video_display"
                            )
                            
                            # 播放演示视频按钮
                            # 功能：触发视频播放（唯一播放方式）
                            # 事件：点击后调用 play_demo_video() 函数
                            # 样式：主要按钮样式（蓝色背景），大号按钮
                            play_video_btn = gr.Button(
                                "Watch Video Input🎬(only play once)", 
                                variant="primary",      # 主要按钮样式
                                size="lg",              # 大号按钮
                                visible=True,           # 可见
                                interactive=True,       # 可交互
                                elem_id="play_video_btn"
                            )

            # ====================================================================
            # 底部容器：操作区域 (Operation Zone)
            # ====================================================================
            # 用途：用户交互和操作控制的主要区域
            # 可见性：初始为 False（隐藏），登录成功后通过回调函数设置为 True（立即显示）
            # 布局：一行三列，左侧动作选择，中间关键点选择，右侧控制面板
            with gr.Group(visible=False) as operation_zone_group:
                with gr.Row():
                    # ============================================================
                    # 左侧列：动作选择 (Action Selection)
                    # ============================================================
                    # 用途：显示可选的动作列表，用户选择要执行的动作
                    # 宽度：由 ACTION_SCALE 配置控制
                    with gr.Column(scale=ACTION_SCALE):
                        gr.Markdown("### Action Selection 🧠")
                        # 动作单选按钮组
                        # 功能：显示可选的动作列表，用户选择要执行的动作
                        # 数据源：从任务配置中获取可用动作列表
                        # 样式：每个选项占满一行（通过 CSS #action_radio 实现）
                        # 事件：选择改变时调用 on_option_select() 函数
                        # 类型：value（返回选中的值，而不是索引）
                        options_radio = gr.Radio(
                            choices=[],           # 初始为空，任务加载时填充
                            label="Action", 
                            type="value",         # 返回选中的值
                            show_label=False,     # 不显示标签
                            elem_id="action_radio"
                        )

                    # ============================================================
                    # 中间列：关键点选择 (Keypoint Selection)
                    # ============================================================
                    # 用途：显示当前观察图像，用户点击选择关键点坐标
                    # 宽度：由 LIVE_OBSERVATION_SCALE 配置控制（通常最大）
                    with gr.Column(scale=LIVE_OBSERVATION_SCALE):
                        gr.Markdown("### Keypoint Selection 🎯")
                        # 实时观察图像显示（可点击）
                        # 功能：显示当前观察图像，用户点击选择关键点坐标
                        # 交互：支持点击选择（select 事件）
                        # 事件：点击后调用 on_map_click() 函数
                        # 高亮：当需要选择坐标时，显示蓝色闪烁边框提示（JavaScript 控制）
                        # 类型：PIL 图像（Python Imaging Library 格式）
                        img_display = gr.Image(
                            label="Live Observation", 
                            interactive=False,   # 禁用直接编辑，但支持点击选择
                            type="pil",           # PIL 图像格式
                            elem_id="live_obs",  # 元素 ID（用于 JavaScript 高亮）
                            show_label=False      # 不显示标签
                        )

                    # ============================================================
                    # 右侧列：控制面板 (Control Panel)
                    # ============================================================
                    # 用途：显示坐标信息、执行按钮和控制按钮
                    # 宽度：由 CONTROL_SCALE 配置控制
                    with gr.Column(scale=CONTROL_SCALE):
                        gr.Markdown("### Control Panel 🎛️")
                        
                        # 坐标组（条件显示）
                        # 可见性：根据选择的动作类型动态显示/隐藏
                        #   - 需要坐标的动作：显示
                        #   - 不需要坐标的动作：隐藏
                        # 高亮：当需要选择坐标且未选择时，显示蓝色闪烁边框（JavaScript 控制）
                        with gr.Group(visible=False, elem_id="coords_group") as coords_group:
                            gr.Markdown("**Coords** 📍")
                            # 坐标文本框
                            # 功能：显示用户点击图像后选择的坐标值
                            # 初始值："please click the keypoint selection image"
                            # 更新：用户点击图像后更新为实际坐标值
                            # 验证：执行前检查是否已选择坐标（通过 JavaScript）
                            coords_box = gr.Textbox(
                                label="Coords", 
                                value="",              # 初始值为空
                                interactive=False,     # 不可编辑（只能通过点击图像选择）
                                show_label=False,      # 不显示标签
                                elem_id="coords_box"  # 元素 ID（用于 JavaScript 验证）
                            )
                        
                        # 执行按钮
                        # 功能：执行用户选择的动作和坐标
                        # 事件：点击后调用 execute_step() 函数
                        # 样式：红色停止按钮样式（variant="stop"），大号按钮
                        # 验证：执行前通过 JavaScript 检查坐标是否已选择
                        exec_btn = gr.Button(
                            "EXECUTE 🤖", 
                            variant="stop",       # 停止按钮样式（红色）
                            size="lg",            # 大号按钮
                            elem_id="exec_btn"    # 元素 ID（用于 JavaScript 验证）
                        )
                        
                        # 下一个任务按钮
                        # 功能：加载下一个任务
                        # 初始状态：禁用（任务完成后启用）
                        # 事件：点击后调用 load_next_task_wrapper() 函数
                        # 样式：主要按钮样式（蓝色背景），大号按钮
                        # 特殊处理：对于 user_test 用户，会跳转回环境选择界面
                        next_task_btn = gr.Button(
                            "Next Task 🔄", 
                            variant="primary",        # 主要按钮样式
                            interactive=False,       # 初始禁用
                            elem_id="next_task_btn"
                        )
                
                # ====================================================================
                # 任务提示区域 (Task Hint Zone)
                # ====================================================================
                # 用途：显示当前任务的提示信息，帮助用户理解任务要求
                # 可见性：初始为 True（显示），任务加载后自动填充内容
                # 布局：垂直排列，Task Hint 在上，教程视频在下，位于操作区域下方
                with gr.Group(visible=True) as task_hint_group:
                    gr.HTML('<hr style="border-top: 3px solid #888; margin: 20px 0;">')
                    with gr.Column():
                        # 上方：Task Hint 标题和内容
                        gr.Markdown("### Task Hint 💡")
                        # 任务提示 Markdown 组件
                        # 功能：显示当前任务的提示信息，自动加载，无需点击按钮
                        # 初始值：空字符串（任务加载时自动填充）
                        # 更新：任务加载时自动更新为对应的提示内容
                        task_hint_display = gr.Markdown(
                            value="",           # 初始值为空，任务加载时自动填充
                            elem_id="task_hint_display"
                        )
                        # 横线分隔
                        gr.HTML('<hr style="border-top: 3px solid #888; margin: 20px 0;">')
                        # 下方：教程视频标题和播放器（仅在episode 98时显示）
                        gr.Markdown("### Tutorial Video🧑‍🏫")
                        # 教程视频播放器
                        # 功能：仅在episode 98时，根据当前任务的 env_id 自动加载对应的教程视频
                        # 初始值：None（无视频）
                        # 可见性：初始为 False，仅在episode 98且有视频时显示
                        # 交互：允许用户控制播放/暂停/进度条等
                        tutorial_video_display = gr.Video(
                            label="Tutorial Video",
                            value=None,         # 初始值为 None
                            visible=False,      # 初始隐藏，仅在episode 98时显示
                            interactive=True,   # 允许用户交互控制
                            show_label=False    # 不显示标签
                        )

        # --- Event Wiring ---

        # ============================================
        # 1. Login 按钮事件绑定
        # ============================================
        # 功能说明：
        # - 用户点击 Login 按钮时，首先显示全屏加载遮罩层
        # - 然后执行登录和任务加载操作
        # - 加载完成后，自动隐藏遮罩层
        # 
        # 事件链流程：
        # 1. click 事件：调用 show_loading_info() 显示遮罩层
        # 2. .then() 链：调用 login_and_load_task() 执行登录和加载
        # 3. login_and_load_task() 返回时，loading_overlay 被设置为空字符串，遮罩层消失
        # ============================================
        login_btn.click(
            # 第一步：显示加载遮罩层
            # 调用 show_loading_info() 函数，返回包含遮罩层 HTML 的字符串
            # 该字符串会更新到 loading_overlay 组件，从而显示全屏遮罩层
            fn=show_loading_info,
            outputs=[loading_overlay]  # 输出到 loading_overlay 组件，显示遮罩层
        ).then(
            # 第二步：执行登录和任务加载
            # 在遮罩层显示后，执行实际的登录和任务加载操作
            fn=login_and_load_task,
            inputs=[username_input, uid_state],  # 输入：用户名和会话ID
            outputs=[
                uid_state, 
                login_group, 
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                no_video_display,  # 无视频提示显示
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                play_video_btn,
                coords_group,
                task_hint_display,  # 任务提示显示（自动加载）
                tutorial_video_display,  # 教程视频显示（仅在episode 98时显示）
                loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
            ]
        ).then(
            fn=lambda u: u,
            inputs=[username_input],
            outputs=[username_state]
        )
        
        # ============================================
        # 1.5 Next Task 按钮事件绑定
        # ============================================
        # 功能说明：
        # - 用户完成当前任务后，点击 "Next Task" 按钮加载下一个任务
        # - 首先显示全屏加载遮罩层
        # - 然后加载下一个任务（如果当前任务已有 actions，则创建新的 attempt）
        # - 对于 user_test 用户，next task 时会跳转回环境 ID 选择界面
        # - 加载完成后，自动隐藏遮罩层
        # 
        # 事件链流程：
        # 1. click 事件：调用 show_loading_info() 显示遮罩层
        # 2. .then() 链：调用 load_next_task_wrapper() 加载下一个任务
        # 3. load_next_task_wrapper() 返回时，loading_overlay 被设置为空字符串，遮罩层消失
        # 
        # 特殊处理：
        # - 如果当前任务已有 actions，会自动创建新的 attempt
        # - 对于 user_test 用户，会跳转回环境 ID 选择界面，而不是直接加载下一个任务
        # ============================================
        next_task_btn.click(
            # 第一步：显示加载遮罩层
            fn=show_loading_info,
            outputs=[loading_overlay]  # 输出到 loading_overlay 组件，显示遮罩层
        ).then(
            # 第二步：加载下一个任务
            fn=load_next_task_wrapper,
            inputs=[username_state, uid_state],  # 输入：用户名状态和会话ID
            outputs=[
                uid_state, 
                login_group, 
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                no_video_display,  # 无视频提示显示
                task_info_box, 
                progress_info_box, 
                login_btn, 
                next_task_btn, 
                exec_btn, 
                demo_video_group, 
                combined_view_group, 
                operation_zone_group, 
                play_video_btn, 
                coords_group, 
                task_hint_display,  # 任务提示显示（自动加载）
                tutorial_video_display,  # 教程视频显示（仅在episode 98时显示）
                loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
            ]
        )
        
        # ========================================================================
        # 2. 播放演示视频按钮事件绑定
        # ========================================================================
        # 功能说明：
        # - 用户点击"播放演示视频"按钮时触发
        # - 启用视频播放
        # 
        # 事件流程：
        # 1. 用户点击播放按钮
        # 2. 调用 play_demo_video() 函数
        # 3. 更新按钮状态（可能禁用播放按钮）
        # ========================================================================
        play_video_btn.click(
            fn=play_demo_video,                    # 回调函数：处理视频播放逻辑
            inputs=[play_video_btn, uid_state],    # 输入：播放按钮本身和会话ID（用于状态更新）
            outputs=[play_video_btn]  # 输出：更新播放按钮状态
        )

        # ========================================================================
        # 3. 视频播放结束事件绑定
        # ========================================================================
        # 功能说明：
        # - 演示视频播放结束时触发
        # - 更新系统日志提示选择动作
        # ========================================================================
        video_display.end(
            fn=on_video_end,
            inputs=[uid_state],
            outputs=[log_output]
        )

        # ========================================================================
        # 4. 图像点击事件绑定
        # ========================================================================
        # 功能说明：
        # - 用户在实时观察图像上点击选择关键点时触发
        # - 处理点击坐标，更新坐标输入框
        # 
        # 事件流程：
        # 1. 用户在图像上点击
        # 2. 调用 on_map_click() 函数
        # 3. 获取点击坐标（相对于图像的像素坐标）
        # 4. 根据选择的动作类型处理坐标
        # 5. 更新坐标输入框（coords_box）显示选择的坐标
        # 6. 更新图像显示（可能添加标记点）
        # 
        # 注意：
        # - 使用 select 事件（Gradio Image 组件的点击选择事件）
        # - 需要先选择动作（options_radio）才能选择坐标
        # ========================================================================
        img_display.select(
            fn=on_map_click,                       # 回调函数：处理图像点击事件
            inputs=[uid_state, username_state, options_radio],  # 输入：会话ID、用户名、选择的动作
            outputs=[img_display, coords_box]      # 输出：更新图像显示和坐标输入框
        )
        
        # ========================================================================
        # 5. 动作选择改变事件绑定
        # ========================================================================
        # 功能说明：
        # - 用户选择不同的动作时触发
        # - 根据动作类型显示/隐藏坐标组
        # - 重置坐标输入框
        # 
        # 事件流程：
        # 1. 用户选择动作（options_radio 改变）
        # 2. 调用 on_option_select() 函数
        # 3. 检查动作是否需要坐标
        # 4. 如果需要坐标：显示坐标组（coords_group），重置坐标输入框
        # 5. 如果不需要坐标：隐藏坐标组，清空坐标输入框
        # 6. 更新图像显示（可能显示不同的观察图像）
        # 
        # 注意：
        # - 使用 change 事件（值改变时触发）
        # - 坐标组的可见性由动作类型决定
        # ========================================================================
        options_radio.change(
            fn=on_option_select,                   # 回调函数：处理动作选择改变
            inputs=[uid_state, username_state, options_radio],  # 输入：会话ID、用户名、选择的动作
            outputs=[coords_box, img_display, coords_group]  # 输出：更新坐标框、图像显示、坐标组可见性
        )

        # ========================================================================
        # 6. 执行按钮事件绑定
        # ========================================================================
        # 功能说明：
        # - 用户点击"EXECUTE"按钮时触发
        # - 执行用户选择的动作和坐标
        # - 更新任务进度和界面状态
        # 
        # 事件流程：
        # 1. 用户点击执行按钮
        # 2. JavaScript 验证坐标是否已选择（如果需要坐标）
        # 3. 调用 execute_step() 函数
        # 4. 执行动作（通过后端 API）
        # 5. 更新图像显示（显示执行后的状态）
        # 6. 更新系统日志（显示执行结果）
        # 7. 更新任务信息和进度
        # 8. 如果任务完成，启用"下一个任务"按钮
        # 9. 更新坐标组可见性（根据新的状态）
        # 
        # 验证：
        # - 执行前通过 JavaScript 检查坐标是否已选择（如果需要坐标）
        # - 如果坐标未选择，显示提示并阻止执行
        # ========================================================================
        exec_btn.click(
            fn=execute_step,                       # 回调函数：执行动作步骤
            inputs=[uid_state, username_state, options_radio, coords_box],  # 输入：会话ID、用户名、动作、坐标
            outputs=[
                img_display,                       # 更新图像显示（执行后的状态）
                log_output,                        # 更新系统日志（执行结果）
                task_info_box,                     # 更新任务信息
                progress_info_box,                 # 更新进度信息
                next_task_btn,                     # 更新"下一个任务"按钮状态（任务完成时启用）
                exec_btn,                          # 更新执行按钮状态
                coords_group                       # 更新坐标组可见性
            ]
        )
        
        # ========================================================================
        # 7. 应用加载事件绑定（自动初始化）
        # ========================================================================
        # 功能说明：
        # - 应用启动时自动触发
        # - 如果URL中包含 'user' 或 'username' 参数，直接登录并进入主界面
        # - 否则显示登录界面
        # 
        # 事件流程：
        # 1. 应用加载完成（demo.load 事件）
        # 2. 调用 init_app() 函数
        # 3. 如果URL中有用户名参数：
        #    - 生成新的会话ID（uid_state）
        #    - 自动调用 login_and_load_task() 登录并加载任务
        #    - 直接显示主界面（main_interface）
        # 4. 如果URL中没有用户名参数：
        #    - 生成新的会话ID（uid_state）
        #    - 隐藏加载界面（loading_group）
        #    - 显示登录界面（login_group）
        #    - 隐藏主界面
        #    - 重置所有组件到初始状态
        # 
        # 注意：
        # - 这是应用的入口点，在页面加载时自动执行
        # - 不需要用户交互，自动初始化界面
        # ========================================================================
        demo.load(
            fn=init_app,                           # 回调函数：初始化应用
            inputs=[],                             # 无输入（自动触发）
            outputs=[
                uid_state,                         # 生成新的会话ID
                loading_group,                      # 隐藏加载界面
                login_group,                       # 显示登录界面
                main_interface,                    # 隐藏主界面
                login_msg,                         # 重置登录消息
                img_display,                       # 重置图像显示
                log_output,                        # 重置日志输出
                options_radio,                    # 重置动作选择
                goal_box,                          # 重置任务目标
                coords_box,                        # 重置坐标框
                combined_display,                  # 重置组合视图
                video_display,                     # 重置视频显示
                no_video_display,                  # 重置无视频提示显示
                task_info_box,                     # 重置任务信息
                progress_info_box,                 # 重置进度信息
                login_btn,                         # 重置登录按钮
                next_task_btn,                     # 重置下一个任务按钮
                exec_btn,                          # 重置执行按钮
                username_state,                    # 重置用户名状态
                demo_video_group,                  # 重置演示视频组
                combined_view_group,               # 重置组合视图组
                operation_zone_group,              # 重置操作区域组
                play_video_btn,                    # 重置播放视频按钮
                coords_group,                      # 重置坐标组
                task_hint_display,                 # 重置任务提示显示
                tutorial_video_display             # 重置教程视频显示
            ]
        )
    
    return demo
