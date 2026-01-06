"""
UI布局模块
定义Gradio界面组件、CSS和JS

界面组件总结：
==================

【主要界面组】
1. loading_group - 加载界面（初始显示）
2. landing_group - 模式选择界面（落地页）
3. login_group - 登录界面
4. env_selection_group - 环境ID选择界面（仅测试模式）
5. main_interface - 主界面（执行界面）

【模式选择界面 (landing_group)】
- test_btn - 自由尝试模式按钮
- record_btn - 录制模式按钮

【登录界面 (login_group)】
- username_input - 用户名下拉选择框
- login_btn - 登录按钮
- login_msg - 登录消息显示

【环境选择界面 (env_selection_group)】
- env_buttons[] - 环境ID按钮列表（15个按钮，每行5个）
- env_selection_msg - 环境选择消息

【主界面 (main_interface)】
├─ 顶部容器：参考区域 (Reference Zone)
│  ├─ 左侧列 (30%)：
│  │  ├─ task_info_box - 当前任务信息框
│  │  ├─ progress_info_box - 进度信息框
│  │  ├─ goal_box - 任务目标/指令框
│  │  └─ log_output - 系统日志输出
│  │
│  └─ 右侧列 (70%)：
│     ├─ demo_video_group - 演示视频组（第一阶段显示）
│     │  ├─ note2_demo - 任务提示（可折叠手风琴）
│     │  ├─ video_display - 演示视频播放器
│     │  ├─ play_video_btn - 播放演示视频按钮
│     │  └─ confirm_demo_btn - 开始任务按钮
│     │
│     └─ combined_view_group - 组合视图组（执行阶段显示）
│        ├─ note2 - 任务提示（可折叠手风琴）
│        └─ combined_display - 执行实时流显示（HTML组件）
│

└─ 底部容器：操作区域 (Operation Zone)
   ├─ 左侧列：动作选择 (Action Selection)
   │  └─ options_radio - 动作单选按钮组
   │
   ├─ 中间列：关键点选择 (Keypoint Selection)
   │  └─ img_display - 实时观察图像显示（可点击）
   │
   └─ 右侧列：控制面板 (Control Panel)
      ├─ coords_group - 坐标组（条件显示）
      │  └─ coords_box - 坐标文本框
      ├─ exec_btn - 执行按钮
      ├─ next_task_btn - 下一个任务按钮
      └─ back_to_landing_btn - 返回模式选择按钮

【状态变量】
- uid_state - 用户唯一标识符状态
- username_state - 用户名状态

【CSS样式类】
- .ref-zone - 参考区域样式
- .compact-log - 紧凑日志样式
- .coords-group-highlight - 坐标组高亮样式
- .live-obs-highlight - 实时观察高亮样式

【JavaScript功能】
- 视频自动播放控制（仅点击按钮播放）
- 坐标选择检查（执行前验证）
- LeaseLost错误处理
- 坐标组和实时观察的高亮动画
"""
import gradio as gr
from user_manager import user_manager
from config import RESTRICT_VIDEO_PLAYBACK, REFERENCE_VIEW_HEIGHT, LIVE_OBSERVATION_SCALE, ACTION_SCALE, CONTROL_SCALE, ENV_IDS, FONT_SIZE
from note_content import get_task_hint
from gradio_callbacks import (
    login_and_load_task,
    load_next_task_wrapper,
    on_map_click,
    on_option_select,
    execute_step,
    init_app,
    confirm_demo_watched,
    play_demo_video,
    select_env_id,
    switch_to_record_mode,
    switch_to_test_mode,
    back_to_landing_page,  # 回退到模式选择页面的函数
    show_task_hint,  # 【新增导入】任务提示延迟加载函数：根据当前session的env_id加载任务提示内容
    show_loading_info  # 【新增导入】显示加载环境提示信息的函数
)

SYNC_JS = """
(function() {
    // 不再自动播放视频，只有点击按钮才播放
    
    function setupVideoAutoplay(v) {
        // 确保muted属性始终为true
        v.muted = true;
        v.setAttribute('muted', 'true');
        v.autoplay = false; // 不自动播放，等待用户交互
        v.setAttribute('autoplay', 'false');
        v.loop = true;
        v.setAttribute('loop', 'true');
        v.controls = false; // 禁用用户控制
        v.setAttribute('controls', 'false');
        v.playsInline = true;
        v.setAttribute('playsinline', 'true');
        v.style.pointerEvents = 'none'; // 禁用所有鼠标交互
        
        // 阻止用户通过键盘控制视频（空格键暂停/播放等）
        v.addEventListener('keydown', (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, true);
        
        // 阻止右键菜单
        v.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, true);
        
        // 不再自动播放，只有点击按钮才播放
        
        // 使用MutationObserver监听muted属性的变化，确保它始终保持为true
        if (!v.dataset.mutedObserverAttached) {
            const mutedObserver = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'muted') {
                        if (!v.muted) {
                            v.muted = true;
                            v.setAttribute('muted', 'true');
                        }
                    }
                });
            });
            mutedObserver.observe(v, {
                attributes: true,
                attributeFilter: ['muted']
            });
            v.dataset.mutedObserverAttached = 'true';
        }
    }
    
    function ensureDemoVideoAutoplay() {
        const videoWrapper = document.getElementById('demo_video');
        if (!videoWrapper) return;
        
        const vids = videoWrapper.querySelectorAll('video');
        if (vids.length === 0) return;
        
        vids.forEach((v) => {
            if (!v.dataset.autoplaySetup) {
                setupVideoAutoplay(v);
                v.dataset.autoplaySetup = 'true';
            } else {
                if (!v.muted) {
                    v.muted = true;
                    v.setAttribute('muted', 'true');
                }
            }
        });
    }

    function findCoordsBox() {
        // 尝试多种选择器查找包含"please click the keypoint selection image"的textarea
        const selectors = [
            '#coords_box textarea',
            '[id*="coords_box"] textarea',
            'textarea[data-testid*="coords"]',
            'textarea'
        ];
        
        for (const selector of selectors) {
            const elements = document.querySelectorAll(selector);
            for (const el of elements) {
                const value = el.value || '';
                if (value.trim() === 'please click the keypoint selection image') {
                    return el;
                }
            }
        }
        return null;
    }
    
    function checkCoordsBeforeExecute() {
        const coordsBox = findCoordsBox();
        if (coordsBox) {
            const coordsValue = coordsBox.value || '';
            // 如果值是"please click the keypoint selection image", 说明需要坐标但用户没有点击
            if (coordsValue.trim() === 'please click the keypoint selection image') {
                alert('please click the keypoint selection image before execute!');
                return false; // 阻止执行
            }
        }
        return true;
    }
    
    // 为EXECUTE按钮添加坐标检查监听器
    function attachCoordsCheckToButton(btn) {
        if (!btn.dataset.coordsCheckAttached) {
            btn.addEventListener('click', function(e) {
                if (!checkCoordsBeforeExecute()) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    return false;
                }
            }, true);
            btn.dataset.coordsCheckAttached = 'true';
        }
    }
    
    // 播放视频的函数（只在点击按钮时调用）
    function playDemoVideo() {
        const videoWrapper = document.getElementById('demo_video');
        if (videoWrapper) {
            const vids = videoWrapper.querySelectorAll('video');
            vids.forEach(v => {
                // 确保视频设置正确
                v.muted = true;
                v.setAttribute('muted', 'true');
                v.loop = true;
                v.setAttribute('loop', 'true');
                
                // 尝试播放视频
                if (v.readyState >= 2) {
                    // 视频已加载，直接播放
                    const playPromise = v.play();
                    if (playPromise && playPromise.catch) {
                        playPromise.catch(() => {});
                    }
                } else {
                    // 如果视频还没准备好，等待加载完成后再播放
                    v.addEventListener('loadeddata', function() {
                        const playPromise = v.play();
                        if (playPromise && playPromise.catch) {
                            playPromise.catch(() => {});
                        }
                    }, { once: true });
                    // 也监听 canplay 事件作为备选
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
    
    // 监听播放视频按钮（唯一触发视频播放的方式）
    function initPlayVideoButtonListener() {
        function attachToPlayVideoButton() {
            const playBtn = document.getElementById('play_video_btn');
            if (playBtn && !playBtn.dataset.playVideoAttached) {
                playBtn.addEventListener('click', function(e) {
                    // 检查按钮是否可交互
                    if (playBtn.disabled || playBtn.hasAttribute('disabled')) {
                        return;
                    }
                    // 点击按钮后立即播放视频
                    playDemoVideo();
                });
                playBtn.dataset.playVideoAttached = 'true';
            }
        }
        
        // 使用MutationObserver等待Gradio加载完成
        const observer = new MutationObserver(function(mutations) {
            attachToPlayVideoButton();
        });
        
        // 开始观察
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // 立即执行一次
        setTimeout(attachToPlayVideoButton, 2000);
    }
    
    // 监听所有按钮点击, 找到EXECUTE按钮并添加检查
    function initExecuteButtonListener() {
        function attachToExecuteButtons() {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                const btnText = btn.textContent || btn.innerText || '';
                if (btnText.trim().includes('EXECUTE')) {
                    attachCoordsCheckToButton(btn);
                }
            }
        }
        
        // 使用MutationObserver等待Gradio加载完成
        const observer = new MutationObserver(function(mutations) {
            attachToExecuteButtons();
        });
        
        // 开始观察
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // 立即执行一次, 处理已经加载的按钮
        setTimeout(attachToExecuteButtons, 2000);
        // 不再自动播放视频，只有点击按钮才播放
    }
    
    // 监听 Gradio 错误, 捕获 LeaseLost 错误
    function initLeaseLostHandler() {
        // 监听全局错误事件
        window.addEventListener('error', function(e) {
            const errorMsg = e.message || e.error?.message || '';
            if (errorMsg.includes('LeaseLost') || errorMsg.includes('lease lost')) {
                e.preventDefault();
                alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                // 可选: 自动刷新页面
                // window.location.reload();
            }
        });
        
        // 监听 Gradio 的错误提示 (Gradio 使用 toast 显示错误)
        // 通过 MutationObserver 监听 DOM 变化, 查找错误消息
        const errorObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) { // Element node
                        // 查找包含错误消息的元素
                        const text = node.textContent || node.innerText || '';
                        if (text.includes('LeaseLost') || text.includes('lease lost') || 
                            text.includes('logged in elsewhere') || text.includes('no longer valid')) {
                            // 显示自定义弹窗
                            setTimeout(() => {
                                alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                            }, 100);
                        }
                    }
                });
            });
        });
        
        // 观察整个文档的变化
        errorObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // 拦截 fetch 请求, 检查响应中的错误
        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            return originalFetch.apply(this, args).then(function(response) {
                // 检查响应中是否包含 LeaseLost 错误
                if (response.ok) {
                    return response.clone().json().then(function(data) {
                        // Gradio API 返回的数据结构
                        if (data && typeof data === 'object') {
                            const dataStr = JSON.stringify(data);
                            if (dataStr.includes('LeaseLost') || dataStr.includes('lease lost')) {
                                setTimeout(() => {
                                    alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                                }, 100);
                            }
                        }
                        return response;
                    }).catch(function() {
                        return response;
                    });
                }
                return response;
            });
        };
    }
    
    // Apply blue flashing border to coords_group element and live_obs
    function applyCoordsGroupHighlight() {
        function removeHighlightStyles(group) {
            group.style.removeProperty('border');
            group.style.removeProperty('border-radius');
            group.style.removeProperty('padding');
            group.style.removeProperty('animation');
            group.classList.remove('coords-group-highlight');
            if (group.id === 'coords_group') {
                group.removeAttribute('id');
            }
        }
        
        function removeLiveObsHighlight(liveObs) {
            if (liveObs) {
                liveObs.style.removeProperty('border');
                liveObs.style.removeProperty('border-radius');
                liveObs.style.removeProperty('animation');
                liveObs.classList.remove('live-obs-highlight');
            }
        }
        
        function applyLiveObsHighlight(liveObs) {
            if (liveObs) {
                const computedStyle = window.getComputedStyle(liveObs);
                const needsStyle = (
                    computedStyle.borderColor !== 'rgb(59, 130, 246)' || 
                    computedStyle.borderWidth !== '3px' ||
                    computedStyle.animationName === 'none' ||
                    !computedStyle.animationName.includes('bluePulse')
                );
                
                if (needsStyle) {
                    liveObs.classList.add('live-obs-highlight');
                    liveObs.style.setProperty('border', '3px solid #3b82f6', 'important');
                    liveObs.style.setProperty('border-radius', '8px', 'important');
                    liveObs.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
                }
            }
        }
        
        function checkForCoordsGroup() {
            const coordsBox = document.querySelector('[id*="coords_box"]');
            const liveObs = document.querySelector('#live_obs');
            let targetGroup = null;
            let shouldHighlightLiveObs = false;
            
            if (coordsBox) {
                // Check if coords_box value indicates that point selection is complete
                const coordsTextarea = coordsBox.querySelector('textarea') || coordsBox;
                const coordsValue = (coordsTextarea.value || '').trim();
                const isCoordsSelected = coordsValue !== 'please click the keypoint selection image';
                
                // Find the direct parent group that contains coords_box
                let parentGroup = coordsBox.parentElement;
                while (parentGroup && !parentGroup.classList.contains('gr-group')) {
                    parentGroup = parentGroup.parentElement;
                }
                
                // If Execute button is in this group, find a more specific parent
                if (parentGroup && parentGroup.querySelector('[id*="exec_btn"]') !== null) {
                    const allGroups = document.querySelectorAll('.gr-group');
                    for (let group of allGroups) {
                        if (group.contains(coordsBox) && !group.querySelector('[id*="exec_btn"]')) {
                            parentGroup = group;
                            break;
                        }
                    }
                }
                
                if (parentGroup && !parentGroup.querySelector('[id*="exec_btn"]')) {
                    const computedStyle = window.getComputedStyle(parentGroup);
                    const isVisible = parentGroup.offsetParent !== null && computedStyle.display !== 'none';
                    const hasCSSAnimation = computedStyle.animationName !== 'none' && computedStyle.animationName.includes('bluePulse');
                    
                    // Don't highlight if:
                    // 1. coords_group is hidden (not visible)
                    // 2. coords are already selected (value is not "please click the keypoint selection image")
                    if (!isVisible || isCoordsSelected) {
                        // Remove highlight if it exists
                        const hasBluePulseAnimation = parentGroup.style.animation && parentGroup.style.animation.includes('bluePulse');
                        if (hasBluePulseAnimation || hasCSSAnimation) {
                            removeHighlightStyles(parentGroup);
                            // Also remove the class that triggers CSS
                            parentGroup.classList.remove('coords-group-highlight');
                        }
                        // Remove live_obs highlight
                        removeLiveObsHighlight(liveObs);
                    } else {
                        // Apply highlight only if visible and coords not selected
                        targetGroup = parentGroup;
                        shouldHighlightLiveObs = true;
                        if (!parentGroup.id || !parentGroup.id.includes('coords_group')) {
                            parentGroup.id = 'coords_group';
                        }
                        
                        // Add class to enable CSS highlight
                        parentGroup.classList.add('coords-group-highlight');
                        
                        const needsStyle = (
                            computedStyle.borderColor === 'rgb(0, 0, 0)' || 
                            computedStyle.borderWidth === '0px' || 
                            computedStyle.borderStyle === 'none' ||
                            computedStyle.animationName === 'none' ||
                            !computedStyle.animationName.includes('bluePulse')
                        );
                        
                        if (needsStyle) {
                            parentGroup.style.setProperty('border', '3px solid #3b82f6', 'important');
                            parentGroup.style.setProperty('border-radius', '8px', 'important');
                            parentGroup.style.setProperty('padding', '15px', 'important');
                            parentGroup.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
                        }
                    }
                }
            }
            
            // Apply or remove live_obs highlight based on shouldHighlightLiveObs
            if (shouldHighlightLiveObs && liveObs) {
                const liveObsComputedStyle = window.getComputedStyle(liveObs);
                const isLiveObsVisible = liveObs.offsetParent !== null && liveObsComputedStyle.display !== 'none';
                if (isLiveObsVisible) {
                    applyLiveObsHighlight(liveObs);
                } else {
                    removeLiveObsHighlight(liveObs);
                }
            } else {
                removeLiveObsHighlight(liveObs);
            }
            
            // Remove styles from groups that shouldn't have highlight
            const allGroups = document.querySelectorAll('.gr-group');
            allGroups.forEach(group => {
                if (group === targetGroup) return;
                const hasExecBtn = group.querySelector('[id*="exec_btn"]') !== null;
                const hasCoordsBox = group.querySelector('[id*="coords_box"]') !== null;
                const hasBluePulseAnimation = group.style.animation && group.style.animation.includes('bluePulse');
                const hasHighlightClass = group.classList.contains('coords-group-highlight');
                const computedStyle = window.getComputedStyle(group);
                const hasCSSAnimation = computedStyle.animationName !== 'none' && computedStyle.animationName.includes('bluePulse');
                
                if ((hasExecBtn || !hasCoordsBox) && (hasBluePulseAnimation || hasHighlightClass || hasCSSAnimation)) {
                    removeHighlightStyles(group);
                }
            });
        }
        
        setInterval(checkForCoordsGroup, 500);
    }
    
    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initExecuteButtonListener();
            initLeaseLostHandler();
            initPlayVideoButtonListener();
            setTimeout(() => {
                applyCoordsGroupHighlight();
            }, 2000);
        });
    } else {
        initExecuteButtonListener();
        initLeaseLostHandler();
        initPlayVideoButtonListener();
        setTimeout(() => {
            applyCoordsGroupHighlight();
        }, 2000);
    }
})();
"""

CSS = f"""/* 全局字体大小配置 - 统一应用到所有UI组件 */
/* 使用多个选择器确保覆盖所有Gradio元素 */
body, html {{
    font-size: {FONT_SIZE} !important;
}}
.gradio-container, #gradio-app {{
    font-size: {FONT_SIZE} !important;
}}
/* 直接对所有文本元素设置统一的字体大小 */
.gradio-container *, #gradio-app *, button, input, textarea, select, label, p, span, div, h1, h2, h3, h4, h5, h6, .gr-button, .gr-textbox, .gr-dropdown, .gr-radio, .gr-checkbox, .gr-markdown {{
    font-size: {FONT_SIZE} !important;
}}
/* 紧凑日志文本框也使用统一的字体大小 */
.compact-log textarea {{
    max-height: 120px !important;
    font-family: monospace;
    font-size: {FONT_SIZE} !important;
}}
#live_obs {{ }}
#control_panel {{ border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; background-color: #f9fafb; }}
.ref-zone {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 10px; }}
#combined_view_html {{ border: none !important; }}
#combined_view_html img {{ max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border: none !important; border-radius: 8px; object-fit: contain; }}
#demo_video {{ border: none !important; }}
#demo_video video {{ border: none !important; }}
/* Action Radio - 每个选项占满一行 */
#action_radio .form-radio {{
    display: block !important;
    width: 100% !important;
    margin-bottom: 8px !important;
}}
#action_radio .form-radio label {{
    width: 100% !important;
    display: block !important;
}}
#action_radio label {{
    display: block !important;
    width: 100% !important;
    margin-bottom: 8px !important;
}}
/* 【环境选择按钮颜色分类样式】
 * 为不同类别的环境按钮定义浅色背景和深色文字，通过颜色区分任务类型：
 * - Counting (计数): 浅蓝色背景 (#dbeafe)，深蓝色文字 (#1e40af)，蓝色边框
 * - Persistence (恒常): 浅绿色背景 (#dcfce7)，深绿色文字 (#166534)，绿色边框
 * - Reference (参考): 浅黄色背景 (#fef9c3)，深黄色文字 (#854d0e)，黄色边框
 * - Behavior (行为): 浅红色背景 (#fee2e2)，深红色文字 (#991b1b)，红色边框
 * 这些样式类通过 elem_classes 参数应用到对应的按钮上，实现视觉分类效果
 */
.btn-counting {{
    background-color: #dbeafe !important;  /* 浅蓝色背景 - Counting类别 */
    color: #1e40af !important;              /* 深蓝色文字 */
    border: 1px solid #bfdbfe !important;   /* 蓝色边框 */
}}
.btn-persistence {{
    background-color: #dcfce7 !important;  /* 浅绿色背景 - Persistence类别 */
    color: #166534 !important;              /* 深绿色文字 */
    border: 1px solid #bbf7d0 !important;   /* 绿色边框 */
}}
.btn-reference {{
    background-color: #fef9c3 !important;  /* 浅黄色背景 - Reference类别 */
    color: #854d0e !important;              /* 深黄色文字 */
    border: 1px solid #fde047 !important;   /* 黄色边框 */
}}
.btn-behavior {{
    background-color: #fee2e2 !important;  /* 浅红色背景 - Behavior类别 */
    color: #991b1b !important;              /* 深红色文字 */
    border: 1px solid #fecaca !important;   /* 红色边框 */
}}
/* Target coords_group by ID or by containing coords_box (but not exec_btn) - only when highlight class is present */
#coords_group.coords-group-highlight,
.gr-group.coords-group-highlight:has([id*="coords_box"]):not(:has([id*="exec_btn"])) {{
    border: 3px solid #3b82f6 !important;
    border-radius: 8px;
    padding: 15px;
    animation: bluePulse 1s ease-in-out infinite;
}}
/* Live Observation highlight - only when highlight class is present */
#live_obs.live-obs-highlight {{
    border: 3px solid #3b82f6 !important;
    border-radius: 8px;
    animation: bluePulse 1s ease-in-out infinite;
}}

@keyframes bluePulse {{
    0%, 100% {{ 
        border-color: #3b82f6;
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.8);
    }}
    50% {{ 
        border-color: #2563eb;
        box-shadow: 0 0 20px 8px rgba(59, 130, 246, 0.6);
    }}
}}
/* Next Task Button - 与 confirm 按钮一致（primary 样式） */
#next_task_btn:disabled,
#next_task_btn[disabled],
#next_task_btn.disabled {{
    opacity: 0.5 !important;
}}
/* Play Video Button - 禁用时变灰 */
#play_video_btn:disabled,
#play_video_btn[disabled],
#play_video_btn.disabled {{
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}}
/* Back to Mode Selection Button - 白色样式 */
#back_to_landing_btn {{
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #d1d5db !important;
}}
#back_to_landing_btn:hover {{
    background-color: #f9fafb !important;
    border-color: #9ca3af !important;
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
/* Operation Zone 提示文字样式 - 放在标题右侧 */
#operation_hint {{
    text-align: right !important;
    font-size: {FONT_SIZE} !important;
    padding: 0 !important;
    margin: 0 !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    flex: 1 !important;
}}"""
if RESTRICT_VIDEO_PLAYBACK:
    CSS += """
    #demo_video {
        pointer-events: none;
    }
    """


def create_ui_blocks():
    """
    创建并返回Gradio Blocks对象
    
    Returns:
        demo: Gradio Blocks对象
    """
    # 注意：在 Gradio 6.0+ 中，css 和 js 参数应该传递给 launch() 或 mount_gradio_app()，而不是 Blocks 构造函数
    with gr.Blocks(title="Oracle Planner Interface") as demo:
        with gr.Row():
            gr.Markdown("## HistoryBench Human Evaluation 🚀🚀🚀")
            operation_hint = gr.Markdown("Read the task goal, and select correct action in the Keypoint Selection🎯 panel and execute it to finish the task", elem_id="operation_hint")
        
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
        # - 用户点击 Record Mode / Test Mode 按钮时
        # - 用户选择环境 ID 时
        # - 用户点击 Next Task 按钮时
        # ============================================
        loading_overlay = gr.HTML(value="", elem_id="loading_overlay")
        
        # State
        uid_state = gr.State(value=None)
        username_state = gr.State(value="")
        
        # --- Loading Section (Visible initially) ---
        with gr.Group(visible=True) as loading_group:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        # --- Landing Section (For URL login choice) ---
        with gr.Group(visible=False) as landing_group:
            gr.Markdown("### Select Mode")
            gr.Markdown("Please choose how you want to proceed:")
            with gr.Row():
                test_btn = gr.Button("Free Try Mode! 🤔", variant="secondary")
                record_btn = gr.Button("Record Mode 📝", variant="primary")

        # --- Login Section ---
        with gr.Group(visible=False) as login_group:
            gr.Markdown("### User Login")
            with gr.Row():
                # Get available usernames from user_manager
                available_users = list(user_manager.user_tasks.keys())
                username_input = gr.Dropdown(
                    choices=available_users,
                    label="Username",
                    value=None
                )
                login_btn = gr.Button("Login", variant="primary")
            login_msg = gr.Markdown("")

        # --- Env ID Selection Section (for user_test only) ---
        # 【环境ID选择界面 - 2x2分类网格布局】
        # 此界面仅在测试模式（user_test）下显示，用于让用户选择要测试的环境ID
        # 布局从原来的简单列表改为2x2分类网格，按任务类型分为四个类别：
        # - Counting (计数): 需要计数操作的任务
        # - Persistence (恒常): 需要持久性/记忆操作的任务
        # - Reference (参考): 需要参考视频/演示的任务
        # - Behavior (行为): 需要特定行为模式的任务
        with gr.Group(visible=False) as env_selection_group:
            gr.Markdown("### Select Environment")
            gr.Markdown("Please select an environment ID to start the task")
            
            # 【按钮列表】存储所有环境按钮及其对应的环境ID，用于后续统一绑定点击事件
            # 格式: [(button对象, env_id字符串), ...]
            env_buttons_with_ids = []
            
            # 【创建环境按钮函数】
            # 功能：创建环境选择按钮并添加到列表中，同时应用对应的CSS样式类
            # 参数：
            #   - env_id: 环境ID字符串（如"BinFill"）
            #   - css_class: CSS样式类名（如"btn-counting"），用于设置按钮颜色
            # 返回：创建的按钮对象
            def create_env_btn(env_id, css_class=""):
                # 创建按钮，使用secondary变体，大尺寸，并应用CSS类
                btn = gr.Button(env_id, variant="secondary", size="lg", elem_classes=css_class)
                # 将按钮和环境ID的元组添加到列表中，供后续事件绑定使用
                env_buttons_with_ids.append((btn, env_id))
                return btn

            # 【第一行：Counting 和 Persistence 类别】
            with gr.Row():
                # 【左上：Counting (计数) 类别】
                # 包含4个环境：BinFill, PickXtimes, SwingXtimes, StopCube
                # 这些任务主要涉及计数操作
                with gr.Column():
                    gr.Markdown("#### 🔢 Counting")
                    # 第一行：BinFill 和 PickXtimes
                    with gr.Row():
                         create_env_btn("BinFill", "btn-counting")      # 浅蓝色按钮
                         create_env_btn("PickXtimes", "btn-counting")   # 浅蓝色按钮
                    # 第二行：SwingXtimes 和 StopCube
                    with gr.Row():
                         create_env_btn("SwingXtimes", "btn-counting")  # 浅蓝色按钮
                         create_env_btn("StopCube", "btn-counting")     # 浅蓝色按钮
                
                # 【右上：Persistence (恒常) 类别】
                # 包含4个环境：VideoUnmask, ButtonUnmask, VideoUnmaskSwap, ButtonUnmaskSwap
                # 这些任务主要涉及持久性/记忆操作
                with gr.Column():
                    gr.Markdown("#### 👁️ Persistence")
                    # 第一行：VideoUnmask 和 ButtonUnmask
                    with gr.Row():
                         create_env_btn("VideoUnmask", "btn-persistence")      # 浅绿色按钮
                         create_env_btn("ButtonUnmask", "btn-persistence")    # 浅绿色按钮
                    # 第二行：VideoUnmaskSwap 和 ButtonUnmaskSwap
                    with gr.Row():
                         create_env_btn("VideoUnmaskSwap", "btn-persistence")  # 浅绿色按钮
                         create_env_btn("ButtonUnmaskSwap", "btn-persistence") # 浅绿色按钮

            # 【第二行：Reference 和 Behavior 类别】
            with gr.Row():
                # 【左下：Reference (参考) 类别】
                # 包含4个环境：PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder
                # 这些任务主要涉及参考视频/演示
                with gr.Column():
                    gr.Markdown("#### 🖼️ Reference")
                    # 第一行：PickHighlight 和 VideoRepick
                    with gr.Row():
                         create_env_btn("PickHighlight", "btn-reference")      # 浅黄色按钮
                         create_env_btn("VideoRepick", "btn-reference")      # 浅黄色按钮
                    # 第二行：VideoPlaceButton 和 VideoPlaceOrder
                    with gr.Row():
                         create_env_btn("VideoPlaceButton", "btn-reference") # 浅黄色按钮
                         create_env_btn("VideoPlaceOrder", "btn-reference")  # 浅黄色按钮

                # 【右下：Behavior (行为) 类别】
                # 包含4个环境：MoveCube, InsertPeg, PatternLock, RouteStick
                # 这些任务主要涉及特定的行为模式
                with gr.Column():
                    gr.Markdown("#### 🖐️ Behavior")
                    # 第一行：MoveCube 和 InsertPeg
                    with gr.Row():
                         create_env_btn("MoveCube", "btn-behavior")    # 浅红色按钮
                         create_env_btn("InsertPeg", "btn-behavior")  # 浅红色按钮
                    # 第二行：PatternLock 和 RouteStick
                    with gr.Row():
                         create_env_btn("PatternLock", "btn-behavior") # 浅红色按钮
                         create_env_btn("RouteStick", "btn-behavior")  # 浅红色按钮

            # 环境选择消息显示区域（用于显示选择结果或错误信息）
            env_selection_msg = gr.Markdown("")

        # --- Main Interface (Hidden initially) ---
        with gr.Group(visible=False) as main_interface:
            # --- Top Container: Reference Zone (35-40% Height) ---
            with gr.Row(elem_classes="ref-zone"):
                # Left: Text Info (30%)
                with gr.Column(scale=3):
                    with gr.Group():
                         gr.Markdown("### 1. Progress Tracker 📊🥳")
                         with gr.Row():
                             task_info_box = gr.Textbox(label="Current Task", interactive=False, show_label=False, scale=2)
                             progress_info_box = gr.Textbox(label="Progress", interactive=False, show_label=False, scale=1)
                    
                    with gr.Group():
                         gr.Markdown("### 2. Task Goal 🏅")
                         goal_box = gr.Textbox(label="Instruction", lines=3, interactive=False, show_label=False)
                    
                    with gr.Group():
                         gr.Markdown("### 3. System Log 📝")
                         log_output = gr.Textbox(label="Log", lines=6, interactive=False, elem_classes="compact-log", show_label=False)

                # Right: Reference Views (70%)
                with gr.Column(scale=7):
                     # Demo Video Group (第一阶段显示)
                     with gr.Group(visible=True) as demo_video_group:
                         # Title row
                         with gr.Row():
                             with gr.Column(scale=1):
                                 gr.Markdown("### Watch video and remember robot actions 👀✍️")
                         
                         # Content row - 单列布局，Hint 在视频上方
                         with gr.Column(scale=1):
                             # 【修改】任务提示延迟加载和切换显示功能（演示视频组）：
                             # 移除 Accordion，直接显示 Task Hint 按钮
                             # 点击按钮后切换显示/隐藏提示内容
                             # Task Hint 按钮
                             show_hint_btn_demo = gr.Button("Task Hint 💡⬇️", size="sm")
                             # 任务提示Markdown组件：初始值为空，点击按钮后切换显示/隐藏
                             note2_demo = gr.Markdown(
                                 value="",  # 初始值为空，延迟加载
                                 elem_id="note2_demo"
                             )
                             # 事件绑定：将"Task Hint"按钮的点击事件绑定到 show_task_hint 函数
                             # 输入：uid_state（用户会话ID）和 note2_demo（当前提示内容，用于切换）
                             # 输出：note2_demo（任务提示Markdown组件），显示或隐藏提示内容
                             show_hint_btn_demo.click(fn=show_task_hint, inputs=[uid_state, note2_demo], outputs=[note2_demo])
                             
                             video_elem_id = "demo_video" if RESTRICT_VIDEO_PLAYBACK else None
                             video_autoplay = False  # 不自动播放，等待用户点击按钮
                             
                             video_display = gr.Video(
                                label="Demonstration Video", 
                                interactive=False,  # 禁用用户控制
                                height=300, 
                                elem_id=video_elem_id, 
                                autoplay=video_autoplay,
                                show_label=False,
                                visible=True
                             )
                             
                             play_video_btn = gr.Button("Start Demonstration Video🎬", variant="primary", size="lg", visible=True, interactive=True, elem_id="play_video_btn")
                             
                             confirm_demo_btn = gr.Button("Start Task", variant="secondary", size="lg", visible=True, interactive=False)

                     # Combined View Group (第一阶段隐藏)
                     with gr.Group(visible=False) as combined_view_group:
                         # Title row - all titles at the same height
                         with gr.Row():
                             with gr.Column(scale=1):
                                 gr.Markdown("### Execution LiveStream 🦾")
                         
                         # Content row - 单列布局，Hint 在视频上方
                         with gr.Column(scale=1):
                            # 【修改】任务提示延迟加载和切换显示功能（执行阶段组合视图组）：
                            # 移除 Accordion，直接显示 Task Hint 按钮
                            # 点击按钮后切换显示/隐藏提示内容
                            # Task Hint 按钮
                            show_hint_btn = gr.Button("Task Hint 💡⬇️", size="sm")
                            # 任务提示Markdown组件：初始值为空，点击按钮后切换显示/隐藏
                            note2 = gr.Markdown(
                                value="",  # 初始值为空，延迟加载
                                elem_id="note2"
                            )
                            # 事件绑定：将"Task Hint"按钮的点击事件绑定到 show_task_hint 函数
                            # 输入：uid_state（用户会话ID）和 note2（当前提示内容，用于切换）
                            # 输出：note2（任务提示Markdown组件），显示或隐藏提示内容
                            show_hint_btn.click(fn=show_task_hint, inputs=[uid_state, note2], outputs=[note2])
                            
                            # Main: Desk + Robot View (Combined) - 使用 HTML 组件显示 MJPEG 流
                            combined_display = gr.HTML(
                                value="<div id='combined_view_html'><p>Waiting for video stream...</p></div>",
                                elem_id="combined_view_html"
                            )

            # --- Bottom Container: Operation Zone (60-65% Height) ---
            # Operation Zone Group (第一阶段隐藏)
            with gr.Group(visible=False) as operation_zone_group:
                with gr.Row():
                    # Left: Action Selection
                    with gr.Column(scale=ACTION_SCALE):
                         gr.Markdown("### Action Selection 🧠")
                         options_radio = gr.Radio(choices=[], label="Action", type="value", show_label=False, elem_id="action_radio")

                    # Middle: Live Observation (Main)
                    with gr.Column(scale=LIVE_OBSERVATION_SCALE):
                         gr.Markdown("### Keypoint Selection 🎯")
                         img_display = gr.Image(
                            label="Live Observation", 
                            interactive=False, 
                            type="pil", 
                            elem_id="live_obs",
                            show_label=False
                         )

                    # Right: Control Panel
                    with gr.Column(scale=CONTROL_SCALE):
                         gr.Markdown("### Control Panel 🎛️")
                         
                         # Coords Group (conditionally visible)
                         with gr.Group(visible=False, elem_id="coords_group") as coords_group:
                             gr.Markdown("**Coords** 📍")
                             coords_box = gr.Textbox(label="Coords", value="", interactive=False, show_label=False, elem_id="coords_box")
                         
                         exec_btn = gr.Button("EXECUTE 🤖", variant="stop", size="lg", elem_id="exec_btn")
                         
                         next_task_btn = gr.Button("Next Task 🔄", variant="primary", interactive=False, elem_id="next_task_btn")
                         
                         # 回退到模式选择页面按钮：允许用户从执行界面返回到模式选择页面，重新选择测试模式或录制模式
                         back_to_landing_btn = gr.Button("Back to Mode Selection 🔙", variant="secondary", interactive=True, elem_id="back_to_landing_btn")

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
                env_selection_group,
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                confirm_demo_btn,
                play_video_btn,
                coords_group,
                note2,
                note2_demo,
                loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
            ]
        ).then(
            fn=lambda u: u,
            inputs=[username_input],
            outputs=[username_state]
        )
        
        # ============================================
        # 1.1 Landing Page - Record Mode 按钮事件绑定
        # ============================================
        # 功能说明：
        # - 用户在模式选择页面点击 "Record Mode" 按钮时触发
        # - 首先显示全屏加载遮罩层
        # - 然后切换到录制模式并执行登录和任务加载
        # - 加载完成后，自动隐藏遮罩层
        # 
        # 事件链流程：
        # 1. click 事件：调用 show_loading_info() 显示遮罩层
        # 2. .then() 链：调用 switch_to_record_mode() 切换到录制模式并加载任务
        # 3. switch_to_record_mode() 返回时，loading_overlay 被设置为空字符串，遮罩层消失
        # ============================================
        record_btn.click(
            # 第一步：显示加载遮罩层
            fn=show_loading_info,
            outputs=[loading_overlay]  # 输出到 loading_overlay 组件，显示遮罩层
        ).then(
            # 第二步：切换到录制模式并加载任务
            fn=switch_to_record_mode,
            inputs=[username_state, uid_state],  # 输入：用户名状态和会话ID
            outputs=[
                uid_state, 
                landing_group, 
                login_group, 
                env_selection_group,
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                confirm_demo_btn,
                play_video_btn,
                coords_group,
                note2,
                note2_demo,
                loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
            ]
        )
        
        # ============================================
        # 1.2 Landing Page - Test Mode 按钮事件绑定
        # ============================================
        # 功能说明：
        # - 用户在模式选择页面点击 "Free Try Mode" 按钮时触发
        # - 首先显示全屏加载遮罩层
        # - 然后切换到测试模式（用户名后添加 _test 后缀）并执行登录和任务加载
        # - 加载完成后，自动隐藏遮罩层
        # 
        # 事件链流程：
        # 1. click 事件：调用 show_loading_info() 显示遮罩层
        # 2. .then() 链：调用 switch_to_test_mode() 切换到测试模式并加载任务
        # 3. switch_to_test_mode() 返回时，loading_overlay 被设置为空字符串，遮罩层消失
        # ============================================
        test_btn.click(
            # 第一步：显示加载遮罩层
            fn=show_loading_info,
            outputs=[loading_overlay]  # 输出到 loading_overlay 组件，显示遮罩层
        ).then(
            # 第二步：切换到测试模式并加载任务
            fn=switch_to_test_mode,
            inputs=[username_state, uid_state],  # 输入：用户名状态和会话ID
            outputs=[
                uid_state, 
                landing_group, 
                login_group, 
                env_selection_group,
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                username_state,  # ADDED THIS: 更新前端用户名状态 / Update frontend username state
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                confirm_demo_btn,
                play_video_btn,
                coords_group,
                note2,
                note2_demo,
                loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
            ]
        )
        
        # ============================================
        # 1.3 环境 ID 选择按钮事件绑定（针对 user_test 用户）
        # ============================================
        # 功能说明：
        # - 在测试模式下，用户需要从环境选择界面选择一个环境 ID
        # - 为所有环境选择按钮（如 PickXtimes、VideoPlaceOrder 等）绑定点击事件
        # - 用户点击环境按钮时，首先显示全屏加载遮罩层
        # - 然后加载对应的环境任务
        # - 加载完成后，自动隐藏遮罩层
        # 
        # 事件链流程：
        # 1. click 事件：调用 show_loading_info() 显示遮罩层
        # 2. .then() 链：调用 select_env_id() 加载选定的环境任务
        # 3. select_env_id() 返回时，loading_overlay 被设置为空字符串，遮罩层消失
        # 
        # 技术说明：
        # - 使用 env_buttons_with_ids 列表，每个元素包含 (按钮对象, 环境ID) 元组
        # - 使用 lambda 函数捕获当前循环的 env_id，确保每个按钮调用时传入正确的环境ID
        # - 这样避免了依赖 ENV_IDS 的顺序，使代码更加健壮和清晰
        # ============================================
        for btn, env_id in env_buttons_with_ids:
            # 为每个环境选择按钮绑定点击事件
            # 使用 lambda 函数捕获当前循环的 env_id，确保每个按钮调用时传入正确的环境ID
            btn.click(
                # 第一步：显示加载遮罩层
                fn=show_loading_info,
                outputs=[loading_overlay]  # 输出到 loading_overlay 组件，显示遮罩层
            ).then(
                # 第二步：加载选定的环境任务
                # 使用 lambda 函数捕获当前循环的 env_id，确保每个按钮调用时传入正确的环境ID
                fn=lambda u, uid, eid=env_id: select_env_id(u, uid, eid),
                inputs=[username_state, uid_state],  # 输入：用户名状态和会话ID
                outputs=[
                    uid_state,
                    login_group,
                    env_selection_group,
                    main_interface,
                    login_msg,
                    img_display,
                    log_output,
                    options_radio,
                    goal_box,
                    coords_box,
                    combined_display,
                    video_display,
                    task_info_box,
                    progress_info_box,
                    login_btn,
                    next_task_btn,
                    exec_btn,
                    demo_video_group,
                    combined_view_group,
                    operation_zone_group,
                    confirm_demo_btn,
                    play_video_btn,
                    coords_group,
                    note2,
                    note2_demo,
                    loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
                ]
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
                env_selection_group,
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                confirm_demo_btn,
                play_video_btn,
                coords_group,
                note2,
                note2_demo,
                loading_overlay  # 【关键】加载完成后返回空字符串，清空 overlay 组件内容，遮罩层自动隐藏
            ]
        )
        
        # 1.5.1 回退到模式选择页面
        # 绑定回退按钮的点击事件：当用户点击"返回模式选择"按钮时，调用 back_to_landing_page 函数
        # 该函数会提取原始用户名（去掉 _test 后缀），并将界面重置为模式选择页面
        back_to_landing_btn.click(
            fn=back_to_landing_page,  # 回调函数：回退到模式选择页面
            inputs=[username_state, uid_state],  # 输入：当前用户名状态和会话唯一标识符
            outputs=[
                uid_state,              # 会话唯一标识符
                loading_group,          # 加载组
                landing_group,          # 模式选择组（关键：显示模式选择页面）
                login_group,           # 登录组
                env_selection_group,    # 环境选择组
                main_interface,        # 主界面
                login_msg,             # 登录消息
                img_display,           # 图片显示
                log_output,            # 日志输出
                options_radio,         # 选项单选
                goal_box,              # 目标框
                coords_box,            # 坐标框
                combined_display,      # 组合视图
                video_display,         # 视频显示
                task_info_box,         # 任务信息
                progress_info_box,     # 进度信息
                login_btn,             # 登录按钮
                next_task_btn,         # 下一个任务按钮
                exec_btn,              # 执行按钮
                username_state,        # 用户名状态（更新为原始用户名）
                demo_video_group,      # 演示视频组
                combined_view_group,   # 组合视图组
                operation_zone_group,  # 操作区域组
                confirm_demo_btn,      # 确认演示按钮
                play_video_btn,        # 播放视频按钮
                coords_group,          # 坐标组
                note2,                 # 提示信息
                note2_demo            # 演示提示信息
            ]
        )
        
        # 1.5 Play Demo Video
        play_video_btn.click(
            fn=play_demo_video,
            inputs=[play_video_btn],
            outputs=[play_video_btn, confirm_demo_btn]
        )
        
        # 1.6 Confirm Demo Watched
        confirm_demo_btn.click(
            fn=confirm_demo_watched,
            inputs=[uid_state, username_state],
            outputs=[
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                confirm_demo_btn,
                play_video_btn,
                exec_btn,
                coords_group,
                note2,
                note2_demo
            ]
        )

        # 2. Image Click
        img_display.select(
            fn=on_map_click,
            inputs=[uid_state, username_state, options_radio],
            outputs=[img_display, coords_box]
        )
        
        # 2.5. Option Select
        options_radio.change(
            fn=on_option_select,
            inputs=[uid_state, username_state, options_radio],
            outputs=[coords_box, img_display, coords_group]
        )

        # 3. Execute
        exec_btn.click(
            fn=execute_step,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[img_display, log_output, task_info_box, progress_info_box, next_task_btn, exec_btn, coords_group]
        )
        
        # 5. Auto Login on Load
        demo.load(
            fn=init_app,
            inputs=[],
            outputs=[
                uid_state,
                loading_group,
                landing_group,
                login_group,
                env_selection_group,
                main_interface, 
                login_msg, 
                img_display, 
                log_output, 
                options_radio, 
                goal_box, 
                coords_box, 
                combined_display, 
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                username_state,
                demo_video_group,
                combined_view_group,
                operation_zone_group,
                confirm_demo_btn,
                play_video_btn,
                coords_group,
                note2,
                note2_demo
            ]
        )
    
    return demo
