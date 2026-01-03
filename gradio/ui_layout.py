"""
UI布局模块
定义Gradio界面组件、CSS和JS
"""
import gradio as gr
from user_manager import user_manager
from config import RESTRICT_VIDEO_PLAYBACK, REFERENCE_VIEW_HEIGHT, LIVE_OBSERVATION_SCALE, ACTION_SCALE, CONTROL_SCALE
from note_content import get_task_hint
from gradio_callbacks import (
    login_and_load_task,
    load_next_task_wrapper,
    on_map_click,
    on_option_select,
    execute_step,
    init_app,
    confirm_demo_watched,
    play_demo_video
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

CSS = f"""#live_obs {{ }}
#control_panel {{ border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; background-color: #f9fafb; }}
.compact-log textarea {{ max-height: 120px !important; font-family: monospace; font-size: 0.85em; }}
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
        gr.Markdown("## HistoryBench Human Evaluation 🚀🚀🚀")
        
        # State
        uid_state = gr.State(value=None)
        username_state = gr.State(value="")
        
        # --- Loading Section (Visible initially) ---
        with gr.Group(visible=True) as loading_group:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

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
                             # Task Hint (Accordion) - 默认收起
                             with gr.Accordion("Task Hint 💡 (点击展开查看提示)", open=False):
                                 note2_demo = gr.Markdown(
                                     value=get_task_hint(""),
                                     elem_id="note2_demo"
                                 )
                             
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
                             # Task Hint (Accordion) - 默认收起
                             with gr.Accordion("Task Hint 💡 (点击展开查看提示)", open=False):
                                 note2 = gr.Markdown(
                                     value=get_task_hint(""),
                                     elem_id="note2"
                                 )
                             
                             # Main: Desk + Robot View (Combined) - 使用 HTML 组件显示 MJPEG 流
                             combined_display = gr.HTML(
                                value="<div id='combined_view_html'><p>等待视频流...</p></div>",
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
                         
                         gr.Markdown("---")
                         exec_btn = gr.Button("EXECUTE 🤖", variant="stop", size="lg", elem_id="exec_btn")
                         
                         gr.Markdown("---")
                         next_task_btn = gr.Button("Next Task 🔄", variant="primary", interactive=False, elem_id="next_task_btn")

        # --- Event Wiring ---

        # 1. Login
        login_btn.click(
            fn=login_and_load_task,
            inputs=[username_input, uid_state],
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
                note2_demo
            ]
        ).then(
            fn=lambda u: u,
            inputs=[username_input],
            outputs=[username_state]
        )
        
        # 1.5 Next Task
        next_task_btn.click(
            fn=load_next_task_wrapper,
            inputs=[username_state, uid_state],
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
                note2_demo
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
