"""
UI布局模块
定义Gradio界面组件、CSS和JS
"""
import gradio as gr
from user_manager import user_manager
from config import RESTRICT_VIDEO_PLAYBACK
from gradio_callbacks import (
    login_and_load_task,
    load_next_task_wrapper,
    on_map_click,
    on_option_select,
    execute_step,
    init_app,
    confirm_demo_watched
)

# --- JS for Video (no sync needed for single video) ---
SYNC_JS = """
(function() {
    function findCoordsBox() {
        // 尝试多种选择器查找包含"please click the image"的textarea
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
                if (value.trim() === 'please click the image') {
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
            // 如果值是"please click the image"，说明需要坐标但用户没有点击
            if (coordsValue.trim() === 'please click the image') {
                alert('please click the image before execute!');
                return false; // 阻止执行
            }
        }
        return true;
    }
    
    // 监听所有按钮点击，找到EXECUTE按钮并添加检查
    function initExecuteButtonListener() {
        // 使用MutationObserver等待Gradio加载完成
        const observer = new MutationObserver(function(mutations) {
            // 查找所有按钮，找到包含"EXECUTE"文本的按钮
            const buttons = document.querySelectorAll('button');
            
            for (const btn of buttons) {
                const btnText = btn.textContent || btn.innerText || '';
                if (btnText.trim().includes('EXECUTE') && !btn.dataset.coordsCheckAttached) {
                    btn.addEventListener('click', function(e) {
                        if (!checkCoordsBeforeExecute()) {
                            e.preventDefault();
                            e.stopPropagation();
                            e.stopImmediatePropagation();
                            return false;
                        }
                    }, true); // 使用捕获阶段，确保在其他处理之前执行
                    btn.dataset.coordsCheckAttached = 'true';
                }
            }
        });
        
        // 开始观察
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // 立即执行一次，处理已经加载的按钮
        setTimeout(() => {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                const btnText = btn.textContent || btn.innerText || '';
                if (btnText.trim().includes('EXECUTE') && !btn.dataset.coordsCheckAttached) {
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
        }, 2000);
    }
    
    // 监听 Gradio 错误，捕获 LeaseLost 错误
    function initLeaseLostHandler() {
        // 监听全局错误事件
        window.addEventListener('error', function(e) {
            const errorMsg = e.message || e.error?.message || '';
            if (errorMsg.includes('LeaseLost') || errorMsg.includes('lease lost')) {
                e.preventDefault();
                alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
                // 可选：自动刷新页面
                // window.location.reload();
            }
        });
        
        // 监听 Gradio 的错误提示（Gradio 使用 toast 显示错误）
        // 通过 MutationObserver 监听 DOM 变化，查找错误消息
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
        
        // 拦截 fetch 请求，检查响应中的错误
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
    
    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initExecuteButtonListener();
            initLeaseLostHandler();
        });
    } else {
        initExecuteButtonListener();
        initLeaseLostHandler();
    }
})();
"""

CSS = """
#live_obs { border: 4px solid #3b82f6; border-radius: 8px; }
#control_panel { border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; background-color: #f9fafb; }
.compact-log textarea { max-height: 120px !important; font-family: monospace; font-size: 0.85em; }
.ref-zone { border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 10px; }
#combined_view_html img {{ max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border: 2px solid #3b82f6; border-radius: 8px; object-fit: contain; }}
"""
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
        gr.Markdown("## HistoryBench Oracle Planner Interface (v2)")
        
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
                         gr.Markdown("### 1. Task Info")
                         with gr.Row():
                             task_info_box = gr.Textbox(label="Current Task", interactive=False, show_label=False, scale=2)
                             progress_info_box = gr.Textbox(label="Progress", interactive=False, show_label=False, scale=1)
                    
                    with gr.Group():
                         gr.Markdown("### 2. Task Goal")
                         goal_box = gr.Textbox(label="Instruction", lines=3, interactive=False, show_label=False)
                    
                    with gr.Group():
                         gr.Markdown("### 3. System Log")
                         log_output = gr.Textbox(label="Log", lines=6, interactive=False, elem_classes="compact-log", show_label=False)

                # Right: Reference Views (70%)
                with gr.Column(scale=7):
                     gr.Markdown("### Reference Views")
                     
                     # Demo Video Group (第一阶段显示)
                     with gr.Group(visible=True) as demo_video_group:
                         video_elem_id = "demo_video" if RESTRICT_VIDEO_PLAYBACK else None
                         video_autoplay = True if RESTRICT_VIDEO_PLAYBACK else False
                         
                         video_display = gr.Video(
                            label="Demonstration (示范)", 
                            interactive=False, 
                            height=300, 
                            elem_id=video_elem_id, 
                            autoplay=video_autoplay
                         )
                         
                         confirm_demo_btn = gr.Button("Confirm - Start Task", variant="primary", size="lg", visible=True, interactive=True)
                     
                     # Combined View Group (第一阶段隐藏)
                     with gr.Group(visible=False) as combined_view_group:
                         # Desk + Robot View (Combined) - 使用 HTML 组件显示 MJPEG 流
                         combined_display = gr.HTML(
                            value="<div id='combined_view_html'><p>等待视频流...</p></div>",
                            elem_id="combined_view_html"
                         )

            # --- Bottom Container: Operation Zone (60-65% Height) ---
            # Operation Zone Group (第一阶段隐藏)
            with gr.Group(visible=False) as operation_zone_group:
                with gr.Row():
                    # Left: Live Observation (Main)
                    with gr.Column(scale=1):
                         gr.Markdown("### Live Observation (交互主视图)")
                         img_display = gr.Image(
                            label="Live Observation", 
                            interactive=False, 
                            type="pil", 
                            elem_id="live_obs",
                            show_label=False
                         )

                    # Right: Control Panel
                    with gr.Column(scale=2):
                         gr.Markdown("### Control Panel")
                         
                         with gr.Group(elem_id="control_panel"):
                             gr.Markdown("**1. Action**")
                             options_radio = gr.Radio(choices=[], label="Action", type="value", show_label=False)
                             
                             gr.Markdown("**2. Coords**")
                             coords_box = gr.Textbox(label="Coords", value="", interactive=False, show_label=False, elem_id="coords_box")
                             
                             gr.Markdown("**3. Execute**")
                             exec_btn = gr.Button("EXECUTE", variant="stop", size="lg", elem_id="exec_btn")
                             
                             gr.Markdown("---")
                             next_task_btn = gr.Button("Next Task", variant="secondary", interactive=False)

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
                confirm_demo_btn
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
                confirm_demo_btn
            ]
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
                exec_btn
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
            outputs=[coords_box, img_display]
        )

        # 3. Execute
        exec_btn.click(
            fn=execute_step,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[img_display, log_output, task_info_box, progress_info_box, next_task_btn, exec_btn]
        )
        
        # 5. Auto Login on Load (Timer 已移除，使用 MJPEG 流式传输)
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
                confirm_demo_btn
            ]
        )
    
    return demo
