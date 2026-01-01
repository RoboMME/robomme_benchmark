import gradio as gr
import cv2
import json
import numpy as np
import os
import sys

class AutoSyncTagger:
    def __init__(self, video_path, output_json_path, output_video_path, target_width=960):
        self.video_path = video_path
        self.output_json_path = output_json_path
        self.output_video_path = output_video_path
        self.target_width = target_width
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 缩放逻辑
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_ratio = 1.0
        if self.orig_w > self.target_width:
            self.scale_ratio = self.target_width / self.orig_w
        self.resize_dims = (int(self.orig_w * self.scale_ratio), int(self.orig_h * self.scale_ratio))
        
        self.keyframes = {}  # Changed from set to dict: {frame_idx: {"option": opt, "point": [x, y]}}
        self.frames_cache = []
        self.preload_video()

    def preload_video(self):
        print(f"正在预加载 {self.total_frames} 帧 (FPS: {self.fps:.2f})...")
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            if self.scale_ratio != 1.0:
                frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_AREA)
            self.frames_cache.append(frame) 
            cnt += 1
            if cnt % 100 == 0:
                sys.stdout.write(f"\r加载: {cnt}/{self.total_frames}")
                sys.stdout.flush()
        print("\n预加载完成！")
        self.cap.release()

    def _draw_overlay(self, frame_bgr, frame_idx):
        """辅助函数：在帧上绘制标记（原地修改）"""
        if frame_idx in self.keyframes:
            data = self.keyframes[frame_idx]
            h, w, _ = frame_bgr.shape
            thick = max(2, int(5 * self.scale_ratio))
            
            # 蓝色框
            cv2.rectangle(frame_bgr, (0, 0), (w, h), (255, 0, 0), thick * 2)
            
            # 显示关键帧信息
            info_text = f"KEY: {frame_idx} | {data.get('option', '')}"
            cv2.putText(frame_bgr, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0 * self.scale_ratio, (255, 0, 0), thick)
            
            # 绘制点击点
            if "point" in data:
                # 原始坐标 -> 显示坐标
                orig_pt = data["point"]
                disp_x = int(orig_pt[0] * self.scale_ratio)
                disp_y = int(orig_pt[1] * self.scale_ratio)
                cv2.circle(frame_bgr, (disp_x, disp_y), 10, (0, 0, 255), -1)  # 实心红点
                
        return frame_bgr

    def get_frame_image(self, frame_idx):
        idx = int(frame_idx)
        if idx < 0: idx = 0
        if idx >= len(self.frames_cache): idx = len(self.frames_cache) - 1
        
        frame_bgr = self.frames_cache[idx].copy()
        frame_bgr = self._draw_overlay(frame_bgr, idx)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # --- 新增：时间格式化辅助函数 ---
    def _format_time(self, seconds):
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"

    # --- 新增：生成时间线文本 ---
    def get_timeline_text(self):
        sorted_keys = sorted(list(self.keyframes.keys()))
        if not sorted_keys:
            return "暂无标记"
        
        lines = []
        lines.append(f"总计标记: {len(sorted_keys)} 帧")
        lines.append("-" * 30)
        for k in sorted_keys:
            t = k / self.fps
            time_str = self._format_time(t)
            data = self.keyframes[k]
            opt = data.get("option", "N/A")
            pt = data.get("point", [])
            pt_str = f"({pt[0]}, {pt[1]})" if pt else "No Point"
            lines.append(f"Frame {k:<5} | Time {time_str} | {opt} | {pt_str}")
        return "\n".join(lines)

    # --- 修改：返回值增加 timeline_text ---
    def toggle_keyframe(self, frame_idx, option_val, point_str):
        idx = int(frame_idx)
        msg = ""
        
        # 解析坐标
        point = []
        try:
            # 格式预期 "X: 123, Y: 456"
            parts = point_str.replace("X:", "").replace("Y:", "").split(",")
            if len(parts) == 2:
                disp_x = int(parts[0].strip())
                disp_y = int(parts[1].strip())
                # 转换回原始坐标
                orig_x = int(disp_x / self.scale_ratio)
                orig_y = int(disp_y / self.scale_ratio)
                point = [orig_x, orig_y]
        except:
            pass
            
        if idx in self.keyframes:
            # 如果只想更新 Option 或 Point，这里其实可以直接覆盖。
            # 但为了保持 Toggle 语义，如果完全一样则删除，否则更新？
            # 简化逻辑：始终覆盖更新，除非用户明确想删除（可以加个删除按钮，但这里Toggle沿用旧逻辑：如果已存在则删除，
            # 不过现在带了数据，直接删除可能误操作。
            # 鉴于需求描述： "取消 (Remove)： 如果当前帧已经是关键帧，点击按钮则删除该条记录。 更新 (Update)： 如果用户只想修改坐标或 Option，重新点击标记即可覆盖旧数据。"
            # 这两个逻辑冲突。Toggle 意味着状态反转。
            # 这种情况下，我们假设如果 Option 和 Point 都没变，或者是空操作，则删除？
            # 为了简单好用，建议：如果已存在，则**覆盖更新**。增加一个单独的"删除当前帧"按钮可能是更好的设计，但遵循当前计划：
            # 让我们微调逻辑：如果 frame_idx 存在，检查是否完全一致。
            # 或者，更符合直觉的是：如果按了标记，就是为了标记。如果要删除，再按一次？
            # 按照用户需求： "标记 (Add)... 取消 (Remove)... 更新 (Update)..."
            # 我们可以判断：如果 frame_idx 存在，则**更新**。
            # 那怎么删除？ 可以在 UI 增加一个 Remove 按钮。
            # 或者：如果当前 Option/Point 与已存储的一致，则删除；否则更新。
            
            # 这里我采用：总是更新。因为用户可能想微调坐标。
            # 为了支持删除，可以在 UI 增加一个 'Remove Keyframe' 按钮。
            # 或者遵循原始 Toggle：存在即删除。
            # 但这样就没法更新了。
            # 让我们折中：如果 keyframes 中有，询问用户？ 不行。
            # 让我们修改 UI 计划：添加一个单独的 "删除" 按钮。
            # 暂时先实现：总是覆盖更新。如果想删除，可以在 UI 上加个逻辑。
            # 为了符合 "Toggle" 的名字，这里还是保留删除逻辑吗？
            # 用户明确说： "更新 (Update)： 如果用户只想修改坐标或 Option，重新点击标记即可覆盖旧数据。"
            # 这意味着不能简单的 Toggle。
            # 既然如此，我把这个函数改名为 add_or_update_keyframe，并移除删除逻辑。
            # 删除逻辑单独处理。
            
            # 但为了不破坏现有绑定，我先保留 toggle 名字，逻辑改为：
            # 总是覆盖。
            self.keyframes[idx] = {"option": option_val, "point": point}
            msg = f"帧 {idx} 已更新: {option_val}"
        else:
            self.keyframes[idx] = {"option": option_val, "point": point}
            msg = f"帧 {idx} 已标记: {option_val}"
        
        # 返回：当前图像，状态消息，更新后的时间线列表
        return self.get_frame_image(idx), msg, self.get_timeline_text()

    def remove_keyframe(self, frame_idx):
        idx = int(frame_idx)
        if idx in self.keyframes:
            del self.keyframes[idx]
            msg = f"帧 {idx} 已移除"
        else:
            msg = f"帧 {idx} 未标记"
        return self.get_frame_image(idx), msg, self.get_timeline_text()

    def _get_keyframe_data_for_frame(self, frame_idx):
        """查找当前帧应该显示的关键帧数据（区间逻辑）"""
        sorted_keys = sorted(list(self.keyframes.keys()))
        if not sorted_keys:
            return None
        
        # 找到第一个大于等于 frame_idx 的关键帧
        target_kf = None
        for k in sorted_keys:
            if k >= frame_idx:
                target_kf = k
                break
        
        # 如果没找到（frame_idx > 最后一个关键帧），用最后一个
        if target_kf is None:
            target_kf = sorted_keys[-1]
            
        return self.keyframes[target_kf]

    def _draw_overlay_for_export(self, frame_bgr, frame_idx):
        data = self._get_keyframe_data_for_frame(frame_idx)
        if data:
            h, w, _ = frame_bgr.shape
            thick = max(2, int(5 * self.scale_ratio))
            
            # 绘制 Option
            opt_text = data.get("option", "")
            cv2.putText(frame_bgr, opt_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        2.0, (0, 255, 0), 3) # 绿色大字
            
            # 绘制 Point
            if "point" in data:
                orig_pt = data["point"]
                # 导出视频通常是原始分辨率，或者和 cache 一样（缩放后的）。
                # 注意：self.frames_cache 存储的是缩放后的帧 (self.resize_dims)。
                # 所以这里用缩放后的坐标。
                disp_x = int(orig_pt[0] * self.scale_ratio)
                disp_y = int(orig_pt[1] * self.scale_ratio)
                cv2.circle(frame_bgr, (disp_x, disp_y), 15, (0, 0, 255), -1) 
                
        return frame_bgr

    def save_all_data(self, progress=gr.Progress()):
        # 1. 保存 JSON
        sorted_keys = sorted(list(self.keyframes.keys()))
        # 构建新格式的数据
        export_keyframes = {}
        for k in sorted_keys:
            export_keyframes[str(k)] = self.keyframes[k]
            
        json_data = {"path": self.video_path, "keyframes": export_keyframes}
        try:
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            json_msg = "✅ JSON 已保存"
        except Exception as e:
            return f"❌ JSON 保存失败: {e}"

        # 2. 导出 MP4 视频
        if not self.frames_cache:
            return "❌ 视频缓存为空，无法导出"

        try:
            h, w, _ = self.frames_cache[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (w, h))
            
            total = len(self.frames_cache)
            for i, frame in enumerate(self.frames_cache):
                frame_copy = frame.copy()
                # 使用新的导出绘制逻辑
                frame_copy = self._draw_overlay_for_export(frame_copy, i)
                out.write(frame_copy)
                
                if i % 10 == 0:
                    progress((i + 1) / total, desc="正在渲染导出视频...")
            
            out.release()
            video_msg = "✅ MP4 已导出"
        except Exception as e:
            return f"{json_msg}\n❌ 视频导出失败: {e}"

        return f"{json_msg}\n{video_msg}\n关键帧数: {len(sorted_keys)}\n视频路径: {self.output_video_path}"

    def sync_from_player(self, time_float):
        try:
            if time_float is None: return 0, self.get_frame_image(0)
            seconds = float(time_float)
            target_frame = int(seconds * self.fps)
            target_frame = min(max(target_frame, 0), self.total_frames - 1)
            return target_frame, self.get_frame_image(target_frame)
        except Exception as e:
            print(f"Sync error: {e}")
            return 0, self.get_frame_image(0)

    def handle_image_click(self, evt: gr.SelectData, frame_idx):
        """处理图片点击：更新坐标显示，并在图上画点"""
        # Gradio evt.index 是 [x, y] (col, row) ? 
        # 文档说是 [x, y]。我们验证一下。通常是 [col, row]。
        x, y = evt.index[0], evt.index[1]
        coord_str = f"X: {x}, Y: {y}"
        
        # 在当前显示的图上画个点反馈给用户
        # 注意：这里的 frame_idx 对应的图可能已经被 get_frame_image 渲染过了（带之前的标记）
        # 我们再叠加一个临时的点击点（黄色？）
        
        # 获取基础图（带已有标记）
        base_img_rgb = self.get_frame_image(frame_idx)
        # 转 BGR 以便 cv2 处理
        img_bgr = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2BGR)
        
        # 画黄色点表示当前点击
        cv2.circle(img_bgr, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(img_bgr, (x, y), 8, (0, 0, 0), 1) # 黑边
        
        out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return coord_str, out_img

    def create_ui(self):
        js_get_time = "(x) => document.querySelector('video').currentTime"

        with gr.Blocks(title="Auto-Sync Tagger", theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"### ⚡ 自动同步打标工具 (带时间线)")
            
            hidden_time = gr.Number(value=0.0, visible=False)
            
            with gr.Row():
                # --- 左列：播放器 ---
                with gr.Column(scale=5):
                    gr.Markdown("**1. 宏观浏览** (点击暂停 -> 右侧同步)")
                    native_video = gr.Video(value=self.video_path, label="原生播放器", interactive=False)
                    
                    # 将时间线显示放在左下角，高度设大一点方便查看
                    gr.Markdown("---")
                    timeline_box = gr.TextArea(
                        label="📋 已标记关键帧时间线", 
                        value="暂无标记", 
                        lines=15, 
                        interactive=False,
                        text_align="left"
                    )

                # --- 右列：编辑器 ---
                with gr.Column(scale=4):
                    gr.Markdown("**2. 微观编辑器**")
                    # interactive=True 允许点击获取坐标
                    editor_img = gr.Image(label="精确帧预览 (点击选择坐标)", interactive=True)
                    
                    with gr.Row():
                        option_selector = gr.Radio(
                            choices=["Option 1", "Option 2", "Option 3"], 
                            value="Option 1", 
                            label="选择动作类型"
                        )
                        coord_display = gr.Textbox(
                            label="当前选中坐标 (显示分辨率)", 
                            value="X: 0, Y: 0", 
                            interactive=False
                        )

                    slider = gr.Slider(minimum=0, maximum=self.total_frames-1, step=1, label="帧索引", value=0)
                    
                    with gr.Row():
                        btn_prev = gr.Button("⬅️ 上一帧")
                        btn_next = gr.Button("➡️ 下一帧")
                    
                    with gr.Row():
                        btn_mark = gr.Button("🔴 标记 / 更新", variant="primary")
                        btn_remove = gr.Button("🗑️ 删除当前帧", variant="secondary")
                    
                    status = gr.Textbox(label="最新操作状态", lines=1)
                    btn_save_all = gr.Button("💾 保存 JSON + MP4", variant="stop")

            # --- 事件绑定 ---
            
            # 播放器同步
            native_video.pause(fn=self.sync_from_player, inputs=[hidden_time], outputs=[slider, editor_img], js=js_get_time)
            native_video.end(fn=self.sync_from_player, inputs=[hidden_time], outputs=[slider, editor_img], js=js_get_time)

            # 滑块改变
            slider.change(fn=self.get_frame_image, inputs=[slider], outputs=[editor_img])

            # 翻页
            def step(curr, delta):
                val = min(max(curr + delta, 0), self.total_frames - 1)
                return val
            
            btn_prev.click(fn=lambda x: step(x, -1), inputs=[slider], outputs=[slider])
            btn_next.click(fn=lambda x: step(x, 1), inputs=[slider], outputs=[slider])

            # 图片点击
            editor_img.select(
                fn=self.handle_image_click,
                inputs=[slider],
                outputs=[coord_display, editor_img]
            )

            # 标记逻辑：修改了 inputs 和 output
            btn_mark.click(
                fn=self.toggle_keyframe, 
                inputs=[slider, option_selector, coord_display], 
                outputs=[editor_img, status, timeline_box] 
            )

            # 删除逻辑
            btn_remove.click(
                fn=self.remove_keyframe,
                inputs=[slider],
                outputs=[editor_img, status, timeline_box]
            )

            # 保存逻辑
            btn_save_all.click(fn=self.save_all_data, inputs=[], outputs=[status])
            
            # 初始化加载
            demo.load(fn=lambda: self.get_frame_image(0), outputs=[editor_img])

        return demo

if __name__ == "__main__":
    base_dir = "/data/hongzefu/historybench-v5.6.1b3-refractor/video-label"
    # base_dir = "./" 
    
    input_video = os.path.join(base_dir, "input.mp4")
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(input_video):
        output_json = os.path.join(base_dir, "keyframes.json")
        output_video = os.path.join(base_dir, "output_labeled.mp4")
        
        tagger = AutoSyncTagger(input_video, output_json, output_video)
        # 将 output_video_path 传递给 UI，或者 UI 里不显示路径也行，这里保持原样
        tagger.create_ui().queue().launch(server_name="0.0.0.0", share=False)
    else:
        print(f"错误: 未找到视频文件 {input_video}")