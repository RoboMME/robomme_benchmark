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
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_ratio = 1.0
        if orig_w > self.target_width:
            self.scale_ratio = self.target_width / orig_w
        self.resize_dims = (int(orig_w * self.scale_ratio), int(orig_h * self.scale_ratio))
        
        self.keyframes = set()
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
            h, w, _ = frame_bgr.shape
            thick = max(2, int(5 * self.scale_ratio))
            # 蓝色框
            cv2.rectangle(frame_bgr, (0, 0), (w, h), (255, 0, 0), thick * 2)
            cv2.putText(frame_bgr, f"KEY: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0 * self.scale_ratio, (255, 0, 0), thick)
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
        sorted_keys = sorted(list(self.keyframes))
        if not sorted_keys:
            return "暂无标记"
        
        lines = []
        lines.append(f"总计标记: {len(sorted_keys)} 帧")
        lines.append("-" * 30)
        for k in sorted_keys:
            t = k / self.fps
            time_str = self._format_time(t)
            lines.append(f"Frame {k:<5} | Time {time_str}")
        return "\n".join(lines)

    # --- 修改：返回值增加 timeline_text ---
    def toggle_keyframe(self, frame_idx):
        idx = int(frame_idx)
        msg = ""
        if idx in self.keyframes:
            self.keyframes.remove(idx)
            msg = f"帧 {idx} 已移除"
        else:
            self.keyframes.add(idx)
            msg = f"帧 {idx} 已标记"
        
        # 返回：当前图像，状态消息，更新后的时间线列表
        return self.get_frame_image(idx), msg, self.get_timeline_text()

    def save_all_data(self, progress=gr.Progress()):
        # 1. 保存 JSON
        sorted_keys = sorted(list(self.keyframes))
        json_data = {"path": self.video_path, "keyframes": sorted_keys}
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
                frame_copy = self._draw_overlay(frame_copy, i)
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
                    editor_img = gr.Image(label="精确帧预览", interactive=False)
                    slider = gr.Slider(minimum=0, maximum=self.total_frames-1, step=1, label="帧索引", value=0)
                    
                    with gr.Row():
                        btn_prev = gr.Button("⬅️ 上一帧")
                        btn_next = gr.Button("➡️ 下一帧")
                    
                    btn_mark = gr.Button("🔴 标记 / 取消 (Toggle)", variant="primary")
                    
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

            # 标记逻辑：修改了 output，加入了 timeline_box
            btn_mark.click(
                fn=self.toggle_keyframe, 
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