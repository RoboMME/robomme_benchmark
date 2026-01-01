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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_cache.append(frame)
            cnt += 1
            if cnt % 100 == 0:
                sys.stdout.write(f"\r加载: {cnt}/{self.total_frames}")
                sys.stdout.flush()
        print("\n预加载完成！")
        self.cap.release()

    def get_frame_image(self, frame_idx):
        idx = int(frame_idx)
        if idx < 0: idx = 0
        if idx >= len(self.frames_cache): idx = len(self.frames_cache) - 1
        
        frame = self.frames_cache[idx].copy()
        
        if idx in self.keyframes:
            h, w, _ = frame.shape
            thick = max(2, int(5 * self.scale_ratio))
            cv2.rectangle(frame, (0, 0), (w, h), (255, 0, 0), thick * 2)
            cv2.putText(frame, f"KEY: {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0 * self.scale_ratio, (255, 0, 0), thick)
        return frame

    def toggle_keyframe(self, frame_idx):
        idx = int(frame_idx)
        msg = ""
        if idx in self.keyframes:
            self.keyframes.remove(idx)
            msg = f"帧 {idx} 已移除"
        else:
            self.keyframes.add(idx)
            msg = f"帧 {idx} 已标记"
        return self.get_frame_image(idx), msg, str(sorted(list(self.keyframes)))

    def save_process(self):
        sorted_keys = sorted(list(self.keyframes))
        data = {"path": self.video_path, "keyframes": sorted_keys}
        with open(self.output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        return f"已保存 {len(sorted_keys)} 个关键帧到 JSON"

    # --- 新增：处理自动同步的函数 ---
    def sync_from_player(self, time_float):
        """接收前端传来的秒数，转换为帧并返回图片"""
        try:
            # 浏览器传来的可能是 None 或 float
            if time_float is None: return 0, self.get_frame_image(0)
            
            seconds = float(time_float)
            target_frame = int(seconds * self.fps)
            
            # 修正范围
            target_frame = min(max(target_frame, 0), self.total_frames - 1)
            
            # 返回：更新滑块数值，更新图片
            return target_frame, self.get_frame_image(target_frame)
        except Exception as e:
            print(f"Sync error: {e}")
            return 0, self.get_frame_image(0)

    def create_ui(self):
        # 定义 JS 代码：获取页面上第一个 video 标签的当前播放时间并返回
        js_get_time = "(x) => document.querySelector('video').currentTime"

        with gr.Blocks(title="Auto-Sync Tagger") as demo:
            gr.Markdown(f"### ⚡ 自动同步打标工具 (暂停即对齐)")
            
            # 隐藏的 Number 组件用于接收 JS 返回的时间值
            hidden_time = gr.Number(value=0.0, visible=False)
            
            with gr.Row():
                # --- 左侧：原生播放器 ---
                with gr.Column(scale=1):
                    gr.Markdown("**宏观浏览** (在此处点击暂停)")
                    # elem_id 用于 CSS 样式（如果需要），这里主要靠 querySelector('video')
                    native_video = gr.Video(value=self.video_path, label="原生播放器", interactive=False)
                    gr.Info("💡 操作提示：在左侧播放视频，点击暂停时，右侧会自动跳转到对应帧。")

                # --- 右侧：微观编辑器 ---
                with gr.Column(scale=1):
                    gr.Markdown("**微观编辑器** (自动跳转，精确操作)")
                    editor_img = gr.Image(label="精确帧预览", interactive=False)
                    slider = gr.Slider(minimum=0, maximum=self.total_frames-1, step=1, label="帧索引", value=0)
                    
                    with gr.Row():
                        btn_prev = gr.Button("< 上一帧")
                        btn_next = gr.Button("下一帧 >")
                        btn_mark = gr.Button("🔴 标记/取消", variant="primary")
                    
                    status = gr.Textbox(label="状态", lines=1)
                    btn_save = gr.Button("💾 保存 JSON")

            # --- 事件绑定 ---

            # 1. 【核心】自动同步逻辑
            # 当 native_video 触发 pause 事件时，执行 js_get_time，将结果传给 sync_from_player
            native_video.pause(
                fn=self.sync_from_player,
                inputs=[hidden_time], # 使用隐藏的 Number 组件接收 JS 返回的时间值
                outputs=[slider, editor_img], # 同时更新滑块和图片
                js=js_get_time # 注入 JS，返回的时间值会更新 hidden_time
            )
            
            # 2. 也可以加上"播放结束"时自动对齐
            native_video.end(
                fn=self.sync_from_player,
                inputs=[hidden_time], # 使用隐藏的 Number 组件接收 JS 返回的时间值
                outputs=[slider, editor_img],
                js=js_get_time
            )

            # 3. 滑块拖动 (缓存级流畅)
            slider.change(fn=self.get_frame_image, inputs=[slider], outputs=[editor_img])

            # 4. 微调按钮
            def step(curr, delta):
                val = min(max(curr + delta, 0), self.total_frames - 1)
                return val # 仅更新滑块，由滑块触发图片更新
            
            btn_prev.click(fn=lambda x: step(x, -1), inputs=[slider], outputs=[slider])
            btn_next.click(fn=lambda x: step(x, 1), inputs=[slider], outputs=[slider])

            # 5. 标记和保存
            btn_mark.click(fn=self.toggle_keyframe, inputs=[slider], outputs=[editor_img, status, status]) # 复用 status 显示列表略显拥挤，这里简化
            btn_save.click(fn=self.save_process, outputs=[status])
            
            # 初始化
            demo.load(fn=lambda: self.get_frame_image(0), outputs=[editor_img])

        return demo

if __name__ == "__main__":
    base_dir = "/data/hongzefu/historybench-v5.6.1b3-refractor/video-label"
    input_video = os.path.join(base_dir, "input.mp4")
    if os.path.exists(input_video):
        output_json = os.path.join(base_dir, "keyframes.json")
        output_video = os.path.join(base_dir, "out.mp4")
        tagger = AutoSyncTagger(input_video, output_json, output_video)
        # server_name="0.0.0.0" 允许远程访问
        tagger.create_ui().queue().launch(server_name="0.0.0.0", share=False)
    else:
        print("未找到视频文件")