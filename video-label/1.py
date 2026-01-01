import gradio as gr
import cv2
import json
import numpy as np
import os
import sys
import time

class CachedVideoTagger:
    def __init__(self, video_path, output_json_path, output_video_path, target_width=960):
        self.video_path = video_path
        self.output_json_path = output_json_path
        self.output_video_path = output_video_path
        self.target_width = target_width
        
        # 初始化
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 缩放比例计算
        self.scale_ratio = 1.0
        if orig_width > self.target_width:
            self.scale_ratio = self.target_width / orig_width
        self.resize_dims = (int(orig_width * self.scale_ratio), int(orig_height * self.scale_ratio))
        
        self.keyframes = set()
        self.frames_cache = []
        self.preload_video()

    def preload_video(self):
        print(f"正在预加载 {self.total_frames} 帧到内存... (FPS: {self.fps})")
        current = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            if self.scale_ratio != 1.0:
                frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_cache.append(frame)
            current += 1
            if current % 100 == 0:
                sys.stdout.write(f"\r加载进度: {current}/{self.total_frames}")
                sys.stdout.flush()
        print("\n预加载完成！")
        self.cap.release()

    def get_frame_image(self, frame_idx):
        frame_idx = int(frame_idx)
        # 循环播放保护：超过总帧数则停在最后一帧
        if frame_idx >= len(self.frames_cache): frame_idx = len(self.frames_cache) - 1
        if frame_idx < 0: frame_idx = 0
        
        frame = self.frames_cache[frame_idx].copy()
        
        if frame_idx in self.keyframes:
            h, w, _ = frame.shape
            thick = max(2, int(5 * self.scale_ratio))
            cv2.rectangle(frame, (0, 0), (w, h), (255, 0, 0), thick * 2)
            cv2.putText(frame, "KEYFRAME", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0 * self.scale_ratio, (255, 0, 0), thick)
        return frame

    # --- 新增：播放逻辑 ---
    def play_video(self, start_frame):
        """生成器函数：从当前帧开始，持续 yield 下一帧"""
        # 计算每帧间隔 (稍微减去一点网络传输的预估损耗，比如0.005秒)
        # 如果网络慢，这个 delay 只能起到限速作用，不能加速
        delay = 1.0 / self.fps 
        
        for i in range(int(start_frame), self.total_frames):
            # 获取当前帧图像
            img = self.get_frame_image(i)
            
            # Yield 返回：(更新滑块值, 更新图片)
            yield i, img
            
            # 控制播放速度
            time.sleep(delay)

    def toggle_keyframe(self, frame_idx):
        frame_idx = int(frame_idx)
        msg = ""
        if frame_idx in self.keyframes:
            self.keyframes.remove(frame_idx)
            msg = f"帧 {frame_idx} 移除"
        else:
            self.keyframes.add(frame_idx)
            msg = f"帧 {frame_idx} 标记"
        sorted_keys = sorted(list(self.keyframes))
        return self.get_frame_image(frame_idx), f"状态: {msg}", str(sorted_keys)

    def save_process(self):
        sorted_keyframes = sorted(list(self.keyframes))
        data = {"video_path": self.video_path, "total_frames": self.total_frames, "keyframes": sorted_keyframes}
        with open(self.output_json_path, 'w') as f: json.dump(data, f, indent=4)
        yield "JSON已保存，开始渲染视频..."
        
        if not sorted_keyframes:
            yield "无关键帧，结束。"
            return

        cap_render = cv2.VideoCapture(self.video_path)
        fps = cap_render.get(cv2.CAP_PROP_FPS)
        w, h = int(cap_render.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_render.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for i in range(self.total_frames):
            ret, frame = cap_render.read()
            if not ret: break
            if i in self.keyframes: cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            out.write(frame)
            if i % 50 == 0: yield f"渲染: {i}/{self.total_frames}"
        
        cap_render.release()
        out.release()
        yield f"完成: {self.output_video_path}"

    def create_ui(self):
        with gr.Blocks(title="Speed Tagger Player") as demo:
            gr.Markdown(f"### 🎬 缓存播放版: `{os.path.basename(self.video_path)}`")
            
            with gr.Row():
                with gr.Column(scale=8):
                    display_img = gr.Image(label="预览", interactive=False)
                    slider = gr.Slider(minimum=0, maximum=self.total_frames-1, step=1, label="进度", value=0)
                    
                    # --- 播放控制区 ---
                    with gr.Row():
                        btn_play = gr.Button("▶️ 播放 (Play)", variant="secondary")
                        btn_pause = gr.Button("⏸️ 暂停 (Pause)", variant="secondary")
                    
                    # --- 导航控制区 ---
                    with gr.Row():
                        btn_prev = gr.Button("<< -1", size='sm')
                        btn_next = gr.Button("+1 >>", size='sm')
                        btn_skip_back = gr.Button("<< -30", size='sm')
                        btn_skip_fwd = gr.Button("+30 >>", size='sm')

                with gr.Column(scale=2):
                    btn_mark = gr.Button("🔴 标记/取消 (K)", variant="primary")
                    status_text = gr.Textbox(label="日志")
                    marked_list = gr.TextArea(label="关键帧列表", value="[]")
                    gr.Markdown("---")
                    btn_save = gr.Button("💾 保存结果", variant="stop")
                    save_output = gr.Textbox(label="保存状态")

            # --- 事件绑定 ---
            
            # 1. 播放功能 (核心)
            # 点击播放 -> 触发生成器 play_video
            play_event = btn_play.click(
                fn=self.play_video, 
                inputs=[slider], 
                outputs=[slider, display_img]
            )
            
            # 点击暂停 -> 强制取消播放事件 (cancels)
            btn_pause.click(fn=None, inputs=None, outputs=None, cancels=[play_event])

            # 2. 常规导航
            slider.change(fn=self.get_frame_image, inputs=[slider], outputs=[display_img])
            
            def step(curr, delta):
                val = min(max(curr + delta, 0), self.total_frames - 1)
                return val, self.get_frame_image(val)

            btn_prev.click(fn=lambda x: step(x, -1), inputs=[slider], outputs=[slider, display_img])
            btn_next.click(fn=lambda x: step(x, 1), inputs=[slider], outputs=[slider, display_img])
            btn_skip_back.click(fn=lambda x: step(x, -30), inputs=[slider], outputs=[slider, display_img])
            btn_skip_fwd.click(fn=lambda x: step(x, 30), inputs=[slider], outputs=[slider, display_img])

            # 3. 标记与保存
            btn_mark.click(fn=self.toggle_keyframe, inputs=[slider], outputs=[display_img, status_text, marked_list])
            btn_save.click(fn=self.save_process, outputs=[save_output])
            
            demo.load(fn=lambda: self.get_frame_image(0), outputs=[display_img])
            
        return demo

if __name__ == "__main__":
    base_dir = "/data/hongzefu/historybench-v5.6.1b3-refractor/video-label"
    input_video = os.path.join(base_dir, "input.mp4")
    if os.path.exists(input_video):
        output_json = os.path.join(base_dir, "keyframes.json")
        output_video = os.path.join(base_dir, "output_marked.mp4")
        tagger = CachedVideoTagger(input_video, output_json, output_video, target_width=960)
        demo = tagger.create_ui()
        demo.queue().launch(server_name="0.0.0.0", share=False)
    else:
        print("Video not found.")