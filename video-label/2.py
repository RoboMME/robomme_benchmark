import gradio as gr
import cv2
import json
import numpy as np
import os
import sys

class ProVideoTagger:
    def __init__(self, video_path, output_json_path, output_video_path, target_width=960):
        self.video_path = video_path
        self.output_json_path = output_json_path
        self.output_video_path = output_video_path
        self.target_width = target_width
        
        # 初始化视频信息
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 缓存缩放设置
        self.scale_ratio = 1.0
        if orig_w > self.target_width:
            self.scale_ratio = self.target_width / orig_w
        self.resize_dims = (int(orig_w * self.scale_ratio), int(orig_h * self.scale_ratio))
        
        self.keyframes = set()
        self.frames_cache = []
        
        # 立即开始预加载
        self.preload_video()

    def preload_video(self):
        print(f"正在加载 {self.total_frames} 帧到内存 (FPS: {self.fps:.2f})...")
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            # 缩放 + 转RGB
            if self.scale_ratio != 1.0:
                frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_cache.append(frame)
            cnt += 1
            if cnt % 100 == 0:
                sys.stdout.write(f"\r进度: {cnt}/{self.total_frames}")
                sys.stdout.flush()
        self.cap.release()
        print("\n加载完成。")

    def get_frame_image(self, frame_idx):
        idx = int(frame_idx)
        if idx < 0: idx = 0
        if idx >= len(self.frames_cache): idx = len(self.frames_cache) - 1
        
        # 使用 copy 防止污染缓存
        frame = self.frames_cache[idx].copy()
        
        # 绘制红框
        if idx in self.keyframes:
            h, w, _ = frame.shape
            thick = max(2, int(5 * self.scale_ratio))
            cv2.rectangle(frame, (0, 0), (w, h), (255, 0, 0), thick * 2)
            cv2.putText(frame, f"KEYFRAME #{idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
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
        
        # 重新获取图片以更新红框，并更新列表
        return self.get_frame_image(idx), msg, str(sorted(list(self.keyframes)))

    def time_jump(self, seconds_str):
        """将秒数转换为帧号"""
        try:
            sec = float(seconds_str)
            target_frame = int(sec * self.fps)
            target_frame = min(max(target_frame, 0), self.total_frames - 1)
            return target_frame, self.get_frame_image(target_frame)
        except ValueError:
            return 0, self.get_frame_image(0)

    def save_process(self):
        # ... 保存逻辑同前 (略) ...
        # 为节省篇幅，这里简化，实际请保留之前的 save_process 代码
        sorted_keys = sorted(list(self.keyframes))
        with open(self.output_json_path, 'w') as f:
            json.dump({"path": self.video_path, "keyframes": sorted_keys}, f, indent=4)
        yield "JSON 已保存 (视频渲染代码请参考上一版本)"

    def create_ui(self):
        with gr.Blocks(title="Pro Video Tagger") as demo:
            gr.Markdown(f"### 🚀 专业打标: 原生播放 + 精确帧控")
            
            with gr.Row():
                # --- 左侧：原生播放器 (用于流畅浏览) ---
                with gr.Column(scale=5):
                    gr.Markdown("#### 1. 宏观浏览 (原生播放器)")
                    # 这里直接加载视频文件，使用浏览器原生播放器
                    native_video = gr.Video(value=self.video_path, label="原始视频 (用于查找时间点)", interactive=False)
                    gr.Markdown("> 💡 提示：在这里流畅播放。看到感兴趣的地方暂停，记下**时间(秒)**，在右侧跳转。")

                # --- 右侧：帧编辑器 (用于精确打标) ---
                with gr.Column(scale=5):
                    gr.Markdown("#### 2. 微观操作 (帧编辑器)")
                    
                    # 跳转控制
                    with gr.Row():
                        time_input = gr.Textbox(label="跳转到(秒)", placeholder="例如: 12.5")
                        btn_jump = gr.Button("🚀 跳转", variant="secondary")
                    
                    editor_img = gr.Image(label="当前帧 (精确)", interactive=False)
                    slider = gr.Slider(minimum=0, maximum=self.total_frames-1, step=1, label="帧索引", value=0)
                    
                    # 导航按钮
                    with gr.Row():
                        btn_prev = gr.Button("< 上一帧")
                        btn_next = gr.Button("下一帧 >")
                    
                    # 标记区
                    with gr.Row():
                        btn_mark = gr.Button("🔴 标记/取消", variant="primary")
                        status = gr.Textbox(label="状态", lines=1)
                    
                    kf_list = gr.TextArea(label="已标记帧", lines=3)
                    btn_save = gr.Button("💾 保存结果")

            # --- 事件逻辑 ---

            # 1. 拖动滑块 -> 实时更新图片 (因为是内存缓存，slider.change 很流畅)
            slider.change(fn=self.get_frame_image, inputs=[slider], outputs=[editor_img])

            # 2. 时间跳转 logic
            # 输入秒数 -> 计算帧号 -> 更新滑块位置 -> (滑块位置变动会自动触发上面的 change 事件更新图片)
            btn_jump.click(fn=self.time_jump, inputs=[time_input], outputs=[slider, editor_img])

            # 3. 按钮微调
            def step(curr, delta):
                val = min(max(curr + delta, 0), self.total_frames - 1)
                return val # 只需返回新值，slider.change 会负责更新图片
            
            btn_prev.click(fn=lambda x: step(x, -1), inputs=[slider], outputs=[slider])
            btn_next.click(fn=lambda x: step(x, 1), inputs=[slider], outputs=[slider])

            # 4. 标记
            btn_mark.click(fn=self.toggle_keyframe, inputs=[slider], outputs=[editor_img, status, kf_list])
            
            # 5. 保存
            btn_save.click(fn=self.save_process, outputs=[status])
            
            # 初始化
            demo.load(fn=lambda: self.get_frame_image(0), outputs=[editor_img])

        return demo

if __name__ == "__main__":
    base_dir = "/data/hongzefu/historybench-v5.6.1b3-refractor/video-label"
    input_video = os.path.join(base_dir, "input.mp4")
    if os.path.exists(input_video):
        # 目标宽度建议 640 或 960，取决于你的屏幕大小
        output_json = os.path.join(base_dir, "keyframes.json")
        output_video = os.path.join(base_dir, "out.mp4")
        tagger = ProVideoTagger(input_video, output_json, output_video, target_width=960)
        tagger.create_ui().queue().launch(server_name="0.0.0.0", share=False)
    else:
        print("未找到视频文件")