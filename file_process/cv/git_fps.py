from moviepy.editor import VideoFileClip

# 指定输入的GIF文件路径
input_gif_path = '/Users/pxguo/Downloads/output_vis/ava_1.gif'

# 指定输出的GIF文件路径
output_gif_path = '/Users/pxguo/Downloads/output_vis/ava_1_slow.gif'

# 使用VideoFileClip加载GIF文件
gif_clip = VideoFileClip(input_gif_path)

# # 减小帧率（例如，将帧率减少到原始的一半）
# new_fps = gif_clip.fps / 10

# 修改视频剪辑的帧率
gif_clip = gif_clip.set_duration(gif_clip.duration*100)

# 将修改后的视频剪辑保存为新的GIF文件
gif_clip.write_gif(output_gif_path)

print(f'已生成新的GIF文件：{output_gif_path}')