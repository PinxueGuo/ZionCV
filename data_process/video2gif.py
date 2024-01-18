from moviepy.editor import VideoFileClip

# 输入MP4视频文件名
input_video_file = '/Users/pxguo/Downloads/output_vis/ava_2.mp4'

# 输出GIF文件名
output_gif_file = '/Users/pxguo/Downloads/output_vis/ava_2.gif'

# 读取MP4视频
video_clip = VideoFileClip(input_video_file)

# # 将视频保存为GIF文件
# video_clip.speedx(0.5).to_gif(output_gif_file, fps=1)
video_clip.write_gif(output_gif_file, fps=1.0)  # 可以自定义帧速率（fps）

# 关闭视频文件
video_clip.reader.close()