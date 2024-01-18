from moviepy.editor import VideoFileClip

input_webm_path = '/Users/pxguo/Downloads/AVA_2.webm'
output_mp4_path = '/Users/pxguo/Downloads/AVA_2.mp4'

# 使用VideoFileClip加载WebM视频
video_clip = VideoFileClip(input_webm_path)

# 将视频保存为MP4文件
video_clip.write_videofile(output_mp4_path, codec='libx264', fps=1)

print(f'已生成MP4视频文件: {output_mp4_path}')