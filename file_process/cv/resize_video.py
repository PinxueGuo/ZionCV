from moviepy.editor import VideoFileClip

# 输入MP4文件名
input_video_file = '/Users/pinxue/Downloads/MOT17-vid2_seg.mp4'

# 输出MP4文件名
output_video_file = '/Users/pinxue/Downloads/MOT17-vid2_seg_lowres.mp4'

# 读取MP4视频
video_clip = VideoFileClip(input_video_file)

# 调整分辨率为0.5倍
resized_clip = video_clip.resize(0.5)

# 保存为新的MP4文件
resized_clip.write_videofile(output_video_file, codec='libx264')

# 关闭视频文件
video_clip.reader.close()
video_clip.audio.reader.close_proc()

print(f'Resized video saved as {output_video_file}')