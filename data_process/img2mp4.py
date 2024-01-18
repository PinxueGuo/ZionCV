import cv2
import os

# 主文件夹的路径，包含多个子文件夹，每个子文件夹包含一个图像序列
main_folder = "/Users/pxguo/Downloads/output_vis"

# 循环遍历主文件夹中的每个子文件夹
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)

    # 获取第一张图像的尺寸以设置视频大小
    first_image_path = os.path.join(subfolder_path, os.listdir(subfolder_path)[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape  # 获取图像的高度和宽度

    # 创建一个空白的视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编解码器
    video_out = cv2.VideoWriter(main_folder+'/'+f"{subfolder}.mp4", fourcc, 3.0, (width, height))  # 设置帧率和视频大小

    # 循环遍历子文件夹中的图像文件并将它们添加到视频中
    for filename in sorted(os.listdir(subfolder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(subfolder_path, filename)
            frame = cv2.imread(image_path)
            video_out.write(frame)

    # 关闭视频写入对象
    video_out.release()

print("转换完成！")