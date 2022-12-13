import cv2
import os


def split_single_video(video_path, frames_dir=""):
	cap = cv2.VideoCapture(video_path)
	cnt = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			success, buffer = cv2.imencode(".png", frame)
			if success:
				file_name = '{:0>5d}'.format(cnt)+'.png'
				file_path = frames_dir+file_name
				with open(file_path, "wb") as f:
					f.write(buffer.tobytes())
					f.flush()
				cnt += 1
		else:
			break


# rename with the directory where you stored videos
video_dir = "/home/et21-guopx/codes/DATA/HonorVOS/example/mp4/"
# rename with the directory where you would like to store frames
frames_dir = "/home/et21-guopx/codes/DATA/HonorVOS/example/JPEGImages/"
all_video_paths = os.listdir(video_dir)
TOTAL = 0
for video_path in all_video_paths:
	v_frame_dir = f"{frames_dir}{video_path[:-4]}/"
	os.mkdir(v_frame_dir)
	split_single_video(f"{video_dir}{video_path}", frames_dir=v_frame_dir)
	TOTAL += 1
	print(TOTAL, video_path)
