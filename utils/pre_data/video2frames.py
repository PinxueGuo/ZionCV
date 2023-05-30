import cv2
import os


def split_single_video(video_path, frames_dir="", rate=1):
	cap = cv2.VideoCapture(video_path)
	cnt = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			success, buffer = cv2.imencode(".png", frame)
			if success:
				if cnt%rate==0:
					with open(f"{frames_dir}{cnt}.png", "wb") as f:
						f.write(buffer.tobytes())
						f.flush()
				cnt += 1
		else:
			break


# rename with the directory where you stored videos
video_dir = "/Users/pxguo/Documents/LocalData/demo_youtube_videos/wedding-7/"
# rename with the directory where you would like to store frames
frames_dir = "/Users/pxguo/Documents/LocalData/demo_youtube_videos/images/"
all_video_paths = os.listdir(video_dir)
TOTAL = 0
for video_path in all_video_paths:
	v_frame_dir = f"{frames_dir}{video_path[:-4]}/"
	os.makedirs(v_frame_dir, exist_ok=True)
	split_single_video(f"{video_dir}{video_path}", frames_dir=v_frame_dir, rate=60)
	TOTAL += 1
	print(TOTAL, video_path)
