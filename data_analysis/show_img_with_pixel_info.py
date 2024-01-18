import cv2

# 定义鼠标事件回调函数
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = img[y, x]
        print(f'坐标 ({x}, {y}) 的 RGB 值为 ({color[2]}, {color[1]}, {color[0]})')

# 读取图像并显示
image_filename = "/Users/pxguo/Desktop/1444.png"
img = cv2.imread(image_filename)
cv2.imshow('image', img)

# 设置鼠标事件回调函数
cv2.setMouseCallback('image', on_mouse)

# 等待用户关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()