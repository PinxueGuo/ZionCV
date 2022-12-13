import os
from PIL import Image


# 初始化图片地址文件夹途径
images_path = 'output_vis/ytvis19_swinL_pretrained/0b6db1c6fd'

# 获取文件列表
files = os.listdir(images_path)
files.sort()

# 定义第一个文件的全局路径
file_first_path = os.path.join(images_path, files[0])

# 获取Image对象
img = Image.open(file_first_path)

# 初始化文件对象数组
images = []
for image in files[1:]:
    # 获取当前图片全量路径
    img_path = os.path.join(images_path, image)
    # 将当前图片使用Image对象打开、然后加入到images数组
    images.append(Image.open(img_path))

# 保存并生成gif动图
img.save('output_vis/0b6db1c6fd.gif', save_all=True, append_images=images, loop=0, duration=200)
