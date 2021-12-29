import cv2
import os

mask_dir = '/Users/pxguo/Documents/LocalData/result_PLR-MAMP/MAST/breakdance'
mask_list = os.listdir(mask_dir)

save_dir = '/Users/pxguo/Documents/LocalData/result_PLR-MAMP/breakdance'
if not os.path.exists(save_dir):
    # 路径不存在的话会保存不下来，但不报错
    os.makedirs(save_dir)

for image_name in mask_list:
    print(image_name)
    image_filename = os.path.join(mask_dir, image_name)
    save_filename = os.path.join(save_dir, image_name)
    img = cv2.imread(image_filename)
    img[img[:,:,2]==128] = [0,128,0]        # mask: red-->green
    cv2.imwrite(save_filename, img)
