import numpy as np
import cv2

def db_eval_iou(annotation,segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
    """
    annotation   = annotation.astype(np.bool)       # uint8-->bool: 0 --> False, 1-255 --> True
    segmentation = segmentation.astype(np.bool)

    cv2.imwrite('/Users/pxguo/Documents/LocalData/result_PLR-MAMP/ckpt_kl_IoU/plr_davis_Abaltion-IoU-Youtube-PL02/DAVIS/04735c5030_00025_I.png', 
                  (annotation&segmentation).astype(np.int8)*255)
    cv2.imwrite('/Users/pxguo/Documents/LocalData/result_PLR-MAMP/ckpt_kl_IoU/plr_davis_Abaltion-IoU-Youtube-PL02/DAVIS/04735c5030_00025_U.png', 
                  (annotation|segmentation).astype(np.int8)*255)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def main():
    img1_filename = '/Users/pxguo/Documents/LocalData/result_PLR-MAMP/ckpt_kl_IoU/plr_davis_Abaltion-IoU-Youtube-PL02/DAVIS/diff/04735c5030/00025.png'
    img2_filename = '/Users/pxguo/Documents/LocalData/result_PLR-MAMP/ckpt_kl_IoU/plr_davis_Abaltion-IoU-Youtube-PL02/DAVIS/kl/04735c5030/00025.png'
    img1 = cv2.imread(img1_filename, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_filename, cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape      # (h, w)
    img2 = cv2.resize(img2, (w,h))
    # unique,count=np.unique(img2,return_counts=True)
    # data_count=dict(zip(unique,count))
    # img1[img1<10]=0
    # img2[img2<35]=0
    iou = db_eval_iou(img1, img2)
    print(iou)

if __name__ == '__main__':
    main()