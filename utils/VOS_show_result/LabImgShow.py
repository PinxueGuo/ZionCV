import torch
import cv2

def LabImg_show(images_lab_gt, outputs):
    reconstruction = images_lab_gt[0].clone().cuda()
    reconstruction = torch.cat((reconstruction[0,:2], outputs[0]), 0)
    # (3,h,w)-->(h,w,3). Fitting cv2-format to do cv2.cvtColor 
    # 进行cv2.COLOR_LAB2BGR之前，先将调整到[-127, 127].
    reconstruction = reconstruction.permute(1,2,0)  # belong[-1, 1]
    reconstruction = reconstruction*127             # belong[-127, 127]
    reconstruction = cv2.cvtColor(reconstruction.detach().cpu().numpy(), cv2.COLOR_LAB2BGR) # belong[0,1]
    reconstruction = reconstruction*255             # belong[0,255]
    cv2.imwrite('reconstruction.jpg', reconstruction)

