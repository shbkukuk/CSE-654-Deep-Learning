from glob import glob
import cv2
import os
import matplotlib.pyplot as plt 


mask_gt = 'C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset/test/mask/*.png'
mask_path = glob(mask_gt)
images_path = glob(mask_gt.replace('mask','images'))
result_gt = 'C:/Users/SBK/Desktop/Deep Learning/finalproject/results/'
exp = ['carsegmentationv2','carsegmentationv4','carsegmentationv5']
save_path = 'C:/Users/SBK/Desktop/Deep Learning/finalproject/results/combine/'
for i,(x,y) in enumerate(zip(images_path,mask_path)):
    name = os.path.basename(x)
    fig ,ax = plt.subplots(1,5)
    img = cv2.imread(x)
    mask = cv2.imread(y,0)
    gt_1 = cv2.imread(result_gt+exp[0]+f'/{name}',0)
    gt_2 = cv2.imread(result_gt+exp[1]+f'/{name}',0)
    gt_3 = cv2.imread(result_gt+exp[2]+f'/{name}',0)
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(mask,cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    ax[2].imshow(gt_1,cmap='gray')
    ax[2].set_title('Resnet Result')
    ax[2].axis('off')
    ax[3].imshow(gt_2,cmap='gray')
    ax[3].set_title('mit-b3 Result')
    ax[3].axis('off')
    ax[4].imshow(gt_3,cmap='gray')
    ax[4].set_title('Fusion Result')
    ax[4].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(save_path + name, bbox_inches='tight')
    plt.clf()
    plt.close()