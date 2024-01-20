import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


def aug_train_vhsv(p=1.0):
    return A.Compose(
          [
              A.ShiftScaleRotate(),
              A.VerticalFlip(),
              A.HorizontalFlip(),
              A.HueSaturationValue(),
              A.Resize(512,512),
              A.Normalize(
                  mean=np.array([0.485,0.456,0.406]),
                  std=np.array([0.229,0.224,0.225]),
                  p=1.0
              ),
              ToTensorV2(p=1.0)
          ],
          p=p,
    )

def aug_base_val(p=1.0):
    return A.Compose(
        [
            A.Normalize(
                  mean=np.array([0.485,0.456,0.406]),
                  std=np.array([0.229,0.224,0.225]),
                  p=1.0
              ),
              A.Resize(512,512),
              ToTensorV2(p=1.0)
        
        ],
        p=p
    )
