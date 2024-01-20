import segmentation_models_pytorch as smp
import csv
import os
import torch
from utils.transform import aug_base_val
import cv2
from torchmetrics import Dice
from glob import glob
from utils.metrics import AverageMeter

dices_m = AverageMeter()
dice_fn = Dice(average='samples').cuda()
@torch.no_grad()
def infer_one_imge(model,image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    transform = aug_base_val()
    image = transform(image=image)["image"]
    model.eval()
    image=image.unsqueeze(0).cuda()
    out = model(image)
    out = out.sigmoid().cuda()
    return out 

def calculateMetrics(tp,fp,fn,tn):
    Recall = tp / (tp + fn) # Sensivity
    Specifity = tn /(tn + fp) #Specifity
    Precision = tp / (tp + fp) #Preicsion
    NegativePV = tn /(tn + fn) #NPV
    Accuracy = (tp + tn) / (tp + tn + fn + fp)
    F1 = 2*Precision*Recall / (Precision + Recall)
    

    return Recall,Specifity,Precision,NegativePV,Accuracy,F1

def calculateConfusionMatrix(Output,GroundTruth):
    tp, fp, fn, tn = smp.metrics.get_stats(Output, GroundTruth, mode="binary", threshold=0.5)
    return  tp.item(),fp.item(),fn.item(),tn.item()


model = torch.jit.load("C:/Users/SBK/Desktop/Deep Learning/finalproject/models/carsegmentationv5/carsegmentation_combine_encoder_unet_epoch_valloss_0.0865.pt", map_location='cuda')
test_images = glob("C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset/test/images/*.png")
test_gt = glob("C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset/test/mask/*.png")
exp = "carsegmentationv5"
result_path = f'C:/Users/SBK/Desktop/Deep Learning/finalproject/results/{exp}'
if not os.path.isdir(result_path):
    os.makedirs(result_path)
TP = 0
FP = 0
TN = 0
FN = 0
Dice_score = 0
for i,(x,y) in enumerate(zip(test_images,test_gt)):
    gt = cv2.imread(y,cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt,(512,512))
    gt = torch.from_numpy(gt).int().unsqueeze(0).unsqueeze(0).cuda()
    output = infer_one_imge(model,x)
    tp,fp,fn,tn = calculateConfusionMatrix(output,gt) 
    dice_val=dice_fn(output,gt)
    dices_m.update(dice_val.item())
    test_name = os.path.basename(x)
    output_result = output.squeeze(1).squeeze(0).cpu().numpy()
    output_result = (output_result>0.5).astype('uint8')*255
    output_result = cv2.resize(output_result,(1280,720))
    cv2.imwrite(result_path + f'/{test_name}',cv2.cvtColor(output_result,cv2.COLOR_GRAY2RGB))
    TP += tp
    FP += fp
    FN += fn
    TN += tn

Recall, Specifity, Precision, NegativePV, Accuracy, F1 = calculateMetrics(TP,FP,FN,TN)
field_names = ['metrics', 'result']
result = [
    {'metrics':'TP','result': TP},
    {'metrics':'FP','result':FP},
    {'metrics':'FN','result':FN},
    {'metrics':'TN','result':TN},
    {'metrics': 'Precision', 'result': Precision},
    {'metrics':'Recall','result':Recall},
    {'metrics': 'F1', 'result': F1},
    {'metrics':'Specifity','result':Specifity},
    {'metrics': 'Accuracy', 'result': Accuracy},
    {'metrics':'Negative Predict Value','result': NegativePV},
    {'metrics':'Dice','result': dices_m.avg}
]

with open(result_path +'/result.csv','w') as js:
    writer = csv.DictWriter(js, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(result)
