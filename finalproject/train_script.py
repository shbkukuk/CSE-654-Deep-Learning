import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp 
import argparse
import os 
import random
import numpy as np
import json
from tqdm import tqdm
from torchmetrics import Dice 
from comet_ml import Experiment
from utils.transform import aug_train_vhsv,aug_base_val
from utils import dataloader
from utils.metrics import *
from encoder.mixedencoder import combineEncoder
from torchvision.models.resnet import BasicBlock 
from pretrainedmodels.models.torchvision_models import pretrained_settings
experiment = Experiment(
  api_key="S1oTMI0gIuhlnnGNbd5nN9TkG",
  project_name="deeplearning",
  workspace="sbk061",
  auto_metric_logging=False,
  disabled=False,
  log_git_metadata=False,)


smp.encoders.encoders["combine_encoder"]={
    "encoder":combineEncoder,
    "pretrained_settings":pretrained_settings['resnet34'],
    "params":{"layers":[2,2,2,2],
              "block":BasicBlock},
}

parser = argparse.ArgumentParser(description="The program write for Deep Learning Project")
parser.add_argument("--experiment", type=str, required=False, default='carsegmentationv5', help=" Expreminet name of training")
parser.add_argument("--data_dir",type=str, default='C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset',help="dataset of path")
parser.add_argument("--encoder",type=str, default='combine_encoder', help="name of  encoder to train")
parser.add_argument("--model",type=str,default="unet",help="name of training model")
parser.add_argument("--weight",type=str,default="",help="pre-trained model name")
parser.add_argument("--use_amp",type=bool,default=True,help="train with mixed precion or not")
parser.add_argument("--optimizer",type=str,default="Adam",help="Optimizer that use for training")
parser.add_argument("-loss",type=str,default="bce",help="loss function for using train section")
parser.add_argument("--learning-rate",type=float,default=3e-4,help="default learning rate to train")
parser.add_argument("--early-stopping",type=int,default=15,help="number of epochs to before early stopping")
parser.add_argument("--augmentation",type=str,default="vhsv",help="augmentation function to use for train")
parser.add_argument("--image-size",type=int,default=512,help="train and validation image size ")
parser.add_argument("--batch-size",type=int,default=8,help="batch size for dataloader")
parser.add_argument("--weight-decay",type=float,default=0)
parser.add_argument("--use-amp",type=bool,default=True,help="trainig with mixed precision")

def create_traced_model(model,model_input,path):
    traced_net = torch.jit.trace(model,model_input)
    traced_net.save(path)
    print(f"model save to {path}")

def seed_evertyhing(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed=seed)
@experiment.train()
def train_one_epoch(epoch,model,loss_fn,dice_fn,train_loader,optimizer,autocast,loss_scaler):
    experiment.log_current_epoch(epoch)
    model.train()
    losses_m = AverageMeter()
    dices_m = AverageMeter()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for step, (img,mask) in pbar:
        imgs = img.cuda()
        gt = mask.cuda()
        with autocast():
            img_preds = model(imgs)
            loss = loss_fn((torch.squeeze(img_preds)),gt.float())
            dice = dice_fn((img_preds),gt.unsqueeze(dim=1).long())

        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        
        losses_m.update(loss.item(),imgs.size(0))
        dices_m.update(dice.item(),imgs.size(0))
        optimizer.zero_grad()

        if ((step+1)%1 == 0) or ((step+1)==len(train_loader)):
            desc = (
                f"epoch {epoch}, loss: {losses_m.val:.4f}, smt_loss: {losses_m.avg:.4f} "
                f"dice: {dices_m.val:.4f}, smt-dice: {dices_m.avg:.4f} "
            )
            pbar.set_description(desc)
    
    loss_train = losses_m.avg
    dice_train = dices_m.avg
    experiment.log_metrics(
        {
            "train_loss":loss_train,
            "dice_train":dice_train,
            
        },
        epoch=epoch
    )
    return loss_train

experiment.validate()
def val_one_epoch(epoch,model,loss_fn,dice_fn,val_loader,autocast):
    model.eval()
    losses_m = AverageMeter()
    dices_m = AverageMeter()
    pbar = tqdm(enumerate(val_loader),total=len(val_loader))
    with torch.no_grad():
        for step, (img,mask) in pbar:
            imgs = img.cuda()
            gt = mask.cuda()
            with autocast():
                img_preds = model(imgs)
                loss = loss_fn((torch.squeeze(img_preds)),gt.float())
                dice = dice_fn((img_preds),gt.unsqueeze(dim=1).long())

            
            
            losses_m.update(loss.item(),imgs.size(0))
            dices_m.update(dice.item(),imgs.size(0))
           

            if ((step+1)%1 == 0) or ((step+1)==len(val_loader)):
                desc = (
                    f"epoch {epoch}, val_loss: {losses_m.val:.4f}, smt_val_loss: {losses_m.avg:.4f} "
                    f"dice: {dices_m.val:.4f}, smt-val-dice: {dices_m.avg:.4f} "
                )
                pbar.set_description(desc)
    
    loss_train = losses_m.avg
    dice_train = dices_m.avg
    experiment.log_metrics(
        {
            "val_loss":loss_train,
            "dice_val":dice_train,
            
        },
        epoch=epoch
    )
    return loss_train



def _parse_args():
    args = parser.parse_args()
    return args

def main():
    seed_evertyhing()
    args = _parse_args()
    experiment.set_name(args.experiment)
    model_save_path = f"C:/Users/SBK/Desktop/Deep Learning/finalproject/models/{args.experiment}"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    #logger 
    experiment.log_parameters(vars(args))
    saveable = torch.rand(1,3,args.image_size,args.image_size).cuda()
    start_epoch = 1
    train_loss_all = []
    val_loss_all = []

    if args.augmentation == "vhsv":
        aug_train = aug_train_vhsv
        aug_val = aug_base_val
    
    train_loader , val_loader = dataloader.data_loader(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.image_size,
        train_aug=aug_train(),
        val_aug=aug_val(),

    )

    best_val_loss = 999999
    best_epoch = 0
    not_improving = 0
    early_stopping = args.early_stopping

    model = smp.Unet(
        encoder_name=args.encoder,
        #encoder_weights=args.weight,
        in_channels=3,
        classes=1
    )
    model.cuda()
    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss().cuda()
        dice_fn = Dice(average="samples").cuda()
    else:
        print(f"{args.loss}  function is not implemented")
    if args.use_amp:
        autocast = torch.cuda.amp.autocast
        loss_scaler= torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
        
    try:
        for epoch in range(start_epoch,100):
            train_loss = train_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss_fn,
                dice_fn=dice_fn,
                train_loader=train_loader,
                optimizer=optimizer,
                autocast=autocast,
                loss_scaler=loss_scaler
            )
            val_loss = val_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss_fn,
                dice_fn=dice_fn,
                val_loader=val_loader,
                autocast=autocast
            )

            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)

            if val_loss < best_val_loss:
                
                print(f"Epoch {epoch}, Metric improved from {best_val_loss:.4f} to -> {val_loss:.4f}. Saving model")
                best_val_loss = val_loss
                best_epoch = epoch
                create_traced_model(
                    model,
                    saveable,
                    f"{model_save_path}/carsegmentation_{args.encoder}_{args.model}_epoch_valloss_{val_loss:.4f}.pt",
                )
            if val_loss > best_val_loss:
                not_improving += 1
                print(f"Epoch {epoch}: train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, Early Stopping  round: {not_improving}"
                      )
            
            if early_stopping == not_improving:
                print("Early Stopping")
                print(f"Best loss was: {best_val_loss:.4f} ,at epoch {best_epoch}")
                break
        loss_dict = {
            'train_loss':train_loss_all,
            'val_loss':val_loss_all
        }
        with open('result.json','w') as f:
            json.dump(loss_dict,f,indent=4)
        del model,optimizer,train_loader,val_loader

    except KeyboardInterrupt:
        pass



if __name__ == '__main__':
    main()