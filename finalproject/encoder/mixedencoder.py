from typing import List
import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from torchvision.models.resnet import ResNet


class combineEncoder(ResNet,EncoderMixin):

    def __init__(self,depth=5,**kwargs):
        super().__init__(**kwargs)
        self._depth = 5
        self._out_channels = [6,64,128,256,576,1024]
        self._in_channels = 3
        self.resnet = get_encoder("resnet34",weights="imagenet")
        self.mixtransformer = get_encoder("mit_b3",weights="imagenet")

    def forward(self,x):
        resnet_feature = self.resnet(x)
        mit_feature = self.mixtransformer(x)

        features =[]
        for i in range(self._depth +1):
            j = resnet_feature[i]
            k = mit_feature[i]
            x = torch.cat([j,k],dim=1)
            features.append(x)

        return features
    def load_state_dict(self,state_dict,**kwargs):
        pass


smp.encoders.encoders["combine_encoder"]={
    "encoder":combineEncoder,
    "pretraining_settings":{},
    "params":{},
}