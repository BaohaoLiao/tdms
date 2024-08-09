import torch.nn as nn
import timm

class StandardModel(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=False, features_only=False, **kwargs):
        super().__init__()
        self.model = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    features_only=features_only,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg',
                                    **kwargs)
    
    def forward(self, x):
        y = self.model(x)
        return y