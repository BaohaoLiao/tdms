import torch.nn as nn
import timm

class StandardModel3D(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=False, features_only=False, **kwargs):
        super().__init__()
        self.encoder = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    in_chans=3,
                                    num_classes=0,
                                    global_pool='',
                                    drop_rate=0., 
                                    drop_path_rate=0.,
                                    **kwargs)
        
    
    def forward(self, x):
        y = self.model(x)
        return y