import torch.nn as nn
import timm

class DensenetNet3D(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=False, features_only=False, **kwargs):
        super().__init__()
        model_name = "_".join(model_name.split("_")[:-1])
        self.encoder = timm.create_model(
            model_name, 
            pretrained=True, 
            in_chans=in_c//3, 
            features_only=False,
            drop_rate=0.1,
            proj_drop_rate=0.1,
            num_classes=1,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = self.encoder.num_features
        #self.proj = nn.Linear(num_features, num_features//3, bias=True)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features*3, n_classes),
        )

    def forward(self, x):
        bs, total_c, img_size, _ = x.shape
        c_per_type = total_c // 3

        x = x.view(-1, c_per_type, img_size, img_size)
        x = self.encoder.forward_features(x)
        x = self.avgpool(x)
        x = x.view(bs, 3, -1)
        #x = self.proj(x)
        x = x.view(bs, -1)
        x = self.head(x)
        return x
