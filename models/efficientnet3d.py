import torch.nn as nn
import timm

class EfficientNet3D(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=False, features_only=False, **kwargs):
        super().__init__()
        model_name = "_".join(model_name.split("_")[:-1])
        drop = 0.
        true_encoder = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=3, 
            global_pool='', 
            num_classes=0,
            drop_rate=drop, 
            drop_path_rate=drop
        )
        self.encoder = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=3, 
            features_only=True,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feats = true_encoder.num_features
        lstm_embed = feats * 1
        self.lstm = nn.LSTM(
            lstm_embed, 
            lstm_embed//2, 
            num_layers=1, 
            dropout=drop, 
            bidirectional=True, 
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3*lstm_embed, n_classes),
        )
        
        # init
        st = true_encoder.state_dict()
        self.encoder.load_state_dict(st, strict=False)      
    
    def forward(self, x):
        bs, n_slice_per_c, img_size, _ = x.shape
        in_chans = 3
        n_slice_per_c = n_slice_per_c // in_chans

        x = x.view(-1, 3, img_size, img_size)
        x = self.encoder(x)[-1]
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.avgpool(x)

        x = x.view(bs, n_slice_per_c, -1)
        x = x.view(bs*3, n_slice_per_c//3, -1)
        
        x, _ = self.lstm(x)
        x = x.mean(dim=1)

        x = x.view(bs, -1)

        x = self.head(x)
        return x