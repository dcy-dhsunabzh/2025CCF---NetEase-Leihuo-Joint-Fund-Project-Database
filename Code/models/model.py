import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class LMK2Rig(nn.Module):
    def __init__(self, rig_dim: int = 45, pretrained: bool = True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        backbone.classifier[-1] = nn.Identity()  

        self.backbone = backbone
        self.feat_dim = 768    

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, rig_dim),  
            nn.Tanh()
        )
        
    def forward(self, img):
        f = self.backbone(img)   
        out   = self.head(f)
        return out