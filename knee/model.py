"""2.5D multi-series model: a shared 2D backbone embeds every window, a small transformer mixes
the windows of all series, and one learned query per finding pools the tokens it needs
(e.g. sagittal windows for the ACL, coronal ones for the MCL)."""

import timm
import torch
import torch.nn as nn

from knee.config import LABELS
from knee.data import SLOTS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


# ResNet-50 pretrained on RadImageNet (1.35 M CT/MR/ultrasound images), MIT licence, pinned revision
RADIMAGENET = ("Lab-Rasool/RadImageNet", "ResNet50.pt", "14460ee4c1276f6925611a63aa9a54a05d39eae0")


def make_backbone(name: str, pretrained: bool, **kwargs) -> nn.Module:
    if name != "radimagenet_resnet50":
        return timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
    model = timm.create_model("resnet50", pretrained=False, num_classes=0, **kwargs)
    if pretrained:  # its keys are backbone.<i>.* of an nn.Sequential over the torchvision ResNet-50 children
        from huggingface_hub import hf_hub_download
        sd = torch.load(hf_hub_download(RADIMAGENET[0], RADIMAGENET[1], revision=RADIMAGENET[2]), map_location="cpu")
        names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
        model.load_state_dict({names[k.split(".")[1]] + k[len("backbone.") + len(k.split(".")[1]):]: v for k, v in sd.items()})
    return model


class KneeNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, dim: int = 256, depth: int = 2,
                 heads: int = 8, dropout: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        kwargs = {"drop_path_rate": drop_path} if drop_path else {}
        if backbone.startswith("vit_"):  # ViTs (e.g. DINOv2) were pretrained at another size: resample positions
            kwargs["dynamic_img_size"] = True
        self.backbone = make_backbone(backbone, pretrained, **kwargs)
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Parameter(torch.zeros(len(SLOTS), dim))
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.queries = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        self.max_head = nn.Linear(dim, len(LABELS))  # max over windows: a focal lesion shows in few windows

    def forward(self, x, pos, mask):
        # x: (B, S, T, 3, H, W) uint8; pos: (B, S, T); mask: (B, S)
        b, s, t = x.shape[:3]
        imgs = x.flatten(0, 2).float()
        imgs = (imgs - MEAN.to(imgs.device)) / STD.to(imgs.device)
        feats = self.proj(self.backbone(imgs.contiguous(memory_format=torch.channels_last))).view(b, s, t, -1)
        tokens = feats + self.slot_emb[None, :, None] + self.pos_emb(pos.unsqueeze(-1))
        tokens = tokens.flatten(1, 2)  # (B, S*T, D)
        pad = ~mask.repeat_interleave(t, dim=1)  # True = ignore
        pad[pad.all(1), 0] = False  # a study with no readable series still gets a (blank) token
        tokens = self.encoder(tokens, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.pool(q, tokens, tokens, key_padding_mask=pad)
        logits = self.head(self.norm(pooled)).squeeze(-1)  # (B, 12)
        max_logits = self.max_head(tokens).masked_fill(pad.unsqueeze(-1), -1e4).amax(1)
        return logits + max_logits
