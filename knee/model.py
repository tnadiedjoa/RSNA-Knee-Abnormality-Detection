"""2.5D multi-series model: a shared 2D backbone embeds every window, a small transformer mixes
the windows of all series, and one learned query per finding pools the tokens it needs
(e.g. sagittal windows for the ACL, coronal ones for the MCL)."""

import timm
import torch
import torch.nn as nn

from knee.config import LABELS
from knee.data import SEQ_CLASSES, SLOTS

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
                 heads: int = 8, dropout: float = 0.1, drop_path: float = 0.1, pool: str = "avg",
                 seq_emb: bool = False, aux_grades: int = 0, spatial_queries: bool = False, in_chans: int = 3):
        super().__init__()
        kwargs = {"drop_path_rate": drop_path} if drop_path else {}
        if in_chans != 3:  # stacked slices as channels (knee/data.py, stack): timm adapts the pretrained stem
            kwargs["in_chans"] = in_chans
        if backbone.startswith("vit_"):  # ViTs (e.g. DINOv2) were pretrained at another size: resample positions
            kwargs["dynamic_img_size"] = True
        self.backbone = make_backbone(backbone, pretrained, **kwargs)
        # pool inside each window: "avg" (the backbone's own head) or "avgmax" (+ max over the feature map,
        # through the head's norm when it has one: a small lesion is diluted by the average)
        self.pool_type = pool
        n_feats = self.backbone.num_features * (2 if pool == "avgmax" else 1)
        self.proj = nn.Sequential(nn.Linear(n_feats, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Parameter(torch.zeros(len(SLOTS), dim))
        # weighting of each slot's series (T1 / PD / T2 / ...), added to its tokens; zero at the start
        self.seq_emb = nn.Embedding(len(SEQ_CLASSES), dim) if seq_emb else None
        if self.seq_emb is not None:
            nn.init.zeros_(self.seq_emb.weight)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.queries = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        # training-only auxiliary head: report grade class of each finding from its pooled vector; scripts/train.py
        # drops "aux_" weights when saving, so checkpoints load into a plain KneeNet
        self.aux_grade_head = nn.Linear(dim, aux_grades) if aux_grades else None
        self.max_head = nn.Linear(dim, len(LABELS))  # max over windows: a focal lesion shows in few windows
        # optional spatial path: one query per finding attends over every feature-map cell of every window (the
        # window tokens above are average-pooled, which dilutes a small lesion); its logit is added to the others,
        # and its output layer starts at zero, so training starts from the plain model
        self.sp_queries = None
        if spatial_queries:
            c = self.backbone.num_features
            self.sp_norm, self.sp_proj = nn.LayerNorm(c), nn.Linear(c, dim)
            self.sp_pos = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))  # in-plane (row, column)
            self.sp_queries = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
            self.sp_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.sp_out_norm, self.sp_head = nn.LayerNorm(dim), nn.Linear(dim, 1)
            nn.init.zeros_(self.sp_head.weight)
            nn.init.zeros_(self.sp_head.bias)

    @classmethod
    def from_config(cls, cfg: dict) -> "KneeNet":
        """Inference network of a run (its config.json / config_folds*.json)."""
        return cls(cfg["backbone"], pretrained=False, dim=cfg["dim"], depth=cfg["depth"], drop_path=0,
                   pool=cfg.get("pool", "avg"), seq_emb=cfg.get("seq_emb", False),
                   spatial_queries=cfg.get("spatial_queries", False), in_chans=cfg.get("stack_slices", 0) or 3)

    def embed(self, imgs, keep_map: bool = False):
        """Pooled window features, and the backbone's feature map (N, C, h, w) when keep_map."""
        if self.pool_type == "avg" and not keep_map:
            return self.backbone(imgs), None
        fmap = self.backbone.forward_features(imgs)  # (N, C, h, w)
        pooled = self.backbone.forward_head(fmap, pre_logits=True)  # = self.backbone(imgs)
        if self.pool_type == "avgmax":
            mx = fmap.amax((2, 3), keepdim=True)
            norm = getattr(getattr(self.backbone, "head", None), "norm", None)  # ConvNeXt: LayerNorm2d after pooling
            if norm is not None:
                mx = norm(mx)
            pooled = torch.cat([pooled, mx.flatten(1)], 1)
        return pooled, (fmap if keep_map else None)

    def spatial_logits(self, fmap, pos, seq, pad, b, s, t):
        """Per-finding attention over all feature-map cells: fmap (B*S*T, C, h, w), pad (B, S*T) True = ignore."""
        _, _, h, w = fmap.shape
        z = self.sp_proj(self.sp_norm(fmap.permute(0, 2, 3, 1)))  # (N, h, w, D)
        rows, cols = torch.meshgrid(torch.linspace(0, 1, h, device=z.device), torch.linspace(0, 1, w, device=z.device),
                                    indexing="ij")
        z = z + self.sp_pos(torch.stack([rows, cols], -1).to(z.dtype))
        window = self.slot_emb[None, :, None] + self.pos_emb(pos.unsqueeze(-1))  # (B, S, T, D): series and depth
        if self.seq_emb is not None:
            window = window + self.seq_emb(seq)[:, :, None]
        z = (z.reshape(b, s, t, h, w, -1) + window[:, :, :, None, None].to(z.dtype)).flatten(1, 4)  # (B, S*T*h*w, D)
        key_pad = pad[:, :, None].expand(-1, -1, h * w).flatten(1)
        q = self.sp_queries.unsqueeze(0).expand(b, -1, -1).to(z.dtype)
        out, _ = self.sp_attn(q, z, z, key_padding_mask=key_pad)
        return self.sp_head(self.sp_out_norm(out)).squeeze(-1)  # (B, 12)

    def forward(self, x, pos, mask, seq=None, return_pooled=False):
        # x: (B, S, T, C, H, W) uint8 (C = 3 adjacent slices, or the stacked slices); pos: (B, S, T); mask: (B, S); seq: (B, S) SEQ_CLASSES indices
        b, s, t = x.shape[:3]
        imgs = x.flatten(0, 2).float()
        if imgs.shape[1] == 3:
            imgs = (imgs - MEAN.to(imgs.device)) / STD.to(imgs.device)
        else:  # stacked grey slices: the ImageNet statistics averaged over colours
            imgs = (imgs - MEAN.mean()) / STD.mean()
        win_feats, fmap = self.embed(imgs.contiguous(memory_format=torch.channels_last), keep_map=self.sp_queries is not None)
        feats = self.proj(win_feats).view(b, s, t, -1)
        tokens = feats + self.slot_emb[None, :, None] + self.pos_emb(pos.unsqueeze(-1))
        if self.seq_emb is not None:
            tokens = tokens + self.seq_emb(seq)[:, :, None]
        tokens = tokens.flatten(1, 2)  # (B, S*T, D)
        pad = ~mask.repeat_interleave(t, dim=1)  # True = ignore
        pad[pad.all(1), 0] = False  # a study with no readable series still gets a (blank) token
        tokens = self.encoder(tokens, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.pool(q, tokens, tokens, key_padding_mask=pad)
        pooled = self.norm(pooled)  # (B, 12, D)
        logits = self.head(pooled).squeeze(-1)  # (B, 12)
        max_logits = self.max_head(tokens).masked_fill(pad.unsqueeze(-1), -1e4).amax(1)
        out = logits + max_logits
        if self.sp_queries is not None:
            out = out + self.spatial_logits(fmap, pos, seq, pad, b, s, t)
        return (out, pooled) if return_pooled else out
