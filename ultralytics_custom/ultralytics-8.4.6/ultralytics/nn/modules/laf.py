# ultralytics/nn/modules/laf.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1) EMA: Efficient Multi-scale Attention (common lightweight impl)
# -------------------------
class EMA(nn.Module):
    """
    Efficient Multi-scale Attention (EMA)
    - Lightweight attention used in many mobile/efficient backbones.
    - Designed to be channel-friendly and cheap.
    """
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.groups = max(1, min(groups, channels))
        assert channels % self.groups == 0, "EMA: channels must be divisible by groups"

        c_g = channels // self.groups
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c_g * 2, c_g, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(c_g, c_g, kernel_size=3, stride=1, padding=1, groups=c_g, bias=False)

        self.gn = nn.GroupNorm(num_groups=1, num_channels=c_g)  # per-group norm
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        xg = x.view(b * g, c // g, h, w)

        x_h = self.pool_h(xg)  # (bg, cg, h, 1)
        x_w = self.pool_w(xg).transpose(2, 3)  # (bg, cg, w, 1) -> transpose to align

        # H != W 대응
        x_w = F.interpolate(x_w, size=(x_h.shape[2], 1), mode="nearest")

        # concat → 1x1 conv
        y = self.conv1(torch.cat([x_h, x_w], dim=1))
        y = self.gn(y)
        y = self.sigmoid(y)

        # local refinement
        y_local = self.conv3(xg)
        out = xg * y + y_local
        return out.view(b, c, h, w)

# -------------------------
# 2) Partial Convolution (PConv) - FasterNet style
# -------------------------
class PartialConv(nn.Module):
    """
    PartialConv:
    - Apply 3x3 conv only to a fraction of channels, keep the rest as identity.
    """
    def __init__(self, channels: int, n_div: int = 4):
        super().__init__()
        self.channels = channels
        self.n_div = max(1, n_div)
        self.c_part = channels // self.n_div
        self.conv = nn.Conv2d(self.c_part, self.c_part, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c_part, self.channels - self.c_part], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)

# -------------------------
# 3) FasterBlock + EMA (논문 FasterBlock-EMA 개념 대응)
#    - PConv -> PW(1x1) expand -> PW(1x1) project + residual
#    - EMA optional
# -------------------------
class FasterBlockEMA(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        use_ema: bool = True,
        ema_groups: int = 8,
        act: str = "silu",
    ):
        super().__init__()
        assert c1 == c2, "FasterBlockEMA: set c1==c2 for clean residual; adjust YAML channels accordingly"
        self.c = c1

        hidden = int(self.c * mlp_ratio)

        self.pconv = PartialConv(self.c, n_div=n_div)

        self.pw1 = nn.Conv2d(self.c, hidden, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        self.pw2 = nn.Conv2d(hidden, self.c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.c)

        if act.lower() == "silu":
            self.act = nn.SiLU()
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU()

        self.ema = EMA(self.c, groups=ema_groups) if use_ema else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.pconv(x)
        x = self.act(self.bn1(self.pw1(x)))
        x = self.bn2(self.pw2(x))
        x = self.ema(x)
        return identity + x

# -------------------------
# 4) SCConv (SRU + CRU) - 표준 SCConv 계열 구현
# -------------------------
class SRU(nn.Module):
    """Spatial Reconstruction Unit (SRU)"""
    def __init__(self, channels: int, groups: int = 16, gate_t: float = 0.5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.gate_t = gate_t

    def forward(self, x):
        # gate based on normalized feature
        gn_x = self.gn(x)
        gate = torch.sigmoid(gn_x)
        x1 = x * (gate >= self.gate_t)
        x2 = x * (gate < self.gate_t)
        return x1 + x2  # simple reconstruction

class CRU(nn.Module):
    """Channel Reconstruction Unit (CRU)"""
    def __init__(self, channels: int, alpha: float = 0.5, squeeze: int = 2, groups: int = 2):
        super().__init__()
        c_up = int(channels * alpha)
        c_low = channels - c_up
        self.c_up, self.c_low = c_up, c_low

        self.squeeze_up = nn.Conv2d(c_up, c_up // squeeze, 1, 1, 0, bias=False)
        self.squeeze_low = nn.Conv2d(c_low, c_low // squeeze, 1, 1, 0, bias=False)

        # grouped conv + pointwise conv branches
        self.gwc = nn.Conv2d(c_up // squeeze, channels, 3, 1, 1, groups=groups, bias=False)
        self.pwc = nn.Conv2d(c_low // squeeze, channels, 1, 1, 0, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_up, x_low = torch.split(x, [self.c_up, self.c_low], dim=1)
        up = self.squeeze_up(x_up)
        low = self.squeeze_low(x_low)

        y1 = self.gwc(up)
        y2 = self.pwc(low)
        y = y1 + y2

        # channel-wise reweighting
        w = F.adaptive_avg_pool2d(y, (1, 1))
        w = self.softmax(w)
        return y * w

class SCConv(nn.Module):
    """SCConv = SRU + CRU"""
    def __init__(self, c1: int, c2: int, sru_groups: int = 16, cru_groups: int = 2, alpha: float = 0.5):
        super().__init__()
        assert c1 == c2, "SCConv: designed for same in/out channels"
        self.sru = SRU(c2, groups=sru_groups)
        self.cru = CRU(c2, alpha=alpha, groups=cru_groups)

    def forward(self, x):
        x = self.sru(x)
        x = self.cru(x)
        return x

# -------------------------
# 5) C2fSCConv: C2f 내부 bottleneck 대신 SCConv 기반 블록 사용 (논문 C2f-SCConv 대응)
# -------------------------
class SCConvBlock(nn.Module):
    def __init__(self, c: int, sru_groups: int = 16, cru_groups: int = 2, alpha: float = 0.5):
        super().__init__()
        self.sc = SCConv(c, c, sru_groups=sru_groups, cru_groups=cru_groups, alpha=alpha)

    def forward(self, x):
        return self.sc(x)

class C2fSCConv(nn.Module):
    """
    Ultralytics C2f-like module:
    - split -> repeated blocks -> concat -> fuse conv
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        e: float = 0.5,
        sru_groups: int = 16,
        cru_groups: int = 2,
        alpha: float = 0.5,
    ):
        super().__init__()
        assert c2 > 0
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * c_, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()

        self.m = nn.ModuleList([SCConvBlock(c_, sru_groups=sru_groups, cru_groups=cru_groups, alpha=alpha) for _ in range(n)])

        self.cv2 = nn.Conv2d((2 + n) * c_, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        y1, y2 = y.chunk(2, 1)
        outs = [y1, y2]
        for block in self.m:
            y2 = block(y2)
            outs.append(y2)
        out = torch.cat(outs, 1)
        out = self.bn2(self.cv2(out))
        return self.act(out)
