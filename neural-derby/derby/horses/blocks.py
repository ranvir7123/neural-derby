"""The building blocks ma'am taught in Module 3, written the way her chapter writes them.

Shapes are (batch, channels, height, width) unless a docstring says tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """conv -> batchnorm -> relu. The basic unit. Padding keeps height and width."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class PlainPair(nn.Module):
    """Two ConvBlocks in a row with NO shortcut. STAIRS is made of these."""

    def __init__(self, c: int):
        super().__init__()
        self.a = ConvBlock(c, c)
        self.b = ConvBlock(c, c)

    def forward(self, x):
        return self.b(self.a(x))


class ResidualBlock(nn.Module):
    """Two convs plus the input added back (x + F(x)). ELEVATOR is made of these."""

    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChannelAttention(nn.Module):
    """A volume knob per channel: squeeze to 1x1, score with two 1x1 convs, sigmoid, multiply."""

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(c // r, 4)
        self.score = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, c, 1))

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        scale = torch.sigmoid(self.score(avg) + self.score(mx))
        return x * scale


class SpatialAttention(nn.Module):
    """A where-mask: mean and max over channels, one 7x7 conv, sigmoid, multiply."""

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        mask = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * mask


class InceptionBlock(nn.Module):
    """Three lens sizes at once (1x1, 3x3, 5x5), glued along channels. Output has 3*branch channels."""

    def __init__(self, cin: int, branch: int):
        super().__init__()
        self.b1 = ConvBlock(cin, branch, k=1)
        self.b3 = ConvBlock(cin, branch, k=3)
        self.b5 = ConvBlock(cin, branch, k=5)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)


class DepthwiseSeparable(nn.Module):
    """One 3x3 filter per input channel (groups=cin), then a 1x1 conv to mix channels."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class PatchEmbed(nn.Module):
    """Cut the image into patch x patch squares, turn each into a token of size dim. Output: tokens (B, T, dim)."""

    def __init__(self, cin: int = 3, dim: int = 64, patch: int = 8, img: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, patch, stride=patch)
        n_tokens = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos


class TransformerEncoderBlock(nn.Module):
    """Self-attention then a small feed-forward, each with a skip. Tokens (B, T, dim) in and out.

    Attention is written with plain Linear layers so the multiply counter in stats.py can see them.
    """

    def __init__(self, dim: int = 64, heads: int = 4, mlp: int = 128):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        b, t, d = x.shape
        h = self.heads
        y = self.ln1(x)
        q = self.q(y).view(b, t, h, d // h).transpose(1, 2)
        k = self.k(y).view(b, t, h, d // h).transpose(1, 2)
        v = self.v(y).view(b, t, h, d // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (d // h) ** 0.5
        y = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.o(y)
        return x + self.mlp(self.ln2(x))
