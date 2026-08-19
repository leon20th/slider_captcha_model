import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Conv + BN + LeakyReLU — 与原 YOLOv1 保持一致"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """1×1 reduce → 3×3 process → residual (YOLO 风格 bottleneck)"""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            CNNBlock(ch, ch // 2, 1),
            CNNBlock(ch // 2, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class MiniYOLO(nn.Module):
    """
    保持 YOLO 架构风格的轻量模型:
      - CNNBlock (Conv+BN+LeakyReLU) 与 teacher 一致
      - 1×1 + 3×3 bottleneck / ResBlock
      - 多阶段 backbone (4 个 stage, stride 逐步降采样)
      - GAP + 小 FC head 替代原来 62M 参数的全连接层

    输入: (B, 3, 282, 162)  → 输出: (B, 2) normalized (x, y)
    ~50K params, teacher 的 1/2500
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Stage 1: /2  (3→16)
            CNNBlock(3, 16, 3, stride=2, padding=1),
            # Stage 2: /4  (16→32)
            CNNBlock(16, 32, 3, stride=2, padding=1),
            ResBlock(32),
            # Stage 3: /8  (32→64)
            CNNBlock(32, 64, 3, stride=2, padding=1),
            ResBlock(64),
            ResBlock(64),
            # Stage 4: /16  (64→128)
            CNNBlock(64, 128, 3, stride=2, padding=1),
            ResBlock(128),
            ResBlock(128),
            # Head convs — 类似 YOLO 最后的 3×3 卷积
            CNNBlock(128, 128, 3, padding=1),
            CNNBlock(128, 64, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.fc(x)


if __name__ == "__main__":
    model = MiniYOLO()
    total = sum(p.numel() for p in model.parameters())
    print(f"MiniYOLO params: {total:,}")
    x = torch.randn(4, 3, 282, 162)
    out = model(x)
    print(f"Output: {out.shape}, sample: {out[0].tolist()}")
