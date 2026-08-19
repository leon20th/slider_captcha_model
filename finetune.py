"""
小量数据微调脚本

用法:
  python finetune.py --data_dir samples/ --epochs 200 --lr 5e-4
"""
import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms

from model import MiniYOLO

IMG_H, IMG_W = 282, 162
NORM_H, NORM_W = 282.0, 161.0
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, 'models', 'mini_yolo_online_best.pth')


class Augment:
    def __call__(self, img):
        if random.random() < 0.7:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.2))
        if random.random() < 0.7:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        return img


class CaptchaDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.resize = transforms.Resize((IMG_H, IMG_W))
        self.to_tensor = transforms.ToTensor()
        self.augmenter = Augment() if augment else None
        self.files = []
        for name in os.listdir(data_dir):
            if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            parts = name.replace('.png', '').replace('.jpg', '').split('_')
            try:
                lb_x, lb_y = int(parts[-2]), int(parts[-1])
                self.files.append((os.path.join(data_dir, name), lb_x, lb_y))
            except (ValueError, IndexError):
                continue

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, lb_x, lb_y = self.files[idx]
        img = Image.open(path).convert('RGB')
        img = self.resize(img)
        if self.augmenter and random.random() < 0.8:
            img = self.augmenter(img)
        inp = self.to_tensor(img)
        tgt = torch.tensor([lb_x / NORM_H, lb_y / NORM_W], dtype=torch.float32)
        return inp, tgt


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    total_err = 0.0
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    n = 0
    for images, targets in loader:
        images = images.to(device)
        preds = model(images).cpu()
        total_err += (preds[:, 0] * NORM_H - targets[:, 0] * NORM_H).abs().sum().item()
        total_err += (preds[:, 1] * NORM_W - targets[:, 1] * NORM_W).abs().sum().item()
        n += images.size(0)
    return total_err / n


def main():
    parser = argparse.ArgumentParser(description='Fine-tune slider captcha model')
    parser.add_argument('--data_dir', required=True, help='Labeled images (filename contains x_y)')
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS, help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save', default=os.path.join(HERE, 'models', 'mini_yolo_finetuned.pth'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained
    model = MiniYOLO()
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    # Datasets
    ds_train = CaptchaDataset(args.data_dir, augment=True)
    ds_eval = CaptchaDataset(args.data_dir, augment=False)
    print(f'[data] {len(ds_train)} labeled images from {args.data_dir}')

    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_err = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            loss = criterion(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            err = evaluate(model, ds_eval, device)
            if err < best_err:
                best_err = err
                os.makedirs(os.path.dirname(args.save), exist_ok=True)
                torch.save({'model_state_dict': model.state_dict()}, args.save)
            print(f'[epoch {epoch+1}/{args.epochs}] avg_err={err:.2f}px  best={best_err:.2f}px')

    print(f'\nDone. Best: {best_err:.2f}px -> {args.save}')


if __name__ == '__main__':
    main()
