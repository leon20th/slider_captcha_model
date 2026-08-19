"""
滑块验证码定位 — 推理脚本

用法:
  python predict.py captcha_image.png                    # 单张
  python predict.py --dir dataset/online_data_label/     # 批量
  python predict.py --dir dataset/online_data_label/ --verbose  # 批量 + 逐张打印
"""
import argparse
import os

import torch
from PIL import Image
import torchvision.transforms as transforms

from model import MiniYOLO

IMG_H, IMG_W = 282, 162
NORM_H, NORM_W = 282.0, 161.0

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'models', 'mini_yolo_online_best.pth')


def load_model(weights_path=None, device=None):
    """加载 MiniYOLO 模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS
    model = MiniYOLO()
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, device


def predict_image(model, img_path, device):
    """预测单张图片, 返回 (x, y) 像素坐标"""
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    x = out[0, 0].item() * NORM_H
    y = out[0, 1].item() * NORM_W
    return x, y


def main():
    parser = argparse.ArgumentParser(description='Slide captcha predictor')
    parser.add_argument('image', nargs='?', help='Single image path')
    parser.add_argument('--dir', help='Directory of images (batch mode)')
    parser.add_argument('--weights', default=None, help='Model weights path')
    parser.add_argument('--verbose', action='store_true', help='Print per-image results')
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        return

    model, device = load_model(args.weights)

    if args.image:
        x, y = predict_image(model, args.image, device)
        print(f'{x:.1f},{y:.1f}')

    if args.dir:
        files = sorted(f for f in os.listdir(args.dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        xs, ys = [], []
        for fname in files:
            x, y = predict_image(model, os.path.join(args.dir, fname), device)
            xs.append(x)
            ys.append(y)
            if args.verbose:
                # 如果文件名包含标签, 显示误差
                parts = fname.replace('.png', '').replace('.jpg', '').split('_')
                try:
                    true_x, true_y = int(parts[-2]), int(parts[-1])
                    err = abs(x - true_x) + abs(y - true_y)
                    print(f'{fname}: pred=({x:.1f},{y:.1f}) true=({true_x},{true_y}) err={err:.1f}px')
                except (ValueError, IndexError):
                    print(f'{fname}: pred=({x:.1f},{y:.1f})')

        if xs:
            print(f'\nBatch: {len(xs)} images, avg_x={sum(xs)/len(xs):.1f}, avg_y={sum(ys)/len(ys):.1f}')


if __name__ == '__main__':
    main()
