"""
增强版验证码数据生成 — 模拟线上数据分布

在原 captcha_gen 流程基础上增加:
  1. 亮度/对比度/色调抖动 (向线上分布靠拢)
  2. 高斯噪声 (模拟真实截图压缩伪影)
  3. 随机 JPEG 压缩质量

用法:
  python -m data_utils.captcha_gen_aug --raw_dir <原图目录> --out_dir <输出目录>
"""
import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import tqdm

from data_utils.anchor_gen import get_all_patch

patchs = get_all_patch()

IMG_W, IMG_H = 280, 161
CROP_N = 5       # 每张原图裁剪数
PATCH_N = 6      # 每次裁剪贴图数


def color_jitter(img):
    """
    颜色增强: 向线上分布 (偏冷、低亮度、高对比) 靠拢
    """
    # 亮度: 线上偏暗 → 0.6~1.1
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.1))
    # 对比度: 线上 std 更高 → 1.0~1.5
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0, 1.5))
    # 色彩饱和度
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img


def add_noise(img, sigma=5):
    """添加高斯噪声模拟真实截图"""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_compress(img, quality_range=(60, 95)):
    """模拟 JPEG 压缩"""
    import io
    q = random.randint(*quality_range)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def generate(raw_dir, out_dir):
    patchs = get_all_patch()
    os.makedirs(out_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f'[captcha2_gen] {len(all_imgs)} raw images, generating {CROP_N * PATCH_N} samples each')

    count = 0
    for img_name in tqdm.tqdm(all_imgs):
        img_path = os.path.join(IMG_DIR, img_name)
        try:
            image = Image.open(img_path).convert('RGBA')
        except Exception:
            continue
        w, h = image.size
        if w < IMG_W or h < IMG_H:
            continue

        for ci in range(CROP_N):
            x = random.randint(0, w - IMG_W)
            y = random.randint(0, h - IMG_H)
            cropped = image.crop((x, y, x + IMG_W, y + IMG_H))

            for pi in range(PATCH_N):
                patch = patchs[random.randint(0, len(patchs) - 1)][0]
                ph, pw = patch.shape
                rx = max(pw // 3 * 2, random.randint(0, IMG_W - pw))
                ry = random.randint(0, IMG_H - ph)

                mask = Image.new('RGBA', cropped.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask)
                alpha = random.uniform(0.55, 0.75)

                for i in range(ph):
                    for j in range(pw):
                        if patch[i, j] > 0:
                            px = cropped.getpixel((rx + j, ry + i))
                            draw.point((rx + j, ry + i),
                                       fill=(255, 255, 255, int(alpha * 255)))
                            draw.point((j, ry + i), fill=px)

                sample = Image.alpha_composite(cropped, mask).convert('RGB')

                # ── 颜色增强: 向线上分布靠拢 ──
                sample = color_jitter(sample)
                # 高斯噪声
                if random.random() < 0.5:
                    sample = add_noise(sample, sigma=random.uniform(3, 8))
                # JPEG 压缩 (模拟截图)
                if random.random() < 0.3:
                    sample = jpeg_compress(sample)

                fname = f'{img_name[:-4]}_c{ci}_p{pi}_xy_{rx}_{ry}.png'
                sample.save(os.path.join(out_dir, fname))
                count += 1

    print(f'[captcha2_gen] done. Generated {count} samples in {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmented captcha training data')
    parser.add_argument('--raw_dir', required=True, help='Directory with raw background images')
    parser.add_argument('--out_dir', default='dataset/captcha2', help='Output directory')
    args = parser.parse_args()
    generate(args.raw_dir, args.out_dir)
