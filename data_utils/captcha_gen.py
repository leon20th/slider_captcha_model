"""
基础验证码数据生成

用法:
  python -m data_utils.captcha_gen --raw_dir <原图目录> --out_dir <输出目录>
"""
import argparse
import os
import random

from PIL import Image, ImageDraw
import tqdm

from data_utils.anchor_gen import get_all_patch

IMG_W, IMG_H = 280, 161
CROP_N = 5       # 每张原图裁剪数
PATCH_N = 6      # 每次裁剪贴图数


def generate(raw_dir, out_dir):
    patchs = get_all_patch()
    os.makedirs(out_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f'[captcha_gen] {len(all_imgs)} raw images, generating {CROP_N * PATCH_N} samples each')

    count = 0
    for img_name in tqdm.tqdm(all_imgs):
        img_path = os.path.join(raw_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGBA")
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
                mask = Image.new("RGBA", cropped.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask)
                alpha = random.uniform(0.55, 0.75)
                for i in range(ph):
                    for j in range(pw):
                        if patch[i, j] > 0:
                            draw.point((rx + j, ry + i), fill=(255, 255, 255, int(alpha * 255)))
                            draw.point((j, ry + i), fill=cropped.getpixel((rx + j, ry + i)))
                sample = Image.alpha_composite(cropped, mask)
                fname = f'{img_name[:-4]}_c{ci}_p{pi}_xy_{rx}_{ry}.png'
                sample.save(os.path.join(out_dir, fname))
                count += 1

    print(f'[captcha_gen] done. Generated {count} samples in {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captcha training data')
    parser.add_argument('--raw_dir', required=True, help='Directory with raw background images')
    parser.add_argument('--out_dir', default='dataset/captcha1', help='Output directory')
    args = parser.parse_args()
    generate(args.raw_dir, args.out_dir)