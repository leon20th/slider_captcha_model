import random
import os
import numpy
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))

def get_all_patch(debug=False):
    patch_raw_width = 35
    patch_raw_height = 35

    patch_circle_rad = 13
    patch_circle_size = 11

    patch_circle = numpy.zeros((patch_circle_rad, patch_circle_rad), dtype=numpy.int16)
    patch_center = patch_circle_rad // 2
    # 填充圆为1
    for i in range(patch_circle_rad):
        for j in range(patch_circle_rad):
            if (i - patch_center) ** 2 + (j - patch_center) ** 2 <= (patch_circle_rad // 2) ** 2:
                patch_circle[i, j] = 1

    def gen_patch(left, right, top, bottom):
        patch_width = patch_raw_width + (max(left, 0) + max(right, 0)) * patch_circle_size
        patch_height = patch_raw_height + (max(top, 0) + max(bottom, 0)) * patch_circle_size
        patch = numpy.zeros((patch_height, patch_width), dtype=numpy.int16)
        x = 0 + max(left, 0) * patch_circle_size
        y = 0 + max(top, 0) * patch_circle_size
        patch[y:y+patch_raw_height, x:x+patch_raw_width] = 1

        left_patch = (patch_circle * left)[:, :-2][:, ::left] if left != 0 else None
        right_patch = (patch_circle * right)[:, 2:][:, ::right] if right != 0 else None
        top_patch = (patch_circle * top)[:-2, :][::top, :] if top != 0 else None
        bottom_patch = (patch_circle * bottom)[2:, :][::bottom, :] if bottom != 0 else None

        if left_patch is not None:
            lp_h, lp_w = left_patch.shape
            patch_x = 0
            patch_y = y + (patch_raw_height // 2) - (patch_circle_rad // 2)
            patch[patch_y:patch_y+lp_h, patch_x:patch_x+lp_w] += left_patch
        if right_patch is not None:
            rp_h, rp_w = right_patch.shape
            patch_x = patch_width - rp_w
            patch_y = y + (patch_raw_height // 2) - (patch_circle_rad // 2)
            patch[patch_y:patch_y+rp_h, patch_x:patch_x+rp_w] += right_patch
        if top_patch is not None:
            tp_h, tp_w = top_patch.shape
            patch_x = x + (patch_raw_width // 2) - (patch_circle_rad // 2)
            patch_y = 0
            patch[patch_y:tp_h, patch_x:patch_x+tp_w] += top_patch
        if bottom_patch is not None:
            bp_h, bp_w = bottom_patch.shape
            patch_x = x + (patch_raw_width // 2) - (patch_circle_rad // 2)
            patch_y = patch_height - bp_h
            patch[patch_y:patch_y+bp_h, patch_x:patch_x+bp_w] += bottom_patch
        return patch

    path = os.path.join(_HERE, 'patchs')
    loop_set = (-1, 0, 1)
    all_patch = []
    for left in loop_set:
        for right in loop_set:
            for top in loop_set:
                for bottom in loop_set:
                    patch = gen_patch(left, right, top, bottom)
                    all_patch.append((patch, (left, right, top, bottom)))
                    if debug:
                        height, width = patch.shape
                        rgb_patch = numpy.zeros((height, width, 3), dtype=numpy.uint8)
                        rgb_patch[patch == 1] = [255, 0, 0]  # 红色
                        rgb_patch[patch == 0] = [255, 255, 0]  # 黄色
                        # 生成图片
                        filename = f'patch_l{left}_r{right}_t{top}_b{bottom}.png'
                        image = Image.fromarray(rgb_patch, mode='RGB')
                        image.save(f'{path}/{filename}')
    return all_patch
#get_all_patch()