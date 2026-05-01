import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
from pycocotools import mask as mask_utils


DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
ANN_FILE = DATA_ROOT / 'train_annotations.json'
IMG_DIR = DATA_ROOT / 'train_images'
OUT_DIR = Path('/home/dll0202/baksh/krones_2.0/Dataset/viz_out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_CATS = {'Roi'}


def polygons_to_mask(segmentation, h, w):
    rles = mask_utils.frPyObjects(segmentation, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


def main(seed=None):
    with open(ANN_FILE) as f:
        data = json.load(f)

    cats = {c['id']: c['name'] for c in data['categories']}
    anns_by_img = {}
    for a in data['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)

    rng = random.Random(seed)
    img_info = rng.choice(data['images'])
    img_path = IMG_DIR / img_info['file_name']
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_info['height'], img_info['width']
    anns = anns_by_img.get(img_info['id'], [])
    anns = [a for a in anns if cats[a['category_id']] not in EXCLUDE_CATS]

    print(f'Image: {img_info["file_name"]}  ({w}x{h})')
    print(f'Annotations (after excluding {EXCLUDE_CATS}): {len(anns)}')
    for a in anns:
        print(f'  - {cats[a["category_id"]]:30s} bbox={a["bbox"]}  area={a["area"]:.0f}')

    combined_mask = np.zeros((h, w), dtype=bool)
    per_class_masks = {}
    for a in anns:
        m = polygons_to_mask(a['segmentation'], h, w)
        combined_mask |= m
        per_class_masks.setdefault(cats[a['category_id']], np.zeros((h, w), bool))
        per_class_masks[cats[a['category_id']]] |= m

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img); axes[0].set_title('Image'); axes[0].axis('off')

    axes[1].imshow(combined_mask, cmap='gray')
    axes[1].set_title(f'Binary mask (all {len(anns)} anns)')
    axes[1].axis('off')

    axes[2].imshow(img)
    cmap = plt.get_cmap('tab20')
    patches, colors = [], []
    for i, a in enumerate(anns):
        for poly in a['segmentation']:
            pts = np.array(poly).reshape(-1, 2)
            patches.append(Polygon(pts, closed=True))
            colors.append(cmap(a['category_id'] % 20))
        x, y, bw, bh = a['bbox']
        axes[2].text(x, y - 4, cats[a['category_id']],
                     color='white', fontsize=8,
                     bbox=dict(facecolor=cmap(a['category_id'] % 20), alpha=0.7, pad=1))
    pc = PatchCollection(patches, facecolors=colors, edgecolors='white', linewidths=1.0, alpha=0.45)
    axes[2].add_collection(pc)
    axes[2].set_title('Image + polygons')
    axes[2].axis('off')

    img_id = img_info['id']
    out_overview = OUT_DIR / f'{img_id}_overview.png'
    out_mask = OUT_DIR / f'{img_id}_mask.png'
    plt.tight_layout(); plt.savefig(out_overview, dpi=120); plt.close(fig)
    Image.fromarray((combined_mask * 255).astype(np.uint8)).save(out_mask)
    print(f'Saved: {out_overview}')
    print(f'Saved: {out_mask}')


if __name__ == '__main__':
    main()
