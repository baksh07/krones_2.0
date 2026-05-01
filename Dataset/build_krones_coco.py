"""
Convert Krones train_annotations.json into Detectron2-ready COCO splits.

What this does:
  - Drops 5 categories that don't help instance-seg training for the verdict rule:
      Roi, No fault, Embossing, Foam residue, Water drop
  - Renumbers the remaining 22 categories to contiguous IDs 1..22.
  - Stratifies a 90/10 image-level train/val split by each image's rarest
    kept class, so rare classes (Insect, Straw, Circlip, Foil, Liquid)
    are guaranteed to appear in val.
  - Preserves images that have zero kept-class annotations (clean bottles)
    so the verdict evaluator can test "no defect -> Reusable" later.

Outputs (under <repo>/Dataset/coco/):
  - krones_train.json
  - krones_val.json
  - krones_classes.json   (list of 22 names in the contiguous order)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
ANN_FILE = DATA_ROOT / 'train_annotations.json'
OUT_DIR = REPO_ROOT / 'Dataset' / 'coco'

DROP_CATEGORIES = {
    'Roi',
    'No fault',
    'Embossing',
    'Foam residue',
    'Water drop',
}


def main(seed: int = 42, val_frac: float = 0.10):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANN_FILE) as f:
        coco = json.load(f)

    old_cats = {c['id']: c['name'] for c in coco['categories']}
    keep_old_ids = sorted(
        cid for cid, name in old_cats.items() if name not in DROP_CATEGORIES
    )
    old_to_new = {old: new for new, old in enumerate(keep_old_ids, start=1)}
    new_categories = [
        {
            'id': old_to_new[old],
            'name': old_cats[old],
            'supercategory': old_cats[old],
        }
        for old in keep_old_ids
    ]
    new_id_to_name = {c['id']: c['name'] for c in new_categories}

    print(f'Kept {len(new_categories)} categories '
          f'(dropped {len(DROP_CATEGORIES)}: {sorted(DROP_CATEGORIES)}):')
    for c in new_categories:
        print(f'  {c["id"]:>2}: {c["name"]}')

    kept_anns = []
    for ann in coco['annotations']:
        if ann['category_id'] not in old_to_new:
            continue
        new_ann = dict(ann)
        new_ann['category_id'] = old_to_new[ann['category_id']]
        kept_anns.append(new_ann)
    print(f'\nAnnotations: {len(coco["annotations"])} -> '
          f'{len(kept_anns)} after dropping {len(DROP_CATEGORIES)} categories')

    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in kept_anns:
        anns_by_img[a['image_id']].append(a)

    class_image_counts: Counter = Counter()
    for anns in anns_by_img.values():
        for cid in {a['category_id'] for a in anns}:
            class_image_counts[cid] += 1

    print('\nImages per kept class (across full dataset):')
    for cid in sorted(new_id_to_name):
        print(f'  {new_id_to_name[cid]:35s} {class_image_counts.get(cid, 0):>6}')

    def stratum(img_id: int) -> int:
        anns = anns_by_img.get(img_id, [])
        if not anns:
            return -1
        cids = {a['category_id'] for a in anns}
        return min(cids, key=lambda c: class_image_counts[c])

    img_by_stratum: dict[int, list[int]] = defaultdict(list)
    for im in coco['images']:
        img_by_stratum[stratum(im['id'])].append(im['id'])

    rng = random.Random(seed)
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    for s, ids in img_by_stratum.items():
        rng.shuffle(ids)
        if len(ids) < 2:
            train_ids.update(ids)
            continue
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])

    print(f'\nSplit: train={len(train_ids)}, val={len(val_ids)} '
          f'(val_frac={val_frac}, seed={seed})')

    val_class_counts: Counter = Counter()
    for img_id in val_ids:
        for cid in {a['category_id'] for a in anns_by_img.get(img_id, [])}:
            val_class_counts[cid] += 1
    print('\nPer-class val image counts (! = zero):')
    for cid in sorted(new_id_to_name):
        n = val_class_counts.get(cid, 0)
        marker = '  (!)' if n == 0 else ''
        print(f'  {new_id_to_name[cid]:35s} {n:>4}{marker}')

    images_by_id = {im['id']: im for im in coco['images']}

    def write_split(name: str, ids: set[int]):
        out = {
            'info': coco.get('info', {}),
            'licenses': coco.get('licenses', []),
            'type': 'instances',
            'images': [images_by_id[i] for i in sorted(ids)],
            'annotations': [a for a in kept_anns if a['image_id'] in ids],
            'categories': new_categories,
        }
        for new_id, ann in enumerate(out['annotations'], start=1):
            ann['id'] = new_id
        path = OUT_DIR / f'krones_{name}.json'
        with open(path, 'w') as f:
            json.dump(out, f)
        n_with_anns = sum(1 for im in out['images'] if anns_by_img.get(im['id']))
        print(f'  {path.name}: {len(out["images"])} images '
              f'({n_with_anns} with kept anns), {len(out["annotations"])} anns')

    print('\nWriting splits:')
    write_split('train', train_ids)
    write_split('val', val_ids)

    classes_path = OUT_DIR / 'krones_classes.json'
    with open(classes_path, 'w') as f:
        json.dump([c['name'] for c in new_categories], f, indent=2)
    print(f'  {classes_path.name}: {len(new_categories)} classes (contiguous)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-frac', type=float, default=0.10)
    args = parser.parse_args()
    main(seed=args.seed, val_frac=args.val_frac)
