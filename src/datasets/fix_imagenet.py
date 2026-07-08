#!/usr/bin/env python3
"""
Prepare Kaggle ImageNet (ILSVRC CLS-LOC) for DINOv3 / I-JEPA / torchvision.

1. Restructures flat val/ into class subdirectories using LOC_val_solution.csv
2. Generates labels.txt (wnid,class_name) from LOC_synset_mapping.txt

Idempotent: safe to re-run. Run from the dataset root (~/datasets/imagenet).
"""

import csv
import os
import shutil
import sys

ROOT = 'ILSVRC/Data/CLS-LOC'
VAL_DIR = os.path.join(ROOT, 'val')
TRAIN_DIR = os.path.join(ROOT, 'train')
TEST_DIR = os.path.join(ROOT, 'test')
VAL_CSV = 'LOC_val_solution.csv'
SYNSET_MAP = 'LOC_synset_mapping.txt'
LABELS_OUT = os.path.join(ROOT, 'labels.txt')

EXPECTED_CLASSES = 1000
EXPECTED_VAL = 50_000
EXPECTED_TRAIN = 1_281_167
EXPECTED_TEST = 100_000


def fail(msg):
    print(f'ERROR: {msg}')
    sys.exit(1)


def preflight():
    print('== Pre-flight checks ==')
    for p in (VAL_DIR, TRAIN_DIR, VAL_CSV, SYNSET_MAP):
        if not os.path.exists(p):
            fail(f'missing: {p} (run this from the dataset root?)')

    n_train_classes = sum(1 for e in os.scandir(TRAIN_DIR) if e.is_dir())
    if n_train_classes != EXPECTED_CLASSES:
        fail(f'train has {n_train_classes} class dirs, expected {EXPECTED_CLASSES}')
    print(f'  train: {n_train_classes} class dirs  OK')

    if not os.path.isdir(TEST_DIR):
        print('  test: missing (fine — unused by I-JEPA, optional for DINOv3)')
    else:
        n_test = sum(1 for e in os.scandir(TEST_DIR)
                     if e.name.endswith('.JPEG'))
        status = 'OK' if n_test == EXPECTED_TEST else f'expected {EXPECTED_TEST}!'
        print(f'  test: {n_test} images  {status}')


def restructure_val():
    print('== Restructuring val ==')
    # read csv first; validate before touching anything
    labels = {}
    with open(VAL_CSV) as f:
        next(f)  # header
        for row in csv.reader(f):
            if len(row) < 2:
                fail(f'malformed csv row: {row}')
            labels[row[0]] = row[1].split()[0]
    if len(labels) != EXPECTED_VAL:
        fail(f'{VAL_CSV} has {len(labels)} entries, expected {EXPECTED_VAL}')

    wnids = set(labels.values())
    if len(wnids) != EXPECTED_CLASSES:
        fail(f'csv references {len(wnids)} classes, expected {EXPECTED_CLASSES}')

    # sanity: csv wnids must match train dirs
    train_wnids = {e.name for e in os.scandir(TRAIN_DIR) if e.is_dir()}
    if wnids != train_wnids:
        fail(f'csv/train wnid mismatch: {len(wnids ^ train_wnids)} differ')

    moved = skipped = 0
    missing = []
    for img_id, wnid in labels.items():
        src = os.path.join(VAL_DIR, f'{img_id}.JPEG')
        dst = os.path.join(VAL_DIR, wnid, f'{img_id}.JPEG')
        if os.path.exists(dst):
            skipped += 1          # already moved (re-run)
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1
        else:
            missing.append(img_id)

    print(f'  moved {moved}, already in place {skipped}')
    if missing:
        fail(f'{len(missing)} images in csv not found on disk '
             f'(first few: {missing[:5]})')

    # no stray flat files left behind?
    leftovers = [e.name for e in os.scandir(VAL_DIR) if e.is_file()]
    if leftovers:
        fail(f'{len(leftovers)} unexpected flat files remain in val/ '
             f'(first few: {leftovers[:5]})')


def write_labels():
    print('== Writing labels.txt ==')
    rows = []
    with open(SYNSET_MAP) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, names = line.split(' ', 1)
            rows.append((wnid, names.split(',')[0].strip()))
    if len(rows) != EXPECTED_CLASSES:
        fail(f'{SYNSET_MAP} has {len(rows)} entries, expected {EXPECTED_CLASSES}')

    with open(LABELS_OUT, 'w') as out:
        for wnid, name in rows:
            out.write(f'{wnid},{name}\n')
    print(f'  wrote {LABELS_OUT} ({len(rows)} classes)')


def verify():
    print('== Final verification ==')
    val_classes = [e for e in os.scandir(VAL_DIR) if e.is_dir()]
    n_val_imgs = sum(
        1 for d in val_classes
        for e in os.scandir(d.path) if e.name.endswith('.JPEG'))
    print(f'  val: {len(val_classes)} class dirs, {n_val_imgs} images')
    if len(val_classes) != EXPECTED_CLASSES or n_val_imgs != EXPECTED_VAL:
        fail('val verification failed')

    with open(LABELS_OUT) as f:
        n_labels = sum(1 for line in f if line.strip())
    print(f'  labels.txt: {n_labels} rows')
    if n_labels != EXPECTED_CLASSES:
        fail('labels.txt verification failed')

    print('All checks passed. Dataset ready for DINOv3 / I-JEPA / torchvision.')


if __name__ == '__main__':
    preflight()
    restructure_val()
    write_labels()
    verify()