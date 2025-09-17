"""
src/data.py
-----------

Data-loading and input-pipeline utilities for the Histopathologic project.

Features:
- Read labels csv and build file lists for train/test (using Kaggle input paths or local paths)
- Ensure train/val/test sizes are divisible by batch_size (trim if requested)
- Build ImageDataGenerator instances for train/val/test using augmentation config
- Create flow_from_dataframe generators (keras) and optionally convert them to tf.data.Dataset
  with unbatch/batch(drop_remainder=True) to be safe for multi-GPU MirroredStrategy.
- Returns generators/dataframes and recommended steps (steps_per_epoch, validation_steps, test_steps)

Usage:
    from src.data import prepare_data, load_config
    cfg = load_config("configs/hyperparams.yaml")
    outputs = prepare_data(cfg)
    train_gen = outputs['train_generator']
    train_steps = outputs['steps_per_epoch']
"""

import os
import math
import gc
from typing import Dict, Optional, Tuple, Any

import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------------------------------
# Helpers: load config
# -----------------------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and return as dict."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# -----------------------------------------------------------------------------
# Build file lists (train/test) from dataset folder and labels csv
# -----------------------------------------------------------------------------
def build_file_lists(train_dir: str, test_dir: str, labels_csv: str) -> Tuple[list, list, pd.DataFrame]:
    """
    Build lists:
      - image_list_train: list of [filepath, label]
      - image_list_test: list of [filepath]
      - df_labels: dataframe read from labels_csv

    Assumes train files are named <id>.tif and labels_csv has columns ['id','label'].
    """
    df_labels = pd.read_csv(labels_csv)
    label_dict = dict(zip(df_labels['id'].astype(str), df_labels['label']))

    image_list_train = []
    for fname in sorted(os.listdir(train_dir)):
        if not fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            continue
        image_id = os.path.splitext(fname)[0]
        filepath = os.path.join(train_dir, fname)
        label = label_dict.get(image_id)
        if label is None:
            # skip or raise warning if label missing
            continue
        image_list_train.append([filepath, int(label)])

    image_list_test = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            continue
        filepath = os.path.join(test_dir, fname)
        image_list_test.append([filepath])

    return image_list_train, image_list_test, df_labels

# -----------------------------------------------------------------------------
# Ensure splits divisible by batch_size
# -----------------------------------------------------------------------------
def make_splits(image_list_train: list,
                image_list_test: list,
                batch_size: int,
                len_train_ratio: float = 0.85,
                len_valid_ratio: float = 0.15,
                enforce_divisible: bool = True
                ) -> Tuple[list, list, list]:
    """
    Split the train list into training and validation lists, and prepare test list.
    If enforce_divisible=True then trims each split so that sizes are multiples of batch_size.
    Returns data_train, data_valid, data_test (lists compatible with creating DataFrames).
    """
    n_total = len(image_list_train)
    # compute sizes (floor to integer then make divisible by batch)
    train_count = int(len_train_ratio * n_total)
    valid_count = n_total - train_count

    if enforce_divisible:
        train_count = (train_count // batch_size) * batch_size
        valid_count = (valid_count // batch_size) * batch_size

    # guard: if after trimming valid_count==0, reduce train_count to make room
    if valid_count == 0 and n_total >= batch_size:
        # reserve at least one batch for validation
        valid_count = batch_size
        train_count = (n_total - valid_count) // batch_size * batch_size

    train_list = image_list_train[:train_count]
    valid_list = image_list_train[train_count:train_count + valid_count]
    test_count = (len(image_list_test) // batch_size) * batch_size
    test_list = image_list_test[:test_count]

    return train_list, valid_list, test_list

# -----------------------------------------------------------------------------
# Build ImageDataGenerators & flow_from_dataframe
# -----------------------------------------------------------------------------
def build_generators(df_train: pd.DataFrame,
                     df_valid: pd.DataFrame,
                     df_test: pd.DataFrame,
                     img_size: Tuple[int, int],
                     batch_size: int,
                     augmentation_cfg: Dict[str, Any],
                     seed: int = 42
                     ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator,
                              Any, Any, Any]:
    """
    Create ImageDataGenerator objects (train/val/test) and build flow_from_dataframe generators.
    Returns: trainGen, valGen, testGen, train_generator, val_generator, test_generator
    """
    # Train augmentations (rescale + augmentation if enabled)
    if augmentation_cfg.get('use_augmentation', True):
        train_gen = ImageDataGenerator(
            rescale=augmentation_cfg.get('rescale', 1./255.0),
            rotation_range=augmentation_cfg.get('rotation_range', 15),
            width_shift_range=augmentation_cfg.get('width_shift_range', 0.1),
            height_shift_range=augmentation_cfg.get('height_shift_range', 0.1),
            zoom_range=augmentation_cfg.get('zoom_range', 0.1),
            horizontal_flip=augmentation_cfg.get('horizontal_flip', True),
            fill_mode=augmentation_cfg.get('fill_mode', 'nearest')
        )
    else:
        train_gen = ImageDataGenerator(rescale=augmentation_cfg.get('rescale', 1./255.0))

    # Validation / Test: only rescale
    val_gen = ImageDataGenerator(rescale=augmentation_cfg.get('rescale', 1./255.0))
    test_gen = ImageDataGenerator(rescale=augmentation_cfg.get('rescale', 1./255.0))

    target_size = img_size

    # flow_from_dataframe expects the x_col to contain paths if full paths are provided
    train_generator = train_gen.flow_from_dataframe(
        dataframe=df_train,
        x_col='id',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    val_generator = val_gen.flow_from_dataframe(
        dataframe=df_valid,
        x_col='id',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    test_generator = test_gen.flow_from_dataframe(
        dataframe=df_test,
        x_col='id',
        y_col=None,
        target_size=target_size,
        class_mode=None,
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen, test_gen, train_generator, val_generator, test_generator

# -----------------------------------------------------------------------------
# Convert Keras generator -> tf.data.Dataset (optional)
# -----------------------------------------------------------------------------
def dataset_from_generator(generator,
                           img_height: int,
                           img_width: int,
                           batch_size: int,
                           has_labels: bool = True,
                           drop_remainder: bool = True) -> tf.data.Dataset:
    """
    Convert a Keras Sequence / Iterator (flow_from_dataframe) to tf.data.Dataset.
    Approach:
      - from_generator with batched output_signature (None, H, W, C) for images and (None,) for labels
      - unbatch() then re-batch with desired batch_size and drop_remainder to guarantee identical batch sizes
        across replicas (important for MirroredStrategy).
    Returns tf.data.Dataset yielding (images, labels) or images only (if has_labels=False).
    """
    if has_labels:
        output_signature = (
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    else:
        output_signature = tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=output_signature
    )

    # unbatch and re-batch with deterministic batch_size and drop_remainder
    dataset = dataset.unbatch().batch(batch_size, drop_remainder=drop_remainder)
    return dataset

# -----------------------------------------------------------------------------
# Main convenience function: prepare_data(cfg)
# -----------------------------------------------------------------------------
def prepare_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level function:
      - Reads paths from cfg (handles kaggle mode if set)
      - Builds file-lists and splits
      - Creates DataFrames expected by flow_from_dataframe (columns: 'id', 'label' or only 'id' for test)
      - Builds generators and optionally converts to tf.data.Dataset
    Returns a dict containing:
      'df_train','df_valid','df_test',
      'train_generator','val_generator','test_generator',
      'train_dataset_tf','val_dataset_tf','test_dataset_tf'  (may be None if not created),
      'steps_per_epoch','validation_steps','test_steps'
    """
    # 1) Resolve paths (support kaggle_mode)
    kaggle_mode = cfg.get('kaggle_mode', False)
    # if kaggle_mode is true and user provided base_dir default, change to the competition input path
    if kaggle_mode:
        # typical path for competition input on Kaggle
        base_input = cfg.get('data', {}).get('base_dir', '/kaggle/input/histopathologic-cancer-detection')
        train_dir = cfg.get('data', {}).get('train_dir', os.path.join(base_input, 'train'))
        test_dir = cfg.get('data', {}).get('test_dir', os.path.join(base_input, 'test'))
        labels_csv = cfg.get('data', {}).get('train_labels', os.path.join(base_input, 'train_labels.csv'))
    else:
        train_dir = cfg.get('data', {}).get('train_dir', cfg.get('data', {}).get('train_dir', './data/train'))
        test_dir = cfg.get('data', {}).get('test_dir', cfg.get('data', {}).get('test_dir', './data/test'))
        labels_csv = cfg.get('data', {}).get('train_labels', cfg.get('data', {}).get('train_labels', './data/train_labels.csv'))

    # 2) Read raw file lists
    print(f"[DATA] Reading files from: train_dir={train_dir}, test_dir={test_dir}, labels_csv={labels_csv}")
    image_list_train, image_list_test, df_labels = build_file_lists(train_dir, test_dir, labels_csv)

    # 3) Ensure split divisibility by batch_size
    batch_size = int(cfg.get('training', {}).get('batch_size', 32))
    len_train_ratio = cfg.get('data', {}).get('len_train_ratio', 0.85)
    len_valid_ratio = cfg.get('data', {}).get('len_valid_ratio', 0.15)
    ensure_divisible = bool(cfg.get('data', {}).get('ensure_divisible_by_batch', True))

    data_train, data_valid, data_test = make_splits(
        image_list_train=image_list_train,
        image_list_test=image_list_test,
        batch_size=batch_size,
        len_train_ratio=len_train_ratio,
        len_valid_ratio=len_valid_ratio,
        enforce_divisible=ensure_divisible
    )

    print(f"[DATA] After trimming: train={len(data_train)}, valid={len(data_valid)}, test={len(data_test)}")

    # 4) Build DataFrames expected by flow_from_dataframe
    df_train = pd.DataFrame(data_train, columns=['id', 'label'])
    df_valid = pd.DataFrame(data_valid, columns=['id', 'label'])
    df_test = pd.DataFrame(data_test, columns=['id'])

    # Convert labels to string (flow_from_dataframe in binary mode can accept strings)
    if 'label' in df_train.columns:
        df_train['label'] = df_train['label'].astype(str)
    if 'label' in df_valid.columns:
        df_valid['label'] = df_valid['label'].astype(str)

    # 5) Build generators
    img_h = int(cfg.get('image', {}).get('height', 96))
    img_w = int(cfg.get('image', {}).get('width', 96))
    augmentation_cfg = cfg.get('augmentation', {})
    # ensure rescale exists
    if 'rescale' not in augmentation_cfg:
        augmentation_cfg['rescale'] = cfg.get('image', {}).get('rescale', 1./255.0)

    _, _, _, train_generator, val_generator, test_generator = build_generators(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        img_size=(img_h, img_w),
        batch_size=batch_size,
        augmentation_cfg=augmentation_cfg,
        seed=int(cfg.get('seed', 42))
    )

    # 6) Optionally convert to tf.data.Dataset
    train_dataset_tf = None
    val_dataset_tf = None
    test_dataset_tf = None

    if cfg.get('training', {}).get('drop_remainder', True) or cfg.get('training', {}).get('prefetch', True):
        # create tf.data.Dataset versions (recommended for high-performance pipelines)
        try:
            train_dataset_tf = dataset_from_generator(train_generator, img_h, img_w, batch_size,
                                                      has_labels=True,
                                                      drop_remainder=bool(cfg.get('training', {}).get('drop_remainder', True)))
            val_dataset_tf = dataset_from_generator(val_generator, img_h, img_w, batch_size,
                                                    has_labels=True,
                                                    drop_remainder=bool(cfg.get('training', {}).get('drop_remainder', True)))
            test_dataset_tf = dataset_from_generator(test_generator, img_h, img_w, batch_size,
                                                     has_labels=False,
                                                     drop_remainder=bool(cfg.get('training', {}).get('drop_remainder', True)))

            if cfg.get('training', {}).get('prefetch', True) and cfg.get('training', {}).get('autotune', True):
                train_dataset_tf = train_dataset_tf.prefetch(tf.data.AUTOTUNE)
                val_dataset_tf = val_dataset_tf.prefetch(tf.data.AUTOTUNE)
                test_dataset_tf = test_dataset_tf.prefetch(tf.data.AUTOTUNE)
        except Exception as e:
            print("[DATA] Warning: failed to convert generators to tf.data.Dataset:", e)
            print("[DATA] Falling back to Keras generators directly.")
            train_dataset_tf = None
            val_dataset_tf = None
            test_dataset_tf = None

    # 7) Compute steps
    steps_per_epoch = len(train_generator)  # number of batches in generator
    validation_steps = len(val_generator)
    test_steps = len(df_test) // batch_size

    return {
        'cfg': cfg,
        'df_train': df_train,
        'df_valid': df_valid,
        'df_test': df_test,
        'train_generator': train_generator,
        'val_generator': val_generator,
        'test_generator': test_generator,
        'train_dataset_tf': train_dataset_tf,
        'val_dataset_tf': val_dataset_tf,
        'test_dataset_tf': test_dataset_tf,
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps,
        'test_steps': test_steps
    }

# -----------------------------------------------------------------------------
# If run directly, a small smoke-test using configs/hyperparams.yaml (if exists)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Basic quick test if executed stand-alone.
    import argparse
    parser = argparse.ArgumentParser(description="Test data.py pipeline")
    parser.add_argument('--config', type=str, default='../configs/hyperparams.yaml', help='path to yaml config')
    args = parser.parse_args()
    cfg = {}
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print("Could not load config:", e)
        # minimal sensible defaults
        cfg = {
            'kaggle_mode': True,
            'data': {'train_dir': '/kaggle/input/histopathologic-cancer-detection/train',
                     'test_dir': '/kaggle/input/histopathologic-cancer-detection/test',
                     'train_labels': '/kaggle/input/histopathologic-cancer-detection/train_labels.csv',
                     'ensure_divisible_by_batch': True},
            'training': {'batch_size': 32, 'drop_remainder': True, 'prefetch': True, 'autotune': True},
            'image': {'height': 96, 'width': 96, 'rescale': 1./255.0},
            'augmentation': {'use_augmentation': True, 'rescale': 1./255.0},
            'seed': 42
        }

    outputs = prepare_data(cfg)
    print("Prepared data. steps_per_epoch:", outputs['steps_per_epoch'], "validation_steps:", outputs['validation_steps'])
