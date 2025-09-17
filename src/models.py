# src/models.py
"""
Modular model definitions used in the Kaggle notebook:
- AlexNet
- VGGNet (custom)
- ResNet34 (custom light implementation with residual blocks)
- EfficientNet B0 / B1 / B2 wrappers (optionally pretrained)

This file provides:
- build_alexnet(...)
- build_vggnet(...)
- build_resnet34(...)
- build_efficientnet(model_type, ...)
- get_model(model_name, cfg, ...)
 
The functions return an uncompiled tf.keras.Model.
The caller (train.py) should compile the model inside strategy.scope().
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2

def _get_input_shape(img_height: int, img_width: int, channels: int) -> Tuple[int,int,int]:
    return (img_height, img_width, channels)

# -----------------------
# AlexNet
# -----------------------
def build_alexnet(img_height: int, img_width: int, channels: int = 3,
                  l2_reg: float = 1e-4, dropout: float = 0.5,
                  num_classes: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=_get_input_shape(img_height, img_width, channels))

    x = layers.Conv2D(96, 7, strides=2, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)

    x = layers.Conv2D(256, 5, strides=1, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)

    x = layers.Conv2D(384, 3, strides=1, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(384, 3, strides=1, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(256, 3, strides=1, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)

    # Binary classification final unit (sigmoid)
    out = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name='AlexNet_custom')
    return model

# -----------------------
# VGGNet (custom variant you used)
# -----------------------
def build_vggnet(img_height: int, img_width: int, channels: int = 3,
                 l2_reg: float = 1e-4, dense_units: int = 512, dropout: float = 0.4,
                 num_classes: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=_get_input_shape(img_height, img_width, channels))

    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(512, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    # If you want a second dense layer like older variant, you can add it; here we keep a compact classifier
    out = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name='VGGNet_custom')
    return model

# -----------------------
# ResNet34 (custom simple implementation using residual_block)
# -----------------------
def residual_block(x_in, filters, l2_reg: float = 1e-4):
    shortcut = x_in
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation=None,
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet34(img_height: int, img_width: int, channels: int = 3,
                   l2_reg: float = 1e-4, dense_units: int = 512, dropout: float = 0.5,
                   num_classes: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=_get_input_shape(img_height, img_width, channels))

    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    for _ in range(3):
        x = residual_block(x, 64, l2_reg)

    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    for _ in range(4):
        x = residual_block(x, 128, l2_reg)

    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    for _ in range(6):
        x = residual_block(x, 256, l2_reg)

    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    for _ in range(3):
        x = residual_block(x, 512, l2_reg)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name='ResNet34_custom')
    return model

# -----------------------
# EfficientNet wrappers (B0, B1, B2)
# -----------------------
def build_efficientnet(model_type: str,
                       img_height: int, img_width: int, channels: int = 3,
                       l2_reg: float = 1e-4, dense_units: int = 512, dropout: float = 0.4,
                       num_classes: int = 1, pretrained: bool = True) -> tf.keras.Model:
    model_type = model_type.upper()
    input_shape = _get_input_shape(img_height, img_width, channels)
    weights = "imagenet" if pretrained else None

    if model_type == "B0":
        base = EfficientNetB0(weights=weights, include_top=False, input_shape=input_shape)
    elif model_type == "B1":
        base = EfficientNetB1(weights=weights, include_top=False, input_shape=input_shape)
    elif model_type == "B2":
        base = EfficientNetB2(weights=weights, include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Invalid EfficientNet type. Choose 'B0','B1' or 'B2'.")

    # freeze base if pretrained (as in your notebook)
    if pretrained:
        base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=out, name=f"EfficientNet{model_type}")
    return model

# -----------------------
# Helper: factory function to pick a model by name
# -----------------------
def get_model(model_name: str, cfg: Optional[dict] = None,
              img_height: Optional[int] = None, img_width: Optional[int] = None,
              channels: int = 3, num_classes: int = 1,
              pretrained: bool = True) -> tf.keras.Model:
    """
    Returns an uncompiled model by model_name.
    - model_name: 'alexnet', 'vggnet', 'resnet34', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2'
    - cfg: optional dict (from hyperparams.yaml) to obtain l2/dropout/dense sizes
    - img_height/img_width: override if needed
    - pretrained: for EfficientNet models
    """
    # Defaults / fallback values (can be overridden from cfg)
    if cfg is None:
        l2_reg = 1e-4
        dropout = 0.5
        dense_units = 512
        ih = img_height or 96
        iw = img_width or 96
    else:
        l2_reg = float(cfg.get('regularization', {}).get('l2', 1e-4))
        dropout = float(cfg.get('regularization', {}).get('dropout', 0.5))
        dense_units = int(cfg.get('models', {}).get('dense_units', 512)) if cfg.get('models') else 512
        ih = img_height or cfg.get('image', {}).get('height', 96)
        iw = img_width or cfg.get('image', {}).get('width', 96)

    name = model_name.strip().lower()
    if name in ['alexnet']:
        return build_alexnet(ih, iw, channels, l2_reg=l2_reg, dropout=dropout, num_classes=num_classes)
    elif name in ['vgg', 'vggnet', 'vggnet_custom']:
        return build_vggnet(ih, iw, channels, l2_reg=l2_reg, dense_units=dense_units, dropout=dropout, num_classes=num_classes)
    elif name in ['resnet34', 'resnet']:
        return build_resnet34(ih, iw, channels, l2_reg=l2_reg, dense_units=dense_units, dropout=dropout, num_classes=num_classes)
    elif name in ['efficientnetb0', 'efficientnet_b0', 'efficient_b0', 'b0']:
        return build_efficientnet("B0", ih, iw, channels, l2_reg=l2_reg, dense_units=dense_units, dropout=dropout, num_classes=num_classes, pretrained=pretrained)
    elif name in ['efficientnetb1', 'efficientnet_b1', 'efficient_b1', 'b1']:
        return build_efficientnet("B1", ih, iw, channels, l2_reg=l2_reg, dense_units=dense_units, dropout=dropout, num_classes=num_classes, pretrained=pretrained)
    elif name in ['efficientnetb2', 'efficientnet_b2', 'efficient_b2', 'b2']:
        return build_efficientnet("B2", ih, iw, channels, l2_reg=l2_reg, dense_units=dense_units, dropout=dropout, num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
