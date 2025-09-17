# src/train.py

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from src.data import dataset_from_generator
from src.utils import load_config  # load YAML
from src import models  # src/models.py

# -------------------------------
# 0) Load config
# -------------------------------
cfg = load_config("configs/hyperparams.yaml")
BATCH_SIZE = cfg["training"]["batch_size"]

# -------------------------------
# 1) Strategy (MirroredStrategy OR CPU)
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPUs detected: {[gpu.name for gpu in gpus]}")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("⚠ No GPU detected! Running on CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")

# -------------------------------
# 2) History Save/Load
# -------------------------------
def load_training_history(history_file):
    if os.path.exists(history_file):
        try:
            data = np.load(history_file)
            print(">>> Load Previous history ...")
            return (list(data['acc']) if 'acc' in data else [], 
                    list(data['val_acc']) if 'val_acc' in data else [],
                    list(data['loss']) if 'loss' in data else [],
                    list(data['val_loss']) if 'val_loss' in data else [])
        except Exception as e:
            print(f"⚠️ Error loading history: {e}")
            return [], [], [], []
    else:
        print("*** First run: Model training starts from scratch ... ")
        return [], [], [], []

def save_training_history(history_file, full_acc, full_val_acc, full_loss, full_val_loss):
    try:
        np.savez(history_file, acc=full_acc, val_acc=full_val_acc, loss=full_loss, val_loss=full_val_loss)
        print("Training history saved!")
    except Exception as e:
        print(f"⚠️ Error save history: {e}")

# -------------------------------
# 3) Training Function
# -------------------------------
def train_model(model_name, model_fn, trainDataset, valDataset, testDataset, df_test, epochs):
    # Clean session
    K.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()

    # Load history
    history_file = f"{model_name}_history.npz"
    prev_acc, prev_val_acc, prev_loss, prev_val_loss = load_training_history(history_file)

    # Build model inside strategy scope
    with strategy.scope():
        if os.path.exists(model_name):
            print(f">>> Loading the model from '{model_name}' to continue training...")
            model = tf.keras.models.load_model(model_name)  
        else:
            print(f"*** Building a new model and starting training ...")
            model = model_fn()  
            model.compile(
                loss=BinaryCrossentropy(),
                optimizer=Adam(learning_rate=cfg["optimizer"]["learning_rate"]),
                metrics=cfg["metrics"]
            )

    if not os.path.exists(model_name):
        model.summary()

    print(f"[INFO] Dataset sizes -> train: {len(trainDataset)}, valid: {len(valDataset)}, test: {len(testDataset)}")
    assert len(trainDataset) % strategy.num_replicas_in_sync == 0, "Train dataset not divisible by number of GPUs"

    # Convert generators to tf.data.Dataset
    trainDataset_tf = dataset_from_generator(trainDataset, has_labels=True).prefetch(tf.data.AUTOTUNE)
    valDataset_tf   = dataset_from_generator(valDataset, has_labels=True).prefetch(tf.data.AUTOTUNE)
    testDataset_tf  = dataset_from_generator(testDataset, has_labels=False).prefetch(tf.data.AUTOTUNE)

    # Callbacks
    reduce_lr = ReduceLROnPlateau(**cfg["callbacks"]["reduce_lr_on_plateau"])
    early_stopping = EarlyStopping(**cfg["callbacks"]["early_stopping"])
    checkpoint = ModelCheckpoint(model_name, **cfg["callbacks"]["model_checkpoint"])
    callbacks = [reduce_lr, early_stopping, checkpoint]

    # Check batch shape
    for sample_batch in trainDataset_tf.take(1):
        if isinstance(sample_batch, tuple):
            print("Batch shape (X):", sample_batch[0].shape)
            print("Batch shape (Y):", sample_batch[1].shape)
        else:
            print("Batch shape:", sample_batch.shape)

    # Steps
    steps_per_epoch = len(trainDataset)
    validation_steps = len(valDataset)
    test_steps = df_test.shape[0] // BATCH_SIZE  
    valid_test_count = test_steps * BATCH_SIZE

    # Train
    history = model.fit(
        trainDataset_tf,
        epochs=epochs,
        validation_data=valDataset_tf,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=cfg["misc"]["verbose"]
    )

    # Merge history
    full_acc = prev_acc + history.history['accuracy']
    full_val_acc = prev_val_acc + history.history['val_accuracy']
    full_loss = prev_loss + history.history['loss']
    full_val_loss = prev_val_loss + history.history['val_loss']
    save_training_history(history_file, full_acc, full_val_acc, full_loss, full_val_loss)

    # Evaluate & Predict
    if os.path.exists(model_name):
        print(f"✅ Model '{model_name}' exists. Loading...")
        best_model = tf.keras.models.load_model(model_name)
    else:
        print(f"❌ Model '{model_name}' NOT found!")
        return None
    
    loss, acc = best_model.evaluate(valDataset_tf, verbose=1, steps=validation_steps)
    print(f"✅ Model '{model_name}' -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    predictions = best_model.predict(testDataset_tf, verbose=1, steps=test_steps)
    binary_predictions = (predictions > 0.5).astype(int).flatten()

    ids = df_test['id'].values[:valid_test_count]
    output_df = pd.DataFrame({'id': ids, 'label': binary_predictions})
    output_filename = f"{model_name.replace('.keras', '')}_predictions.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"📁 Predictions saved to '{output_filename}'")

    return best_model

# -------------------------------
# 4) Plotting Functions
# -------------------------------
def plot_accuracy(model_name):
    history_file = f"{model_name}_history.npz"
    if not os.path.exists(history_file):
        print("⚠️No history was found for this model.")
        return
    data = np.load(history_file)
    plt.figure(figsize=(8, 6))
    plt.plot(data['acc'], label='Training Accuracy')
    plt.plot(data['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy - {model_name}')
    plt.legend(); plt.grid(); plt.show()

def plot_loss(model_name):
    history_file = f"{model_name}_history.npz"
    if not os.path.exists(history_file):
        print("⚠️No history was found for this model.")
        return
    data = np.load(history_file)
    plt.figure(figsize=(8, 6))
    plt.plot(data['loss'], label='Training Loss')
    plt.plot(data['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title(f'Loss - {model_name}')
    plt.legend(); plt.grid(); plt.show()

def compare_models_accuracy(models):
    plt.figure(figsize=(10, 6))
    for model_name in models:
        history_file = f"{model_name}_history.npz"
        if os.path.exists(history_file):
            data = np.load(history_file)
            plt.plot(data['acc'], label=f"{model_name} - Train")
            plt.plot(data['val_acc'], linestyle='dashed', label=f"{model_name} - Val")
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.title('Comparison of Models - Accuracy')
    plt.legend(); plt.grid(); plt.show()

def compare_models_loss(models):
    plt.figure(figsize=(10, 6))
    for model_name in models:
        history_file = f"{model_name}_history.npz"
        if os.path.exists(history_file):
            data = np.load(history_file)
            plt.plot(data['loss'], label=f"{model_name} - Train")
            plt.plot(data['val_loss'], linestyle='dashed', label=f"{model_name} - Val")
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title('Comparison of Models - Loss')
    plt.legend(); plt.grid(); plt.show()


____________________________________________

# ⏳ How to call each model

from src.models import AlexNet, VGGNet, ResNet34, EfficientNetB0

best_model = train_model("AlexNet.keras", AlexNet, trainDataset, valDataset, testDataset, df_test, epochs=10)
