# Histopathologic Image Classification

> **Kaggle project:** Training multiple CNNs on the *Histopathologic Cancer Detection* dataset using **2×T4 GPUs (MirroredStrategy)**.
> Original Kaggle notebook: [Use GPU T4x2 — Kaggle](https://www.kaggle.com/code/sepehrkh/use-gpu-t4x2-two-gpus-on-histopathologic-images)

---

## 📌 Project Goal

Binary classification of histopathologic image patches (96×96) — detect whether the central region contains **tumor tissue**.

This project demonstrates:

* Multi-GPU training with **MirroredStrategy**
* Comparing multiple CNN architectures (AlexNet, VGGNet, ResNet34, EfficientNet B0/B1/B2)
* Efficient pipelines with `tf.data` + `prefetch`
* Callbacks for learning rate scheduling & overfitting control

---

## 🗂 Repository Structure

```
Histopathologic-Image-Classification/
├── notebooks/         # Jupyter/Kaggle notebooks (summaries + originals)
│   ├── 01-exploration.ipynb
│   ├── 02-training.ipynb
│   └── kaggle-original.ipynb
├── src/               # Modular Python code (data, models, train, utils)
├── results/           # Training histories (.npz), predictions (.csv), plots (.png)
│   └── figures/
├── models/            # Trained models (.keras) — NOT pushed to GitHub
├── data/              # Raw dataset (via Kaggle API) — NOT pushed to GitHub
├── configs/           # Hyperparameters (.yaml/.yml)
├── logs/              # Execution logs
├── assets/            # Sample images for README visualization
├── README.md          # Project documentation
├── requirements.txt   # Dependencies
└── .gitignore         # Ignore large/temporary files
```

---

## 📊 Dataset

* **Source:** [Histopathologic Cancer Detection — Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)
* Structure:

  * `train/` (220,025 patches, labeled via `train_labels.csv`)
  * `test/` (57,458 patches)
  * `sample_submission.csv` (submission template)
  * Labels: (0 = no cancer, 1 = cancer)

🔑 **Important:** Number of samples in each split must be divisible by **batch size** (required for multi-GPU training).

---

## 🖼️ Sample Data

Example images from the dataset:

<p align="center">
  <img src="assets/samples/sample1.png" width="200"/>
  <img src="assets/samples/sample2.png" width="200"/>
  <img src="assets/samples/sample3.png" width="200"/>
</p>

---

## 📊 Training Curves

Training/Validation performance curves are stored in `results/figures/`.
Example:

<p align="center">
  <img src="results/figures/plots.png" width="600"/>
</p>

---

## 🧠 Architectures Tested

* **AlexNet** (custom implementation)
* **VGGNet** (custom, inspired by VGG16)
* **ResNet34** (custom / modified)
* **EfficientNet B0, B1, B2**

---

## ⚙ Training Configuration

* **Batch Size:** 64 (optimal for 2×T4 GPUs)
* **Learning Rate:** 0.001 (Adam optimizer)
* **Dropout:** 0.4 – 0.6
* **Total Train Samples:** 200,000
* **Image Size:** (96, 96)
* **Callbacks:**

  * `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5)`
  * `EarlyStopping(patience=3, restore_best_weights=True)`
  * `ModelCheckpoint(save_best_only=True)`
* **Strategy:** `tf.distribute.MirroredStrategy` with 2×T4 GPUs
* **Mixed Precision:** Disabled (caused NaN in loss)
* **Data Pipeline:** `tf.data` with `shuffle=True`, `drop_remainder=True`, `prefetch(AUTOTUNE)`

---

## ✅ Lessons Learned

1. Dataset size must be divisible by batch size → prevents GPU desync.
2. Always specify `steps` during `evaluate/predict` when using infinite datasets.
3. Use `K.clear_session()`, `gc.collect()`, and `tf.compat.v1.reset_default_graph()` before creating strategies/models.
4. Enable `prefetch + AUTOTUNE` → prevents GPU idle time.
5. Proper `LR scheduling` and `batch handling` → avoid overfitting & NaN losses.
6. Do **NOT** push raw datasets or large models to GitHub. Instead, use:

   * Kaggle API (`kaggle competitions download`)
   * Or cloud storage (Google Drive, AWS S3, Hugging Face Hub)

---

## 📚 Key Implementation Highlights

* Proper data partitioning (batch divisible) for **multi-GPU training**
* Conversion of generators to `tf.data.Dataset` → leverage **prefetch & AUTOTUNE**
* Consistent `steps_per_epoch`, `validation_steps`, and `test_steps`
* Storing training histories in `.npz` format for reproducibility
* Predictions stored in `.csv` with `id,label` columns

---

## 🚀 Sample Results

> Extracted from `.npz` training histories in `results/`.

| Model          | Val Accuracy | Val Loss |
| -------------- | ------------ | -------- |
| VGGNet         | \~93.4%      | \~0.211  |
| ResNet34       | \~86.4%      | \~0.354  |
| AlexNet        | \~59.5%      | \~0.68   |
| EfficientNetB2 | \~59.4%      | \~0.67   |

---

## 🔁 Reproduce This Project

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/Histopathologic-Image-Classification.git
cd Histopathologic-Image-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (via Kaggle API)
kaggle competitions download -c histopathologic-cancer-detection
unzip histopathologic-cancer-detection.zip -d data/

# 4. Download trained models & predictions
kaggle kernels output sepehrkh/use-gpu-t4x2-two-gpus-on-histopathologic-images -p ./models/

# 5. Train model with chosen config
python src/train.py --config configs/hyperparams.yaml

# 6. Evaluate & Predict
python src/train.py --config configs/hyperparams.yaml --predict
```

---

## 💽 Notes on Storage

* **Models (.keras)** are large → not pushed to GitHub.
* **Dataset (8 GB)** → must be downloaded directly from Kaggle competition.
* Only **scripts/configs/lightweight results** are kept in GitHub.

---

## 📜 License

**MIT** License

---
