
---

## 🌲 Tree Counting and Object Detection Evaluation (Faster R-CNN, Multi-Fold, IoU ≥ 0.5)

### 🔍 Overview

This repository implements a **multi-fold evaluation pipeline** for object detection using Faster R-CNN, tailored for counting trees in high-density forest images **without red-dot annotations**.

The pipeline covers:

* 📁 **Cross-validation (5-fold)**
* 📦 Model prediction and CSV export per image
* 🧼 IoU-based filtering of predictions (≥ 0.5)
* 📊 Confusion matrix & metrics (precision, recall, F1-score)
* 📈 Aggregated final Excel report across folds
* 🌲 Per-fold tree count from `merged_predictions.csv`

---

### 📁 Directory Structure

```
E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset\
│
└── checkpoints_cross-without dotmap
    ├── fold{0..4}_best.pth        ← Checkpoints
    ├── result{0..4}\
    │   └── COMBINE_RESULT\
    │       ├── predicted_boxes\   ← Raw predictions per image
    │       ├── filtered_predicted_dots\ ← After IoU filtering
    │       ├── merged_predictions.csv
    │       ├── confusion_matrix.csv
    │       ├── metrics.xlsx
    │       ├── total_samples.csv
    │       └── result_matrix.csv
    │
    ├── final_report.xlsx          ← Multi-sheet summary (all folds)
    └── tree_count_report.csv      ← Total rows per merged_predictions.csv
```

---

### ⚙️ Key Functionalities

#### ✅ **1. Model Inference (Faster R-CNN)**

Each image in the test set is passed through a pretrained and fine-tuned Faster R-CNN model to generate bounding box predictions.

#### ✅ **2. IoU Filtering (≥ 0.5)**

Predictions are filtered using IoU with the best-matching ground-truth box per image:

```python
if iou ≥ 0.5 and higher than previous best → accept
```

#### ✅ **3. Metrics Computation**

* Confusion matrix (with class 0 zeroed)
* Per-class precision, recall, F1-score
* Macro average and accuracy
* Stored in both `.csv` and `.xlsx`

#### ✅ **4. Aggregated Reports**

A final Excel workbook is created with:

* `Metrics`: per-fold + average
* `Total Trees`: number of predicted rows (after filtering)
* `Confusion Matrix`: raw matrices
* `Total Samples`: total predictions from confusion matrix
* `Tree Count`: from `merged_predictions.csv` line count

---

### 📄 Output Reports

| File                     | Description                                   |
| ------------------------ | --------------------------------------------- |
| `metrics.xlsx`           | Per-fold evaluation metrics                   |
| `total_samples.csv`      | Number of predictions (from confusion matrix) |
| `merged_predictions.csv` | Final filtered predictions per fold           |
| `confusion_matrix.csv`   | Confusion matrix used for metrics             |
| `tree_count_report.csv`  | Number of prediction rows per fold            |
| `final_report.xlsx`      | Consolidated report across all folds          |

---

### 📊 Example Metrics (Macro Avg)

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.8421 |
| Recall    | 0.7893 |
| F1-Score  | 0.8125 |
| Accuracy  | 0.8604 |

> 📌 These are **placeholders**. Actual values are computed and stored in `final_report.xlsx`.

---

### 📦 Dependencies

* Python 3.8+
* PyTorch ≥ 1.13
* torchvision
* pandas, numpy, matplotlib
* openpyxl, xlsxwriter
* tqdm
* seaborn (optional)

---

### 🚀 Run Instructions

1. Place your trained checkpoints as:
   `checkpoints_cross-without dotmap/fold0_best.pth` ... `fold4_best.pth`

2. Run the script directly (no CLI args needed):

   ```bash
   python evaluate_multifold.py
   ```

3. Results will be saved to:
   `checkpoints_cross-without dotmap/final_report.xlsx`

---

### 🧠 Notes

* The pipeline auto-generates missing directories and handles empty or corrupt CSVs gracefully.
* All `.tif` image names are normalized during merging and filtering.
* Filtering logic ensures **only the best prediction per ground-truth box** is kept.

---

### 📬 Citation / Acknowledgment

If you use this pipeline or its components, please consider citing or referencing the original code authors or the research context it was built for.

