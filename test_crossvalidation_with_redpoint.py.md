حتماً! در ادامه یک گزارش کامل، استاندارد، و حرفه‌ای برای فایل `test_crossvalidation_with_redpoint.py` ارائه می‌دهم که مناسب بارگذاری در GitHub به‌عنوان مستندات رسمی این اسکریپت است.

---

## 📊 Project Report: Tree Species Detection & Classification using Faster R-CNN with Dot-based Filtering

### 📁 Script: `test_crossvalidation_with_redpoint.py`

**Last updated:** 2025-07-15

---

### 🔍 Overview

This repository evaluates a fine-tuned [Faster R-CNN](https://arxiv.org/abs/1506.01497) model on **aerial images of dense forests** to classify individual trees by species.

The evaluation pipeline leverages **center-point (dot) annotations** as ground truth and uses **point-in-box logic** for filtering predictions. Classification performance is assessed using standard metrics such as **precision**, **recall**, and **F1-score**, derived via a **confusion matrix**.

---

### 🌲 Task Description

The model performs:

* 🎯 **Tree detection**: Locating individual trees in dense canopy
* 🧬 **Species classification**: Assigning a species class (1–5) to each detected tree

Ground truth consists of:

* `test_labels.csv`: True bounding boxes and species labels
* `dots_csv/*.csv`: Center-point annotations of trees (cx, cy)

---

### 🧠 Pipeline Summary

| Step                          | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| 🖼️ Load test images          | Reads all test samples and their image filenames                           |
| 🤖 Load model                 | Loads a pretrained Faster R-CNN model with custom head (5 species classes) |
| 🔍 Predict bounding boxes     | Applies model to detect trees and predict class & confidence score         |
| 📍 Filter with dot-points     | Keeps only boxes that **contain a GT point** and are **closest** to it     |
| 📤 Export filtered boxes      | Saves final boxes to per-image CSVs and a merged global CSV                |
| 📊 Compute metrics            | Calculates confusion matrix, precision, recall, and F1-score               |
| 🎨 Visualize confusion matrix | Plots confusion matrix using `seaborn`                                     |

---

### 📁 Directory Structure

```bash
dataset/
├── images/                    # Aerial RGB images
├── test_labels.csv           # GT bounding boxes + labels
├── dots_csv/*.csv            # GT dot (cx, cy) annotations per image
├── checkpoints_cross/        # Saved model weights
├── COMBINE_RESULT/
│   ├── predicted_boxes/      # Raw model outputs
│   ├── filtered_predicted_dots/  # Filtered results using point-in-box logic
│   ├── merged_predictions.csv    # Combined predictions
│   └── classification_metrics.csv  # Precision, recall, F1
```

---

### ⚙️ Step-by-Step Workflow

1. **Image & Dot Loader**

   * Loads each test image
   * Reads corresponding center-points from `.csv` in `dots_csv`

2. **Model Inference**

   * Loads a `.pth` checkpoint from `checkpoints_cross`
   * For each image:

     * Detects bounding boxes, class labels, and confidence scores

3. **Point-in-Box Filtering**

   * For each `(cx, cy)` GT point:

     * Searches all predicted boxes that contain the point
     * Selects the box with **highest score and closest center**
   * Removes duplicate or overlapping predictions

4. **Export & Merge**

   * Saves filtered boxes to:

     * Per-image CSV files (`filtered_predicted_dots`)
     * Global merged CSV (`merged_predictions.csv`)

5. **Evaluation**

   * Compares predicted class vs. true class from `test_labels.csv`
   * Computes:

     * 📏 Precision (macro)
     * 🧲 Recall (macro)
     * 🧮 F1-score (macro)
   * Confusion matrix is computed and visualized

---

### 📊 Example Output

```txt
Precision: 0.90
Recall:    0.88
F1 Score:  0.89
```

<p align="center">
  <img src="assets/confusion_matrix_example.png" width="600"/>
</p>

---

### ✅ Key Characteristics

* 📍 **Dot-based filtering**: Boxes are retained **only** if they cover a GT point
* 🔁 **No use of IoU**: IoU is **not computed** in this script; matching is spatial (point-in-box)
* 📚 **Dual GTs used**:

  * Dots for localization
  * Boxes for classification
* 📈 **Confusion matrix** provides full insight into classification accuracy per species

---

### 💾 Output Files

| File                            | Description                                |
| ------------------------------- | ------------------------------------------ |
| `merged_predictions.csv`        | Final filtered predictions                 |
| `classification_metrics.csv`    | Precision, recall, F1                      |
| `filtered_predicted_dots/*.csv` | Per-image predictions after filtering      |
| `confusion matrix image`        | Heatmap-style image of prediction accuracy |

---

### 🔧 Requirements

```bash
torch
torchvision
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

---

### 📌 Notes

* Assumes test set is **fixed**, and model checkpoint is trained externally.
* Designed for **dense forests** where multiple trees occur in close proximity.
* Custom logic avoids overlapping predictions by choosing highest-score + shortest-distance box.

---

### 🚀 Suggested Improvements

* Add IoU filtering for stricter box validation
* Visualize per-class detection success
* Include PR curves (per class) and mAP metric

---

اگر خواستی نسخه فارسی هم آماده می‌کنم 🌱
می‌خوای تو ریپازیتوری یه `README.md` کامل هم برات تولید کنم؟
