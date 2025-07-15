
---

## 🌲 Cross-Validated Tree Detection with Red-Point Filtering

### Accurate Bounding Box Extraction Using Faster R-CNN + GT Dot Alignment

---

### 🧠 Overview

This project evaluates a **Faster R-CNN** object detection model using a **dot-based filtering mechanism** and **5-class classification** (including background). It’s tailored for aerial imagery tree detection tasks and incorporates red point annotations (`cx, cy`) to refine predictions before metric evaluation.

**Workflow includes:**

* Running inference over test images
* Filtering boxes based on red dot locations
* Merging cleaned results
* Evaluating with a 5×5 confusion matrix and classification metrics (Precision, Recall, F1)

---


---

### 📁 Folder Structure

```
dataset2/
├── balanced_dataset/
│   ├── images/                   ← Test images
│   ├── test_labels.csv           ← Ground truth labels (used in evaluation)
│   ├── dots_csv/csv/             ← GT dot annotations (cx, cy)
│   ├── checkpoints_cross/
│   │   └── fold0_best.pth        ← Trained model weights
│   └── checkpoints_cross/COMBINE_RESULT/
│       ├── predicted_boxes/      ← Raw output from model
│       ├── filtered_predicted_dots/ ← Filtered boxes matched to points
│       ├── merged_predictions.csv   ← Final merged results
│       ├── classification_metrics.csv ← Final metrics
```

---

### 🔩 Dependencies

```bash
torch
torchvision
pandas
tqdm
seaborn
scikit-learn
matplotlib
```

---

### 🧪 How to Run

> ⚠️ All file paths are absolute and must be adapted for your machine.

```bash
python test_crossvalidation_with_redpoint.py
```

---

### 🔍 Step-by-Step Breakdown

#### 1️⃣ Load Test Set

* Loads test image filenames from `test_labels.csv`
* Returns tensors for model inference

#### 2️⃣ Run Inference

* Loads `fasterrcnn_resnet50_fpn`
* Loads weights from `fold0_best.pth`
* Predicts bounding boxes, class labels, and scores
* Saves raw outputs to `predicted_boxes/*.csv`

#### 3️⃣ Dot-Based Filtering

* Loads GT dot annotations from `dots_csv/`
* For each `(cx, cy)` point:

  * Selects the highest-scoring predicted box that contains the point
  * Applies proximity and score priority
* Saves matched boxes to `filtered_predicted_dots/*.csv`

#### 4️⃣ Merge Filtered CSVs

* Combines all filtered `.csv` files into `merged_predictions.csv`
* Fixes filenames and removes unused `score` column

#### 5️⃣ Compute Confusion Matrix & Metrics

* Compares `merged_predictions.csv` vs `test_labels.csv`
* Calculates:

  * ✅ **Precision**
  * ✅ **Recall**
  * ✅ **F1 Score**
  * ✅ **Confusion Matrix (5×5)**
* Outputs `classification_metrics.csv`

---

### 📊 Output Metrics

Example Confusion Matrix:

| GT \ Pred | 0 | 1 | 2 | 3 | 4 |
| --------- | - | - | - | - | - |
| **0**     | x | x | x | x | x |
| **1**     | x | x | x | x | x |
| ...       |   |   |   |   |   |

---

### ✅ Outputs Summary

| File                            | Description                             |
| ------------------------------- | --------------------------------------- |
| `predicted_boxes/*.csv`         | Raw predictions from Faster R-CNN       |
| `filtered_predicted_dots/*.csv` | Filtered boxes matched to GT points     |
| `merged_predictions.csv`        | Final cleaned prediction set            |
| `classification_metrics.csv`    | Precision, Recall, F1                   |
| Confusion Matrix Plot           | Visual display of classification result |

---

### 🧠 Evaluation Strategy

* **Red Dot Filtering**: Ensures predicted boxes are relevant by using GT location supervision.
* **Cross-validation weights**: Supports fold-based training-evaluation.
* **No duplicate predictions** thanks to spatial proximity logic.

---

### 💡 Future Improvements

* [ ] Auto-thresholding for score filtering
* [ ] Integration with COCO-style evaluators
* [ ] Class-wise metric reporting
* [ ] Export as annotation-ready format (Pascal VOC or COCO)



