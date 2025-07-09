# Full Evaluation Report: `00_COMPLETE_TEST.py`

## 🔖 Title:

**Complete Evaluation Pipeline for Tree Species Detection Using Faster R-CNN and Synthetic Dataset**

---

## 🌟 Purpose

This script performs a full evaluation of a trained Faster R-CNN model on a synthetic dataset for tree species identification. It covers the following:

* Running inference over a test image set
* Filtering predictions using reference dot coordinates
* Aggregating prediction outputs
* Computing classification metrics (precision, recall, F1-score)
* Plotting and saving the confusion matrix

---

## 📅 Workflow Steps

### 1. **Load Trained Model**

* Model is loaded from a saved file (e.g., `best_model_7.pth`)
* Uses PyTorch's Faster R-CNN with a custom number of output classes

### 2. **Define Paths**

* Input images: `images&DOT&dot/`
* Dot coordinate files: `dots_csv/`
* Output results: `COMBINE_RESULT/`

### 3. **Run Predictions**

* The model is run on each image
* Predicted bounding boxes, labels, and scores are extracted
* Raw predictions saved to: `predicted_boxes/*.csv`

### 4. **Filter Predictions by Dots**

* For each reference point (cx, cy):

  * The closest predicted box is selected
  * Selection is based on shortest Euclidean distance from box center
  * The selected box must exceed a score threshold
* Filtered results saved to: `filtered_predicted_dots/*.csv`

### 5. **Merge All Predictions**

* All filtered CSVs are concatenated into `merged_predictions.csv`

### 6. **Compute Metrics**

* Compare `merged_predictions.csv` against `test_labels.csv`
* Metrics computed:

  * Precision (macro)
  * Recall (macro)
  * F1-score (macro)
* Confusion matrix (5x5) is also calculated

### 7. **Save Outputs**

* `classification_metrics.csv`: Precision, Recall, F1
* `confusion_matrix_raw.csv`: Raw matrix values
* `confusion_matrix.png`: Visual matrix plot

---

## 📂 Input File Requirements

| File/Folder                   | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| `test_labels.csv`             | Ground-truth bounding boxes for test set                  |
| `dots_csv/`                   | CSV files containing dot coordinates `(cx, cy)` per image |
| `images&DOT&dot/`             | Image folder for test set                                 |
| `best_model_7.pth`            | Trained Faster R-CNN model checkpoint                     |
| `compute_confusion_matrix.py` | External helper for generating confusion matrix           |

---

## 📃 Output Files

| File/Folder                     | Description                               |
| ------------------------------- | ----------------------------------------- |
| `predicted_boxes/*.csv`         | Raw predictions from model                |
| `filtered_predicted_dots/*.csv` | Filtered predictions per dot position     |
| `merged_predictions.csv`        | Consolidated prediction results           |
| `classification_metrics.csv`    | Precision, Recall, F1-score               |
| `confusion_matrix_raw.csv`      | Raw values of confusion matrix            |
| `confusion_matrix.png`          | Heatmap visualization of confusion matrix |

---

## ⚖️ Dependencies

* Python >= 3.7
* Libraries:

  * `torch`, `torchvision`
  * `pandas`, `numpy`
  * `matplotlib`, `seaborn`
  * `scikit-learn`

---

## ✅ Summary

The `00_COMPLETE_TEST.py` script is a complete and automated evaluation framework for object detection models in forestry datasets. It integrates dot-based filtering, class-level metrics, and confusion matrix visualization into a single workflow.

Ideal for:

* Benchmarking trained models
* Post-training validation
* Paper-ready visualizations and metrics
