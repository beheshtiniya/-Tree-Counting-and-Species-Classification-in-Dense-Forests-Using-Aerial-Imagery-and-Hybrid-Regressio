## 📊 Project Report: Tree Species Classification and Localization using Faster R-CNN with Dual Ground Truth

**📁 Repository:** `02_test_extracted_best_save_csvs for predicted.py`
**🗓️ Last Updated:** 2025-07-15

---

### 🔍 Overview

This project implements and evaluates a **fine-tuned Faster R-CNN** model for tree detection and species classification in **dense aerial forest imagery**. It uniquely utilizes both:

* **GT Dots (cx, cy):** Annotated center points of trees for localized matching.
* **GT Boxes:** Ground truth bounding boxes for final metric evaluations.

---

### 🎯 Key Capabilities

| Feature                              | Description                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------- |
| ✅ Model Inference                    | Uses pretrained `Faster R-CNN ResNet-50 FPN`, fine-tuned for 5 tree classes |
| ✅ Dot-Based Filtering                | Matches predicted boxes with GT dots using spatial enclosure                |
| ✅ GT Box Evaluation                  | Computes metrics using IoU against ground truth bounding boxes              |
| ✅ IOU Matching (0.1 threshold)       | Extracts overlapping boxes and labels based on intersection ratio           |
| ✅ Confusion Matrix                   | Plots full classification performance matrix                                |
| ✅ Visualization                      | Saves color-coded bounding boxes for qualitative inspection                 |
| ✅ Per-Image CSV & Merged Predictions | Saves both individual and unified prediction CSVs                           |

---

### 📁 Directory Structure

```
root_dir/
├── images/                          # RGB input images
├── train_labels.csv                # GT boxes and labels for training
├── test_labels.csv                 # GT boxes and labels for testing
├── dots_csv/                       # GT dot annotations for test images
├── predicted_boxes/               # Extracted CSVs per image
├── output_images/                 # PNGs of visualized predictions
└── 00_combined_predictions.csv    # Merged prediction results
```

---

### 🛠️ How It Works

1. **Dataset Loading**

   * Loads both GT bounding boxes (`*_labels.csv`) and center dots (`dots_csv/`).
   * Converts per-image labels into tensors for evaluation.

2. **Model Inference**

   * Loads `.pth` model with custom predictor head for 5 classes.
   * Applies prediction on test set with adjustable score threshold (`0.2`).

3. **Dot-Based Filtering**

   * For each GT point `(cx, cy)`, searches for highest-scoring predicted box that contains it.
   * Removes duplicate boxes and consolidates labels.

4. **IoU-Based Validation**

   * Calculates IoU between predicted and GT boxes (`IoU > 0.1`).
   * Retains overlapping predictions and updates class labels accordingly.

5. **Evaluation Metrics**

   * Constructs `confusion_matrix()` using final predicted and GT labels.
   * Computes per-class overlap statistics.
   * Visualizes results using `ConfusionMatrixDisplay()`.

6. **Output Files**

   * Saves `.csv` files of predictions for each image.
   * Merges results into `00_combined_predictions.csv`.
   * Outputs `.png` image with bounding boxes color-coded by class:

     * Red = Class 1, Green = Class 2, Blue = Class 3, Yellow = Class 4

---

### 🧪 Example Output Metrics

```
Number of undetected trees (based on GT): 83
Predicted instances: 920
GT class distribution:
  - Class 1: 533
  - Class 2: 467
  - Class 3: 134
  - Class 4: 769
IoU threshold: 0.1 | Score threshold: 0.2
```

📊 *Confusion matrix is plotted using `matplotlib` and includes all detected and missed classes.*

---

### 🔍 Highlights

* Robust hybrid filtering (dot-level + box-level)
* High transparency in false negatives via GT comparison
* Dataset-agnostic implementation with minor path changes

---

### ⚠️ Considerations

* Assumes all images have `.jpg` extension.
* Predictions are filtered strictly by spatial containment first, then IoU.
* Performance sensitive to score/IoU thresholds.

---

### 🚀 Future Enhancements

* Add per-class precision/recall reporting.
* Dynamic threshold tuning for IoU and confidence.
* Export confusion matrix to CSV/JSON.
* Replace matplotlib with interactive `Plotly` version for dashboards.

---

### ✅ Dependencies

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn
```

---

### 👩‍💻 Maintainer

Prepared by \[Your Name], July 2025
Contact: [your.email@example.com](mailto:your.email@example.com)
