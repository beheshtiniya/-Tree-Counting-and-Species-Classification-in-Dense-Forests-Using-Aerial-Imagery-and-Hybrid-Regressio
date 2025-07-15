## 📊 Project Report: Evaluation of Tree Species Detection via Faster R-CNN

### 📁 Repository: -Tree-Counting-and-Species-Classification-in-Dense-Forests-Using-Aerial-Imagery-and-Hybrid-Regressio

**Last updated:** 2025-07-15

---

### 🔍 Overview

This project evaluates a fine-tuned [Faster R-CNN](https://arxiv.org/abs/1506.01497) object detection model applied to **aerial forest imagery** for the dual task of:

* **Tree counting** (instance detection)
* **Tree species classification** (multi-class prediction)

The core script (referred to as **Code 2**) performs a full pipeline from loading test images, running the trained model, filtering predictions, comparing with ground truth (GT), and generating evaluation metrics.

---

### 🧠 Key Features

| Module                | Description                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| 🔄 Model Inference    | Uses pretrained Faster R-CNN with custom head (5 species classes)                                   |
| 📍 GT Comparison      | Evaluates predictions by matching them with **annotated tree dots (cx, cy)** and **bounding boxes** |
| 🎯 Evaluation Metrics | Computes precision, recall, F1-score, and a detailed confusion matrix                               |
| 📁 Export             | Saves predicted bounding boxes per image and exports merged results to CSV                          |
| 🖼️ Visualization     | Saves predicted images with bounding boxes color-coded per class                                    |

---

### 📂 Directory Structure

```bash
dataset/
├── images/                    # RGB images (.jpg/.tif)
├── test_labels.csv           # GT bounding boxes + class
├── dots_csv/                 # GT dot annotations (cx, cy)
├── predicted_boxes/          # Model output boxes per image
└── output_images/            # Visualization of predictions
```

---

### ⚙️ How It Works

1. **Dataset Loading**

   * Loads test images and annotations (bounding boxes and labels).
   * For each image, GT dot file is read from `dots_csv`.

2. **Model Inference**

   * Loads the trained model from `.pth` file.
   * For each image:

     * Predicts boxes, scores, labels.
     * Filters out low-confidence boxes.

3. **Matching Predictions to GT**

   * For each GT dot `(cx, cy)`:

     * Finds the best predicted box that encloses the point.
     * Assigns that box's label to the point (if valid).
   * Filters and deduplicates predictions.

4. **IoU Matching and Label Correction**

   * Calculates IoU between predicted and GT boxes.
   * Matches class labels only when IoU exceeds a threshold (default `0.1`).
   * Counts undetected trees and incorrect predictions.

5. **Visualization**

   * Saves images with color-coded bounding boxes for each class:

     * Red: Class 1, Green: Class 2, Blue: Class 3, Yellow: Class 4.

6. **CSV Merging**

   * Combines per-image CSVs into a unified prediction file.

7. **Metrics and Evaluation**

   * Computes final confusion matrix and classification scores.
   * Prints and saves metrics.

---

### 📌 Evaluation Outputs

* `00_combined_predictions.csv`: Combined prediction results
* `classification_metrics.csv`: Final evaluation metrics
* `output_images/*.png`: Visual outputs for qualitative analysis
* `Confusion Matrix`: Visualized with matplotlib/seaborn

---

### 📊 Example Metrics Output

```text
Precision: 0.90
Recall:    0.88
F1 Score:  0.89
```

---

### 🔫 Dependencies

* `torch`, `torchvision`
* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`
* `tqdm`

---

### ⚠️ Notes & Considerations

* GT includes both **bounding boxes** and **tree center points (dots)**.
* Filtering is **dot-based**: predictions are accepted if they cover a GT dot.
* Misaligned or low-IoU predictions are **penalized** in metrics.
* Visualization helps with human validation of detections.

---

### 🔄 Future Improvements

* Add PR-curves per class.
* Integrate IoU threshold sweep to optimize performance.
* Apply post-processing to merge overlapping boxes (e.g. NMS tuning).
