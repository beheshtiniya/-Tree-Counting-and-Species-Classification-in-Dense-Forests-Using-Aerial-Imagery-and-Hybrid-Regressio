---

## 📦 02_test_extracted_best_save_csvs  

### Detect, Extract & Evaluate Bounding Boxes for Tree Species

---

### 🚀 Overview

This project applies a fine-tuned **Faster R-CNN (ResNet50-FPN)** model to detect, extract, and classify trees in aerial or satellite images. It integrates both **dot-based filtering** (via ground truth cx, cy points) and **IoU-based evaluation**, generating:

* Per-image visualizations
* Cleaned predicted bounding box CSVs
* A unified prediction file
* Confusion Matrix and statistics

---

### 🧠 Key Features

* 🔍 **Dot-based matching**: For each labeled tree point `(cx, cy)`, the model picks the highest-scoring predicted box that contains it.
* 🧽 **Duplicate removal**: Boxes are deduplicated using coordinate hashing.
* 📊 **IoU evaluation**: Predicted boxes are validated against GT using a threshold of `IoU > 0.1`.
* 📁 **CSV Export**: Each image's predictions are stored individually, plus a combined master CSV.
* 📷 **Visual Output**: Saved images include color-coded predicted boxes.
* 📉 **Confusion Matrix**: Final metrics are plotted for precision, recall, and class distribution.


### 🗂 Folder Structure

```
.
├── dataset/
│   ├── origin_data/
│   │   ├── images/                   # Raw images
│   │   ├── dots_csv/                # GT dot annotations (cx, cy)
│   │   ├── test_labels.csv          # GT boxes & labels for test
│   │   ├── new syntetic dataset/
│   │   │   ├── predicted_boxes/     # Predicted box CSVs
│   │   │   ├── output_images/       # Visualized predictions
│   └── images_masked/
│       └── checkpoints/
│           └── best_model.pth       # Trained Faster R-CNN weights
├── 02_test_extracted_best_save_csvs for predicted.py
```

---

### 🛠 Dependencies

```bash
torch
torchvision
pandas
matplotlib
scikit-learn
Pillow
```

---

### 🧪 How to Run

1. **Prepare the dataset**

   * Place raw `.jpg` images inside: `dataset//images`
   * Place dot annotations as `.csv`: `dataset//dots_csv`
   * Place GT box labels: `test_labels.csv`

2. **Run the evaluation**

```bash
python "02_test_extracted_best_save_csvs for predicted.py"
```

3. **Outputs generated:**

   * `predicted_boxes/*.csv` per image
   * `00_combined_predictions.csv` for all images
   * Visualizations in `output_images/*.png`
   * Confusion Matrix shown on screen

---

### 📊 Example Output

* **Prediction CSV Row Format:**

```csv
filename,class,xmin,ymin,xmax,ymax
img_001.jpg,2,103,140,215,255
```

* **Confusion Matrix:**

The following stats are printed and visualized:

```
Number of trees in GT not detected: 48
Number of predicted trees: 1523
Number of 1 in GT: 533
Number of 2 in GT: 467
...
```

---

### 🎯 Evaluation Strategy

#### 1. Dot-Based Filtering

* For each `(cx, cy)` point:

  * Find the predicted box containing it
  * Select the one with the highest score

#### 2. Duplicate Box Removal

* Convert boxes to tuple hash
* Use dictionary to keep unique ones

#### 3. IoU-Based Label Correction

* Match predicted boxes to GT using IoU > 0.1
* Use GT label if matched; otherwise default to `0`

---

### 🎨 Visualization Color Map

| Class | Color     |
| ----- | --------- |
| 1     | 🔴 Red    |
| 2     | 🟢 Green  |
| 3     | 🔵 Blue   |
| 4     | 🟡 Yellow |

---

### 📌 To-Do / Future Work

* [ ] Add mAP & per-class metrics
* [ ] Integrate validation dataset
* [ ] Optimize thresholds dynamically
* [ ] Add COCO-format export option

---

### 👨‍💻 Author

Developed by **\[Your Name]** as part of tree-counting and classification research project.
**Model:** `fasterrcnn_resnet50_fpn`
**Framework:** PyTorch + torchvision

---

### 📜 License

This project is under MIT License.

---

> For any inquiries or model retraining requests, feel free to open an issue or contact me directly.

---

