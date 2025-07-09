import os
import csv
import math
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from confusion_matrix import compute_confusion_matrix
# تنظیمات CUDA
CUDA_LAUNCH_BLOCKING = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Device:", device)

# مسیر داده‌ها
root_dir = r"E:\FASTRCNN\FASTRCNN\dataset\00_source_code_data_ADDING BEST 0.95 TO TRAIN AND VAL\step_0_masked_with redpoint"
combine_result_dir = os.path.join(root_dir, "checkpoints_batch16", "COMBINE_RESULT")

# ایجاد پوشه‌های خروجی
os.makedirs(combine_result_dir, exist_ok=True)

# ---------------------------- 1️⃣ خواندن داده‌های تست ----------------------------
class trDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase
        self.targets = pd.read_csv(os.path.join(root, f'{phase}_labels.csv'))
        self.imgs = self.targets['filename'].astype(str)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, r'\FASTRCNN\FASTRCNN\dataset\00_source_code_data_ADDING BEST 0.95 TO TRAIN AND VAL', 'images&DOT&dot', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)

        filename = self.imgs[idx]
        return img, filename

    def __len__(self):
        return len(self.imgs)

# بارگذاری مجموعه داده
test_dataset = trDataset(root_dir, 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------------------- 2️⃣ اجرای مدل و ذخیره نتایج ----------------------------
output_folder = os.path.join(combine_result_dir, "predicted_boxes")
os.makedirs(output_folder, exist_ok=True)
print(f"📂 مسیر ذخیره خروجی مدل: {output_folder}")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5)
model.load_state_dict(torch.load(os.path.join(root_dir, r"E:\FASTRCNN\FASTRCNN\dataset\00_source_code_data_ADDING BEST 0.95 TO TRAIN AND VAL\step_0_masked_with redpoint\checkpoints_batch16", "best_model_7.pth")))
model.to(device)

model.eval()
print("🔍 پردازش تصاویر با مدل...")


with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="📸 پردازش تصاویر"):
        images = [image.to(device) for image in images]
        out = model(images)

        for i, (filename, prediction) in enumerate(zip(filenames, out)):


            # 2️⃣ دریافت نام فایل به‌درستی
            if isinstance(filename, list) or isinstance(filename, tuple):
                filename_str = filename[0]  # اگر لیست یا تاپل است، مقدار اول را بگیر
            else:
                filename_str = str(filename)  # در غیر این صورت، آن را رشته کن

            # 3️⃣ حذف مسیر و پسوند `.jpg` یا `.png` بدون تغییر مقدار
            filename_str = os.path.basename(str(filename_str))  # فقط نام فایل را بگیر
            filename_str = os.path.splitext(filename_str)[0]  # پسوند را حذف کن



            # 5️⃣ مسیر ذخیره فایل CSV
            output_csv_file = os.path.join(output_folder, f"{filename_str}.csv")

            # 6️⃣ پردازش خروجی مدل
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # 7️⃣ ساختن لیست داده‌ها قبل از تبدیل به DataFrame
            data = [
                [filename_str, label, int(xmin), int(ymin), int(xmax), int(ymax), float(score)]
                for box, label, score in zip(boxes, labels, scores)
                for xmin, ymin, xmax, ymax in [box]
            ]

            # 8️⃣ تبدیل به DataFrame
            df = pd.DataFrame(data, columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])

            # 9️⃣ اطمینان از اینکه `filename` به عنوان `str` ذخیره شود
            df['filename'] = df['filename'].astype(str)

            # 🔟 ذخیره CSV بدون تغییر نام فایل
            df.to_csv(output_csv_file, index=False)

print("✅ پردازش تصاویر به پایان رسید.")

# ---------------------------- 3️⃣ فیلتر کردن باکس‌ها بر اساس امتیاز ----------------------------
filtered_dots_folder = os.path.join(combine_result_dir, "filtered_predicted_dots")
os.makedirs(filtered_dots_folder, exist_ok=True)

points_folder = os.path.join(root_dir, r"E:\FASTRCNN\FASTRCNN\dataset\00_source_code_data_ADDING BEST 0.95 TO TRAIN AND VAL", "dots_csv")
print("points_folder",points_folder)
def calculate_distance(point, box_center):
    return math.sqrt((point[0] - box_center[0])**2 + (point[1] - box_center[1])**2)

def process_files(boxes_file, points_file, output_file):
    boxes_df = pd.read_csv(boxes_file)
    points_df = pd.read_csv(points_file)

    output_rows = []
    for _, point_row in points_df.iterrows():
        cx, cy = point_row['cx'], point_row['cy']

        best_box = None
        best_score = -1
        min_distance = float('inf')

        for _, box_row in boxes_df.iterrows():
            xmin, ymin, xmax, ymax = box_row['xmin'], box_row['ymin'], box_row['xmax'], box_row['ymax']
            score = box_row['score']

            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                box_center = ((xmax + xmin) / 2, (ymax + ymin) / 2)
                distance = calculate_distance((cx, cy), box_center)

                if score > best_score or (score == best_score and distance < min_distance):
                    best_box = box_row
                    best_score = score
                    min_distance = distance

        if best_box is not None:
            output_rows.append(best_box.to_dict())

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_file, index=False)

for points_filename in tqdm(os.listdir(points_folder), desc="📂 پردازش نقاط"):
    if points_filename.endswith(".csv"):
        boxes_file = os.path.join(output_folder, points_filename)
        points_file = os.path.join(points_folder, points_filename)
        output_file = os.path.join(filtered_dots_folder, points_filename)

        if os.path.exists(boxes_file) and os.path.exists(points_file):
            process_files(boxes_file, points_file, output_file)

print("✅ تمام فایل‌های نقاط پردازش و ذخیره شدند.")

# ---------------------------- 4️⃣ تجمیع فایل‌های CSV ----------------------------
# ---------------------------- 4️⃣ تجمیع فایل‌های CSV ----------------------------
merged_file_path = os.path.join(combine_result_dir, "merged_predictions.csv")

merged_data = []
for csv_file in tqdm(os.listdir(filtered_dots_folder), desc="🔄 تجمیع فایل‌ها"):
    if csv_file.endswith(".csv"):
        file_path = os.path.join(filtered_dots_folder, csv_file)
        df = pd.read_csv(file_path)

        # اصلاح نام فایل‌ها: حذف ".0" و اطمینان از پسوند ".tif"
        df['filename'] = df['filename'].astype(str).str.replace('.0', '', regex=False) + ".tif"

        # حذف ستون "score" اگر وجود داشته باشد
        if 'score' in df.columns:
            df = df.drop(columns=['score'])

        merged_data.append(df)

final_df = pd.concat(merged_data, ignore_index=True)
final_df.to_csv(merged_file_path, index=False)

print(f"✅ فایل تجمیع شده ذخیره شد: {merged_file_path}")




# تنظیمات CUDA
CUDA_LAUNCH_BLOCKING = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Device:", device)

# مسیر داده‌ها
root_dir = r"E:\FASTRCNN\FASTRCNN\dataset\00_source_code_data_ADDING BEST 0.95 TO TRAIN AND VAL\step_0_masked_with redpoint"
combine_result_dir = os.path.join(root_dir, "checkpoints_batch16", "COMBINE_RESULT")

# ایجاد پوشه‌های خروجی
os.makedirs(combine_result_dir, exist_ok=True)

# مسیر فایل‌های مورد نیاز
merged_file_path = os.path.join(combine_result_dir, "merged_predictions.csv")
test_labels_path = os.path.join(root_dir, "test_labels.csv")

# فراخوانی تابع محاسبه کانفیوشن ماتریس
conf_matrix, true_labels, pred_labels = compute_confusion_matrix(test_labels_path, merged_file_path)

# محاسبه متریک‌ها
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

# نمایش ماتریس کانفیوشن
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (5x5)")
plt.show()

# نمایش متریک‌ها
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ذخیره متریک‌ها در فایل CSV
metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score"],
    "Value": [precision, recall, f1]
})
metrics_df.to_csv(os.path.join(combine_result_dir, "classification_metrics.csv"), index=False)
print("✅ پردازش کامل شد.")
