import os
import numpy as np
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
from torchvision.ops import box_iou
from collections import defaultdict

# تنظیمات CUDA
CUDA_LAUNCH_BLOCKING = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Device:", device)

# مسیر داده‌ها
root_dir = r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset"

# حلقه برای اجرای هر fold
for fold in range(5):
    print(f"\n{'=' * 50}")
    print(f"🔥 شروع پردازش برای fold {fold}")
    print(f"{'=' * 50}")

    # مسیر خروجی برای این fold
    combine_result_dir = os.path.join(root_dir, "checkpoints_cross", f"result{fold}")
    os.makedirs(combine_result_dir, exist_ok=True)

    # ---------------------------- 1️⃣ خواندن داده‌های تست ----------------------------
    class trDataset(torch.utils.data.Dataset):
        def __init__(self, root, phase):
            self.root = root
            self.phase = phase
            self.targets = pd.read_csv(os.path.join(root, f'{phase}_labels.csv'))
            self.imgs = self.targets['filename'].astype(str)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root, r'E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset', 'images',
                                    self.imgs[idx])
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
    checkpoint = torch.load(os.path.join(root_dir, r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset\checkpoints_cross",
                                         f"fold{fold}_best.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()
    print("🔍 پردازش تصاویر با مدل...")

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="📸 پردازش تصاویر"):
            images = [image.to(device) for image in images]
            out = model(images)

            for i, (filename, prediction) in enumerate(zip(filenames, out)):
                if isinstance(filename, list) or isinstance(filename, tuple):
                    filename_str = filename[0]
                else:
                    filename_str = str(filename)

                filename_str = os.path.basename(str(filename_str))
                filename_str = os.path.splitext(filename_str)[0]

                output_csv_file = os.path.join(output_folder, f"{filename_str}.csv")

                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()

                data = [
                    [filename_str, label, int(xmin), int(ymin), int(xmax), int(ymax), float(score)]
                    for box, label, score in zip(boxes, labels, scores)
                    for xmin, ymin, xmax, ymax in [box]
                ]

                df = pd.DataFrame(data, columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
                df['filename'] = df['filename'].astype(str)
                df.to_csv(output_csv_file, index=False)

    print("✅ پردازش تصاویر به پایان رسید.")

    # ---------------------------- 3️⃣ فیلتر کردن باکس‌ها بر اساس امتیاز ----------------------------
    filtered_dots_folder = os.path.join(combine_result_dir, "filtered_predicted_dots")
    os.makedirs(filtered_dots_folder, exist_ok=True)

    points_folder = os.path.join(root_dir, r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset\dots_csv", "csv")
    print("points_folder", points_folder)

    def calculate_distance(point, box_center):
        return math.sqrt((point[0] - box_center[0]) ** 2 + (point[1] - box_center[1]) ** 2)

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
    merged_file_path = os.path.join(combine_result_dir, "merged_predictions.csv")

    merged_data = []
    for csv_file in tqdm(os.listdir(filtered_dots_folder), desc="🔄 تجمیع فایل‌ها"):
        if csv_file.endswith(".csv"):
            file_path = os.path.join(filtered_dots_folder, csv_file)

            if os.stat(file_path).st_size == 0:
                print(f"⚠️ فایل {csv_file} خالی است. از تجمیع آن صرف نظر می‌شود.")
                continue

            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"⚠️ فایل {csv_file} خالی است یا داده‌ای در آن وجود ندارد. از آن عبور می‌کنیم.")
                continue

            df['filename'] = df['filename'].astype(str).str.replace('.0', '', regex=False) + ".tif"
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)
            merged_data.append(df)

    if merged_data:
        final_df = pd.concat(merged_data, ignore_index=True)
        final_df.to_csv(merged_file_path, index=False)
        print(f"✅ فایل تجمیع شده ذخیره شد: {merged_file_path}")
    else:
        print("❌ هیچ داده‌ای برای تجمیع یافت نشد.")

    # ---------------------------- محاسبه ماتریس کانفیوشن و متریک‌های طبقه‌بندی ----------------------------
    test_labels_path = os.path.join(root_dir, "test_labels.csv")

    # فراخوانی تابع محاسبه کانفیوشن ما
    conf_matrix, true_labels, pred_labels = compute_confusion_matrix(test_labels_path, merged_file_path)

    # ذخیره ماتریس کانفیوشن به‌صورت CSV
    conf_df = pd.DataFrame(conf_matrix, columns=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])
    conf_csv_path = os.path.join(combine_result_dir, "confusion_matrix.csv")
    conf_df.to_csv(conf_csv_path)
    print(f"📁 ماتریس کانفیوشن به‌صورت CSV ذخیره شد: {conf_csv_path}")

    # ---------------------------- پردازش ماتریس کانفیوژن ----------------------------
    # 1. خواندن ماتریس کانفیوشن از فایل CSV
    conf_matrix_df = pd.read_csv(conf_csv_path, index_col=0)
    conf_matrix = conf_matrix_df.values

    # 2. محاسبه مجموع تمام عناصر ماتریس
    total_samples = conf_matrix.sum()
    print(f"تعداد کل نمونه‌ها: {total_samples}")

    # 3. ذخیره مجموع در فایل اکسل
    total_samples_df = pd.DataFrame({'Total Samples': [total_samples]})
    total_samples_path = os.path.join(combine_result_dir, "total_samples.csv")
    total_samples_df.to_csv(total_samples_path, index=False)
    print(f"📊 تعداد کل نمونه‌ها ذخیره شد: {total_samples_path}")

    # 4. صفر کردن سطر مربوط به class0
    conf_matrix[0, :] = 0

    # 5. ایجاد ماتریس جدید با نام result
    result_matrix = conf_matrix.copy()

    # 6. ذخیره ماتریس result در فایل اکسل جدید
    result_df = pd.DataFrame(result_matrix,
                             columns=[f'pred_class{i}' for i in range(result_matrix.shape[1])],
                             index=[f'true_class{i}' for i in range(result_matrix.shape[0])])
    result_path = os.path.join(combine_result_dir, "result_matrix.csv")
    result_df.to_csv(result_path)
    print(f"📁 ماتریس result ذخیره شد: {result_path}")

    # 7. نمایش اطلاعات
    print("\nماتریس کانفیوژن اصلی:")
    print(conf_matrix_df)
    print(f"\nمجموع تمام عناصر: {total_samples}")
    print("\nماتریس result (پس از صفر کردن سطر class0):")
    print(result_df)

    # ---------------------------- محاسبه متریک‌ها از ماتریس result ----------------------------
    def calculate_metrics(confusion_matrix):
        cm = np.array(confusion_matrix)
        num_classes = cm.shape[0]

        # محاسبه TP, FP, FN برای هر کلاس
        class_metrics = []
        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics.append({
                'Class': f'class{i}',
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4),
                'Support': int(cm[i, :].sum())
            })

        # محاسبه accuracy کل
        accuracy = np.trace(cm) / cm.sum()

        # محاسبه میانگین‌ها
        macro_precision = np.mean([m['Precision'] for m in class_metrics[1:]])
        macro_recall = np.mean([m['Recall'] for m in class_metrics[1:]])
        macro_f1 = np.mean([m['F1-Score'] for m in class_metrics[1:]])

        # اضافه کردن متریک‌های کلی
        overall_metrics = {
            'Class': 'overall (macro)',
            'Precision': round(macro_precision, 4),
            'Recall': round(macro_recall, 4),
            'F1-Score': round(macro_f1, 4),
            'Support': int(cm.sum())
        }

        # اضافه کردن accuracy
        accuracy_metrics = {
            'Class': 'accuracy',
            'Precision': '',
            'Recall': '',
            'F1-Score': round(accuracy, 4),
            'Support': ''
        }

        return pd.DataFrame(class_metrics + [overall_metrics, accuracy_metrics])

    # محاسبه متریک‌ها
    metrics_df = calculate_metrics(result_matrix)

    # ذخیره در فایل اکسل
    metrics_path = os.path.join(combine_result_dir, "metrics.xlsx")
    metrics_df.to_excel(metrics_path, index=False)
    print(f"📊 متریک‌ها در فایل ذخیره شدند: {metrics_path}")

    # نمایش نتایج در کنسول
    print("\nنتایج متریک‌ها:")
    print(metrics_df)
