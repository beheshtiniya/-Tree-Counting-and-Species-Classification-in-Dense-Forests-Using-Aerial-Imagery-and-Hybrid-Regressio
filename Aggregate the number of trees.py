import os
import pandas as pd

# مسیرهای مورد نظر
base_dir = r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset\checkpoints_cross-without dotmap"
result_dirs = [f"result{i}\\COMBINE_RESULT" for i in range(5)]

# لیست برای ذخیره نتایج
tree_counts = []

# پردازش هر مسیر
for result_dir in result_dirs:
    full_path = os.path.join(base_dir, result_dir, "merged_predictions.csv")
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        count = len(df)
    else:
        count = 0  # اگر فایل وجود نداشت
    tree_counts.append({"result": result_dir.split("\\")[0], "tree_count": count})

# ذخیره گزارش نهایی
report_df = pd.DataFrame(tree_counts)
report_path = os.path.join(base_dir, "tree_count_report.csv")
report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

print("✅ گزارش ذخیره شد در:", report_path)
