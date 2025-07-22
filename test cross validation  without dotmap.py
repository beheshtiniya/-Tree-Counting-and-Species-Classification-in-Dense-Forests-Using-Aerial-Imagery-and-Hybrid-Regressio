# -*- coding: utf-8 -*-
"""
Code 1 — Multi‑fold edition (rev‑4) ✅
• مرحلهٔ 1: منطق اصلی Code 1 (تست روی تصاویر بدون نقطه، فیلتر IoU ≥ 0.5).
• مرحلهٔ 2: از کانفیوشن ماتریس به بعد کاملاً مطابق Code 2.
• حلقهٔ for fold in range(NUM_FOLDS) اضافه شد تا همهٔ فولدها را پشت‌سر‌هم پردازش کند و در پایان یک فایل «final_report.xlsx» بسازد.
"""

import os, math, csv, numpy as np, pandas as pd, torch, torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from confusion_matrix import compute_confusion_matrix

# ---------------------------- تنظیمات عمومی ----------------------------
CUDA_LAUNCH_BLOCKING = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device:", DEVICE)

NUM_FOLDS = 5                       # ← اگر فولد دیگری داری مقدار را عوض کن
FOLD_DIR  = "checkpoints_cross-without dotmap"  # پوشهٔ چک‌پوینت‌ها
ROOT_DIR  = r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset"

# ---------------------------- DataSet تعریف ----------------------------
class TRDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase
        meta = pd.read_csv(os.path.join(root, f"{phase}_labels.csv"))
        self.img_files = meta["filename"].astype(str)

    def __getitem__(self, idx):
        img_name = self.img_files.iloc[idx]
        img_path = os.path.join(self.root, "images_without red point", img_name)
        img = Image.open(img_path).convert("RGB")
        return F.to_tensor(img), img_name

    def __len__(self):
        return len(self.img_files)

test_ds  = TRDataset(ROOT_DIR, "test")
TEST_LOADER = DataLoader(test_ds, batch_size=1, shuffle=False)

# ---------------------------- تابع IoU و فیلتر ----------------------------

def iou(box1, box2):
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter   = max(xi2-xi1, 0) * max(yi2-yi1, 0)
    area1   = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2   = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def filter_iou(pred_csv, gt_df, out_csv, thr=0.5):
    pred_df, out_rows = pd.read_csv(pred_csv), []
    tif_name = os.path.splitext(os.path.basename(pred_csv))[0] + ".tif"
    gts = gt_df[gt_df["filename"] == tif_name]
    for _, g in gts.iterrows():
        gt_box = [g["xmin"], g["ymin"], g["xmax"], g["ymax"]]
        best, best_iou, best_score = None, -1, -1
        for _, r in pred_df.iterrows():
            i = iou(gt_box, [r["xmin"], r["ymin"], r["xmax"], r["ymax"]])
            if i >= thr and (i>best_iou or (i==best_iou and r["score"]>best_score)):
                best, best_iou, best_score = r, i, r["score"]
        if best is not None:
            out_rows.append(best.to_dict())
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)

# ---------------------------- حلقهٔ Cross‑Validation ----------------------------
for fold in range(NUM_FOLDS):
    print("\n"+"="*60)
    print(f"🔥 شروع پردازش fold {fold}")
    print("="*60)

    # مسیرهای خروجی
    fold_out_dir  = os.path.join(ROOT_DIR, FOLD_DIR, f"result{fold}", "COMBINE_RESULT")
    os.makedirs(fold_out_dir, exist_ok=True)
    raw_box_dir   = os.path.join(fold_out_dir, "predicted_boxes")
    filt_box_dir  = os.path.join(fold_out_dir, "filtered_predicted_dots")
    for d in (raw_box_dir, filt_box_dir):
        os.makedirs(d, exist_ok=True)

    # -------------------- بارگذاری مدل --------------------
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feat, 5)

    ckpt = torch.load(os.path.join(ROOT_DIR, FOLD_DIR, f"fold{fold}_best.pth"), map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(DEVICE).eval()

    # -------------------- پیش‌بینی و ذخیره --------------------
    with torch.no_grad():
        for imgs, names in tqdm(TEST_LOADER, desc="📸 inference"):
            preds = model([img.to(DEVICE) for img in imgs])
            for pred, name in zip(preds, names):
                stem = os.path.splitext(os.path.basename(name))[0]
                out_csv = os.path.join(raw_box_dir, f"{stem}.csv")
                rows = [
                    [stem, int(l), int(x1), int(y1), int(x2), int(y2), float(s)]
                    for (x1,y1,x2,y2), l, s in zip(pred["boxes"].cpu().numpy(),
                                                   pred["labels"].cpu().numpy(),
                                                   pred["scores"].cpu().numpy())
                ]
                pd.DataFrame(rows, columns=["filename","class","xmin","ymin","xmax","ymax","score"]).to_csv(out_csv, index=False)

    print("✅ پیش‌بینی تمام شد.")

    # -------------------- فیلتر IoU --------------------
    gt_csv = os.path.join(ROOT_DIR, "test_labels.csv")
    gt_df  = pd.read_csv(gt_csv)
    for fn in tqdm(os.listdir(raw_box_dir), desc="🔍 IoU filter"):
        if fn.endswith(".csv"):
            filter_iou(os.path.join(raw_box_dir, fn), gt_df, os.path.join(filt_box_dir, fn))

    # -------------------- Merge --------------------
    merged_csv = os.path.join(fold_out_dir, "merged_predictions.csv")
    dfs = []
    for csvf in tqdm(os.listdir(filt_box_dir), desc="🔄 merge"):
        if not csvf.endswith(".csv"): continue
        path = os.path.join(filt_box_dir, csvf)
        if os.stat(path).st_size==0: continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        df["filename"] = df["filename"].astype(str).str.replace(".0", "", regex=False)+".tif"
        if "score" not in df.columns:
            df["score"] = 0.0
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"هیچ CSV معتبری برای fold {fold} پیدا نشد!")
    pd.concat(dfs, ignore_index=True).to_csv(merged_csv, index=False)

    # -------------------- Confusion & Metrics (منطق Code 2) --------------------
    conf_mx, y_true, y_pred = compute_confusion_matrix(gt_csv, merged_csv)
    conf_df = pd.DataFrame(conf_mx, columns=range(conf_mx.shape[1]), index=range(conf_mx.shape[0]))
    conf_df.to_csv(os.path.join(fold_out_dir, "confusion_matrix.csv"))

    # total samples
    total = conf_mx.sum()
    pd.DataFrame({"Total Samples":[total]}).to_csv(os.path.join(fold_out_dir, "total_samples.csv"), index=False)

    # zero row class0 & result matrix
    conf_mx[0,:] = 0
    result_df = pd.DataFrame(conf_mx,
        columns=[f"pred_class{i}" for i in range(conf_mx.shape[1])],
        index=[f"true_class{i}" for i in range(conf_mx.shape[0])])
    result_df.to_csv(os.path.join(fold_out_dir, "result_matrix.csv"))

    # metrics
    def calc_metrics(cm):
        cm = np.array(cm); n = cm.shape[0]
        rows = []
        for i in range(n):
            TP = cm[i,i]
            FP = cm[:,i].sum()-TP
            FN = cm[i,:].sum()-TP
            prec = TP/(TP+FP) if TP+FP>0 else 0
            rec  = TP/(TP+FN) if TP+FN>0 else 0
            f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
            rows.append({"Class":f"class{i}","Precision":round(prec,4),"Recall":round(rec,4),"F1-Score":round(f1,4),"Support":int(cm[i,:].sum())})
        accuracy = np.trace(cm)/cm.sum()
        macro_prec = np.mean([r["Precision"] for r in rows[1:]])
        macro_rec  = np.mean([r["Recall"]    for r in rows[1:]])
        macro_f1   = np.mean([r["F1-Score"]  for r in rows[1:]])
        rows.append({"Class":"overall (macro)","Precision":round(macro_prec,4),"Recall":round(macro_rec,4),"F1-Score":round(macro_f1,4),"Support":int(cm.sum())})
        rows.append({"Class":"accuracy","Precision":"","Recall":"","F1-Score":round(accuracy,4),"Support":""})
        return pd.DataFrame(rows)

    metrics_df = calc_metrics(conf_mx)
    metrics_df.to_excel(os.path.join(fold_out_dir, "metrics.xlsx"), index=False)

    print(f"✅ fold{fold} done | Precision = {metrics_df.loc[metrics_df['Class']=='overall (macro)','Precision'].values[0]:.4f}")

# ---------------------------- گزارش نهایی ----------------------------
print("\n📊 ساخت گزارش نهایی across folds…")
BASE_DIR = os.path.join(ROOT_DIR, FOLD_DIR)
res_folders = [f"result{i}" for i in range(NUM_FOLDS)]

all_mets, all_tot, all_conf = [], [], []
for i, fdir in enumerate(res_folders):
    d = os.path.join(BASE_DIR, fdir, "COMBINE_RESULT")
    if not os.path.isdir(d): continue
    mp = os.path.join(d, "metrics.xlsx");
    if os.path.exists(mp):
        df = pd.read_excel(mp); df["Fold"] = f"fold{i}"; all_mets.append(df)
    tp = os.path.join(d, "total_samples.csv");
    if os.path.exists(tp):
        df = pd.read_csv(tp); df["Fold"] = f"fold{i}"; all_tot.append(df)
    cp = os.path.join(d, "confusion_matrix.csv");
    if os.path.exists(cp):
        df = pd.read_csv(cp); df["Fold"] = f"fold{i}"; all_conf.append(df)

report_path = os.path.join(BASE_DIR, "final_report.xlsx")
writer = pd.ExcelWriter(report_path, engine="xlsxwriter")

if all_mets:
    met = pd.concat(all_mets, ignore_index=True)
    met = met[["Fold","Class","Precision","Recall","F1-Score","Support"]]
    avg = met.groupby("Class").mean(numeric_only=True).reset_index()
    avg.insert(0,"Fold","Average");
    final_met = pd.concat([met,avg], ignore_index=True)
    final_met.to_excel(writer, sheet_name="Metrics", index=False)

    ws = writer.sheets["Metrics"]
    ws.set_column("A:A",12); ws.set_column("B:B",15); ws.set_column("C:E",10); ws.set_column("F:F",10)
    fmt = writer.book.add_format({"bg_color":"#FFF2CC","bold":True})
    for idx,row in final_met.iterrows():
        if row["Fold"]=="Average": ws.set_row(idx+1,None,fmt)

if all_tot:
    tot = pd.concat(all_tot, ignore_index=True).rename(columns={"Total Samples":"Total Trees"})
    tot = tot[["Fold","Total Trees"]]
    tot_avg = pd.DataFrame({"Fold":["Average"],"Total Trees":[tot["Total Trees"].mean()]})
    final_tot = pd.concat([tot, tot_avg], ignore_index=True)
    final_tot.to_excel(writer, sheet_name="Total Trees", index=False)
    writer.sheets["Total Trees"].set_column("A:B",15)
    writer.sheets["Total Trees"].set_row(len(final_tot), None, fmt)

if all_conf:
    pd.concat(all_conf, ignore_index=True).to_excel(writer, sheet_name="Confusion Matrix", index=False)

writer.close()
print(f"✅ گزارش نهایی ذخیره شد: {report_path}")
# ---------------------------- 𝟞️⃣ افزودن Total_Samples خلاصهٔ پنج فولد ----------------------------
import pandas as pd
from pathlib import Path

# مسیر فولدرهای نتیجه (هر فولد داخل COMBINE_RESULT)
base_results_dir = Path(ROOT_DIR) / "checkpoints_cross-without dotmap"
result_folders   = [f"result{f}" for f in range(5)]

total_frames = []
for f_idx, fold_name in enumerate(result_folders):
    csv_path = base_results_dir / fold_name / "COMBINE_RESULT" / "total_samples.csv"
    if csv_path.exists():
        df_tmp = pd.read_csv(csv_path)
        if "Total Samples" in df_tmp.columns:
            df_tmp = df_tmp.rename(columns={"Total Samples": "Total_Samples"})
        df_tmp["Fold"] = f"fold{f_idx}"
        total_frames.append(df_tmp[["Fold", "Total_Samples"]])

if total_frames:
    total_df = pd.concat(total_frames, ignore_index=True)
    mean_row = pd.DataFrame({"Fold": ["Average"], "Total_Samples": [total_df["Total_Samples"].mean()]})
    total_df = pd.concat([total_df, mean_row], ignore_index=True)

    # الحاق یا ایجاد برگه در گزارش نهایی
    final_report_path = base_results_dir / "final_report.xlsx"
    with pd.ExcelWriter(final_report_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        total_df.to_excel(writer, sheet_name="Total Samples", index=False)

    print(f"📊 برگهٔ ‘Total Samples’ به {final_report_path} اضافه شد.")
else:
    print("⚠️ هیچ total_samples.csv در مسیر فولدها پیدا نشد؛ برگه ساخته نشد.")



# ---------- تنظیم مسیرهای فولدرهای result ----------
BASE_DIR = r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset\checkpoints_cross-without dotmap"
RESULT_FOLDERS = [f"result{i}" for i in range(5)]             # result0 … result4
SUBDIR = "COMBINE_RESULT"                                     # زیرپوشهٔ حاوی CSVها

# ---------- جمع‌آوری total_samples.csv های هر فولدر ----------
records = []
for folder in RESULT_FOLDERS:
    csv_path = os.path.join(BASE_DIR, folder, SUBDIR, "total_samples.csv")
    if not os.path.isfile(csv_path):
        print(f"⚠️  فایل یافت نشد: {csv_path}")
        continue

    try:
        df = pd.read_csv(csv_path)            # انتظار ستون «Total Samples»
        total = df.iloc[0, 0]                 # اولین مقدار عددی جدول
        records.append({"Fold": folder, "Total_Samples": int(total)})
    except Exception as e:
        print(f"⚠️  خطا در خواندن {csv_path}: {e}")

# ---------- ذخیرهٔ خروجی ترکیبی ----------
if records:
    out_df = pd.DataFrame(records)
    out_path = os.path.join(BASE_DIR, "all_total_samples_without_dotmap.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ فایل تجمیعی ذخیره شد: {out_path}")
else:
    print("❌ هیچ total_samples.csv معتبری پیدا نشد.")
# ---------------------------- 𝟟️⃣ شمارش تعداد ردیف‌های merged_predictions برای هر فولد ----------------------------
print("\n📦 شمارش تعداد درختان از فایل merged_predictions.csv برای هر result")

tree_counts = []
for fold in range(NUM_FOLDS):
    merged_csv_path = os.path.join(ROOT_DIR, FOLD_DIR, f"result{fold}", "COMBINE_RESULT", "merged_predictions.csv")
    if os.path.exists(merged_csv_path):
        try:
            df = pd.read_csv(merged_csv_path)
            count = len(df)
        except Exception as e:
            print(f"⚠️ خطا در خواندن فایل fold{fold}: {e}")
            count = 0
    else:
        print(f"⚠️ فایل یافت نشد: {merged_csv_path}")
        count = 0
    tree_counts.append({"Fold": f"fold{fold}", "Tree_Count": count})

# ذخیره به عنوان فایل CSV
tree_report_path = os.path.join(ROOT_DIR, FOLD_DIR, "tree_count_report.csv")
pd.DataFrame(tree_counts).to_csv(tree_report_path, index=False, encoding="utf-8-sig")
print(f"✅ فایل شمارش درختان ذخیره شد: {tree_report_path}")
