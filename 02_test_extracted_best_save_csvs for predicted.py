import os
import pandas as pd
import torch
import torchvision
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms import functional as F
import numpy as np
import os
import csv


CUDA_LAUNCH_BLOCKING = 1.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("device:", device)
image_and_targets=[]
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.patches as patches

def read_points_from_excel(file_path):
    df = pd.read_excel(file_path)
    points = df[['cx', 'cy']].values.tolist()
    return points

# Function to read points from a CSV file
import pandas as pd

# Function to read points from a CSV file
def read_points_from_csv(file_path):
    df = pd.read_csv(file_path)
    points = df[['cx', 'cy']].values.tolist()
    return points

class trDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase

        # self.imgs = os.listdir(os.path.join(root, 'images'))
        self.targets = pd.read_csv(os.path.join(root, '{}_labels.csv'.format(phase)))
        self.imgs = self.targets['filename']
        self.name=self.targets['filename']


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\images', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)
        # print("idx:",idx)
        filename = self.imgs[idx]
        # print("Filename:", filename)
        box_list = self.targets[self.targets['filename'] == self.imgs[idx]]
        box_list = box_list[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.tensor(box_list, dtype=torch.float32)
        label_list = self.targets[self.targets['filename'] == self.imgs[idx]]
        label_list = label_list[['class']].values.squeeze(1)
        labels = torch.tensor(label_list, dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        return img, target, filename

    def __len__(self):
        return len(self.imgs)
train_dataset = trDataset(r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data', 'train')
test_dataset = trDataset(r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data', 'test')
# print("************************************")
# print(train_dataset.__getitem__(10))
# print("************************************")
def new_concat(batch):
  return tuple(zip(*batch))
train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=False,
                            collate_fn=new_concat)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=new_concat)
def evaluate(model, test_dataloader):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    final_all_true_labels =[]
    final_all_pred_labels =[]
    num_tree_un_detect=0
    counter=0
    last_filename = None
    with torch.no_grad():
        for images, targets, filename in test_dataloader:
            images = list(image.to(device) for image in images)
            if (last_filename == filename):
                continue

            last_filename = filename
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(images)
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print("counter:",counter)
            # counter=counter+1
            # print("file name:",filename)
            # print("out:", out)
            # print("out[0]:", out[0])
            scores = out[0]['scores'].cpu().numpy()
            # print('scores', scores)
            treshold_score=0.2
            inds = scores > treshold_score
            # print("inds",inds)
            bxs = out[0]['boxes'].cpu().numpy()
            bxs = bxs[inds]
            lbl = out[0]['labels'].cpu().numpy()
            lbl = lbl[inds]
            # print("lbl:",lbl)
            score=scores[inds]
            # print("score:",score)
            # print("bxs:",bxs)
            filename_str = filename[0] if isinstance(filename, tuple) else filename

            # Remove the '.jpg' extension and add '.csv'
            filename_without_extension = os.path.splitext(filename_str)[0]
            new_filename = filename_without_extension + '.csv'

            # Construct the file path
            file_path = os.path.join(
            r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\dots_csv',
                new_filename)
            points = read_points_from_csv(file_path)
            # print("points:",points)
            gt_labels = targets[0]['labels'].cpu().numpy()
            pred_labels = lbl
            print("*****************************************************************")
            print("file name:",filename)
            print('gt_labels:',gt_labels)
            # Visualization code (unchanged)
            gt = targets[0]['boxes'].cpu().numpy()
            name = targets[0]['labels'].cpu().numpy()
            img = images[0].permute(1, 2, 0).cpu().numpy()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            counter+=1

            # select best boxes
            extracted_boxes = []
            extracted_lbl = []
            updated_labels1 =[]
            # print("len points: ",len (points))
            for point in points:
                x, y = point
                # print("x:",x)
                # print("y:",y)
                # ax.plot(x, y, 'wo', markersize=10)  # 'bo' means blue color, circle marker
                closest_box = None
                min_distance = float('inf')
                highest_score = -float('inf')
                # print("highest_score:",highest_score)
                fallback_box = None
                for box, label, scr in zip(bxs, lbl, score):
                    xmin, ymin, xmax, ymax = box
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        if scr > highest_score:
                            highest_score = scr
                            closest_box = box
                            closest_box_lbl = label
                if closest_box is not None:
                    extracted_boxes.append(closest_box)
                    extracted_lbl.append(closest_box_lbl)
            #------------------------------------------------------
            # print ("extracted_boxes: ",extracted_boxes)
            # print("extracted_lbl:",extracted_lbl)

            # Convert to list of tuples for easy comparison
            boxes_tuples = [tuple(box) for box in extracted_boxes]

            # Use a dictionary to remove duplicates while preserving order
            unique_boxes = {}
            for box, lbl in zip(boxes_tuples, extracted_lbl):
                if box not in unique_boxes:
                    unique_boxes[box] = lbl

            # Convert back to list of arrays and labels
            unique_extracted_boxes = [np.array(box) for box in unique_boxes.keys()]
            unique_extracted_lbl = list(unique_boxes.values())
            extracted_boxes =unique_extracted_boxes
            # print("Unique extracted_boxes:", extracted_boxes)
            print("Unique extracted_lbl:", unique_extracted_lbl)
            print("all_true_labels:", gt_labels)


            # Define the folder path for saving predicted boxes
            output_folder = r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset\predicted_boxes'

            # Ensure the folder exists
            os.makedirs(output_folder, exist_ok=True)

            # Construct the output CSV file path
            output_csv_file = os.path.join(output_folder, new_filename)

            # Prepare data for saving
            rows = []
            for box, label in zip(unique_extracted_boxes, unique_extracted_lbl):
                xmin, ymin, xmax, ymax = box
                rows.append([filename_without_extension + '.jpg', label, int(xmin), int(ymin), int(xmax), int(ymax)])

            # Save the data into a CSV file
            with open(output_csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])  # Write the header
                writer.writerows(rows)  # Write each row



            # Define the folder path for saving predicted boxes
            combined_csv_file = r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset\predicted_boxes\00_combined_predictions.csv'

            # print("all_true_boxes:", gt)
            #------------------------------------------------------
            # Function to calculate IoU
            def calculate_iou(box1, box2):
                x1, y1, x2, y2 = box1
                x1g, y1g, x2g, y2g = box2

                xi1 = max(x1, x1g)
                yi1 = max(y1, y1g)
                xi2 = min(x2, x2g)
                yi2 = min(y2, y2g)

                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (x2g - x1g) * (y2g - y1g)
                union_area = box1_area + box2_area - inter_area

                return inter_area / union_area


            # Step 1: Calculate IoU and find overlaps
            # Step 1: Calculate IoU and find overlaps
            overlap_indices = []
            all_true_labels =[]
            gt_indices =[]
            for i, extracted_box in enumerate(unique_extracted_boxes):
                for j,true_box in enumerate(gt):
                # print("gt:",gt)
                    iou = calculate_iou(extracted_box,true_box)
                    treshold_iou1=0.1
                    if iou > treshold_iou1:
                        overlap_indices.append(i)
                        gt_indices.append(j)
                        break
            print("overlap_indices:", overlap_indices)
            print("gt_indices:", gt_indices)
            # Step 2: Update unique_extracted_lbl based on overlap
            first_part = [gt_labels[idx] for idx in gt_indices]
            second_part = [gt_labels[idx] for idx in range(len(gt_labels)) if idx not in gt_indices]
            update_all_tru_labels = first_part + second_part
            print("update_all_tru_labels:",update_all_tru_labels)

            first_part_extracted_lbl = [0] * len(update_all_tru_labels)
            if len(overlap_indices)==0:
                second_part_part_extracted_lbl = unique_extracted_lbl
                updated_labels1.extend(first_part_extracted_lbl)
                # print("second_part_part_extracted_lbl:", second_part_part_extracted_lbl)
                updated_labels1.extend(second_part_part_extracted_lbl)
                num_tree_un_detect += len(update_all_tru_labels)
            # Update the values at the specified indices
            for i, idx in enumerate(overlap_indices):

                first_part_extracted_lbl[i] = unique_extracted_lbl[idx]

                print("first_part_extracted_lbl:",first_part_extracted_lbl)
                second_part_part_extracted_lbl = [0 if i in overlap_indices else unique_extracted_lbl[i] for i in
                                                  range(len(unique_extracted_lbl))]
                print("second_part_part_extracted_lbl:", second_part_part_extracted_lbl)

                second_part_part_extracted_lbl = [label for label in second_part_part_extracted_lbl if
                                                  label != 0]  # delete zeros
                print("second_part_part_extracted_lbl after delet zero:", second_part_part_extracted_lbl)

            if len(overlap_indices)!=0:
                updated_labels1.extend(first_part_extracted_lbl)
                updated_labels1.extend(second_part_part_extracted_lbl)

                num_tree_un_detect += len([label for label in first_part_extracted_lbl if label == 0])
                print("updated_labels1:", updated_labels1)
            # Step 3: Add missing labels to all_true_labels and fill with 5 ifnecessary
            if len(updated_labels1) > len(update_all_tru_labels):
                    update_all_tru_labels.extend([0] * (len(updated_labels1) - len(update_all_tru_labels)))

            # Step 4: Create the final updated labels
            final_all_true_labels.extend(update_all_tru_labels)
            final_all_pred_labels.extend(updated_labels1)
            print(" all_true_labels:",update_all_tru_labels)
            print(" pred labels:",updated_labels1)
            if(len(update_all_tru_labels) != len(updated_labels1)):
                print("error$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                break
            # print("final_all_pred_labels:", final_all_pred_labels)
            # print("final_all_true_labels:", final_all_true_labels)

            if counter<10000:
                for i in range(len(extracted_boxes)):
                    # print("*************************")
                    # print("i:",i)
                    if unique_extracted_lbl[i] == 1:
                        rect = patches.Rectangle((int(extracted_boxes[i][0]), int(extracted_boxes[i][1])), abs(extracted_boxes[i][0] - extracted_boxes[i][2]),
                                                 abs(extracted_boxes[i][1] - extracted_boxes[i][3]), linewidth=3, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        # print("class1")
                        ax.text(int(extracted_boxes[i][0]), int(extracted_boxes[i][1]), 'Class 1', color='r', fontsize=12, weight='bold')
                    elif unique_extracted_lbl[i] == 2:
                        rect = patches.Rectangle((int(extracted_boxes[i][0]), int(extracted_boxes[i][1])), abs(extracted_boxes[i][0] - extracted_boxes[i][2]),
                                                 abs(extracted_boxes[i][1] - extracted_boxes[i][3]), linewidth=3, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                        # print("class2")
                        ax.text(int(extracted_boxes[i][0]), int(extracted_boxes[i][1]), 'Class 2', color='g', fontsize=12, weight='bold')
                    elif unique_extracted_lbl[i] == 3:
                        rect = patches.Rectangle((int(extracted_boxes[i][0]), int(extracted_boxes[i][1])), abs(extracted_boxes[i][0] - extracted_boxes[i][2]),
                                                 abs(extracted_boxes[i][1] - extracted_boxes[i][3]), linewidth=3, edgecolor='b', facecolor='none')
                        ax.add_patch(rect)
                        # print("class3")
                        ax.text(int(extracted_boxes[i][0]), int(extracted_boxes[i][1]), 'Class 3', color='b', fontsize=12, weight='bold')
                    elif unique_extracted_lbl[i] == 4:
                        rect = patches.Rectangle((int(extracted_boxes[i][0]), int(extracted_boxes[i][1])), abs(extracted_boxes[i][0] - extracted_boxes[i][2]),
                                                 abs(extracted_boxes[i][1] - extracted_boxes[i][3]), linewidth=3, edgecolor='y', facecolor='none')
                        ax.add_patch(rect)
                        # print("class4")
                        ax.text(int(extracted_boxes[i][0]), int(extracted_boxes[i][1]), 'Class 4', color='y', fontsize=12, weight='bold')
                filename = filename[0] if isinstance(filename, tuple) else filename
                fig.savefig(r"E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset\output_images\{}.png".format(filename.split('.')[0]), dpi=90, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free up memory
            # print("44444444444444444444444444444444444444444444444444444444444444444444444444444444444444")
            # # Debugging print statements

    # Collect all CSV files in the output_folder
    csv_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.csv')]

    # Initialize an empty DataFrame to hold combined data
    combined_df = pd.DataFrame()

    # Iterate over each CSV file and concatenate them
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(combined_csv_file, index=False)

    print(f"Combined CSV file saved to: {combined_csv_file}")

    print("all_pred_labels:",updated_labels1)
    print("Length of final_all_true_labels:", len(final_all_true_labels))
    print("Length of final_all_pred_labels:", len(final_all_pred_labels))

    # Number of elements in final_all_pred_labels without zeros
    number_of_predicted_tree = len([label for label in final_all_pred_labels if label != 0])

    # Number of values equal to 1 in final_all_true_labels
    number_of_1_in_gt = final_all_true_labels.count(1)

    # Number of values equal to 2 in final_all_true_labels
    number_of_2_in_gt = final_all_true_labels.count(2)

    # Number of values equal to 3 in final_all_true_labels
    number_of_3_in_gt = final_all_true_labels.count(3)
    # Number of values equal to 4 in final_all_true_labels
    number_of_4_in_gt = final_all_true_labels.count(4)
    number_of_0_in_gt = final_all_true_labels.count(0)
    if len(final_all_true_labels) == len(final_all_pred_labels):
        cm = confusion_matrix(final_all_true_labels, final_all_pred_labels)
        unique_labels = sorted(set(final_all_true_labels) | set(final_all_pred_labels))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)

        title = (
            f"The number of trees in gt that is not predicted: {num_tree_un_detect}\n"
            f"number_of_predicted_tree: {number_of_predicted_tree}\n "
            f"number_of_overlap with_1_in_gt: {number_of_1_in_gt}\n"
            f"number_of_overlap with_2_in_gt: {number_of_2_in_gt}\n "
            f"number_of_overlap with_3_in_gt: {number_of_3_in_gt}\n"
            f"number_of_overlap with_4_in_gt: {number_of_4_in_gt}\n"
            f" threshold_iou:{treshold_iou1}, treshold_score:{treshold_score}")
        # title = (
        #     f"The number of trees in gt that is not predicted: {num_tree_un_detect}\n"
        #     f"number_of_predicted_tree: {number_of_predicted_tree}\n "
        #     f"number_of_1_in_gt: {533}\n"
        #     f"number_of_2_in_gt: {467}\n "
        #     f"number_of_3_in_gt: {134}\n"
        #     f"number_of_4_in_gt: {769}\n"
        #     f" threshold_iou:{treshold_iou1}, treshold_score:{treshold_score}")
        # Display confusion matrix
        # plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title, fontsize=12)  # Adjust the fontsize as needed
        plt.tight_layout()
        plt.show()
    else:
        print("Error: Inconsistent number of samples between true and predicted labels.")


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5 )
model.roi_heads.nms_thresh = 0.1
model.roi_heads.score_thresh = 0.1
model.load_state_dict(torch.load(r"E:\FASTRCNN\FASTRCNN\dataset\images_masked\checkpoints\best_model.pth"))
model.to(device)

evaluate(model, test_loader)