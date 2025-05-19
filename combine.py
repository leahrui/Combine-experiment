import os
import cv2
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image, ImageDraw, ImageFont
import random

# 配置参数
class Config:
    # 模型路径
    DET_MODEL_PATH = "/home/dyy/sda/dyy/v11/ultralytics-main/runs/train/33coco/weights/best.pt"
    SEG_FH_MODEL_PATH = "/home/dyy/sda/dyy/v11/ultralytics-main/runs/train/FH33/weights/best.pt"
    SEG_JS_MODEL_PATH = "/home/dyy/sda/dyy/v11/ultralytics-main/runs/train/JS33/weights/best.pt"

    # 数据路径
    TRAIN_TXT = "/home/dyy/sda/dyy/objectdetectdataset/33dataset/train.txt"  # train_t.txt
    VAL_TXT = "/home/dyy/sda/dyy/objectdetectdataset/33dataset/val.txt"
    TEST_TXT = "/home/dyy/sda/dyy/objectdetectdataset/33dataset/newtest0410.txt" #"/home/dyy/sda/dyy/objectdetectdataset/33dataset/newtest.txt"

    '''DET_MODEL_PATH = "/home/dyy/sda/dyy/v11/ultralytics-main/runs/train/124dataset/weights/best.pt"
    SEG_FH_MODEL_PATH = "/home/dyy/sda/dyy/v11/ultralytics-main/runs/train/FH2203/weights/best.pt"
    SEG_JS_MODEL_PATH = "/home/dyy/sda/dyy/v11/ultralytics-main/runs/train/segJS2202/weights/best.pt"

    # 数据路径
    TRAIN_TXT = "/home/dyy/sda/dyy/objectdetectdataset/124dataset/train.txt"
    VAL_TXT = "/home/dyy/sda/dyy/objectdetectdataset/124dataset/val.txt"
    TEST_TXT = "/home/dyy/sda/dyy/objectdetectdataset/124dataset/test.txt"
    '''

    # 模型保存目录
    MODEL_SAVE_DIR = "/home/dyy/sda/dyy/v11/ultralytics-main/models/20250410test/"
    TEST_RESULTS_CSV = "/home/dyy/sda/dyy/v11/ultralytics-main/models/20250410test/test_Result.csv"
    # 检测参数
    DET_CONF_THRESH = 0.1  # 新增检测置信度阈值
    FEATURE_IMPORTANCE_THRESHOLD = 0.05
    ABNORMAL_P = 0.35


class FeatureExtractor:
    def __init__(self):
        self.det_model = YOLO(Config.DET_MODEL_PATH)
        self.seg_fh_model = YOLO(Config.SEG_FH_MODEL_PATH)
        self.seg_js_model = YOLO(Config.SEG_JS_MODEL_PATH)

    """
    def _get_dominant_detection(self, det_results):
        #获取置信度最高的检测结果，如果没有置信度高于0.7的框，则返回置信度最高的框
        if len(det_results[0].boxes) == 0:
            return None, 0.0  # 如果没有检测到目标，返回 None 和置信度 0.0

        boxes = det_results[0].boxes
        confs = boxes.conf.cpu().numpy()

        # 找到置信度最高的框
        max_idx = np.argmax(confs)
        max_conf = confs[max_idx]
        max_cls = int(boxes.cls[max_idx].item())

        # 如果最高置信度低于阈值，仍然返回该框
        if max_conf < Config.DET_CONF_THRESH:
            print(f"Warning: No detection with confidence >= {Config.DET_CONF_THRESH}. "
                  f"Returning detection with highest confidence: {max_conf:.4f}")

        return max_cls, max_conf

    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            return None

        # 目标检测特征
        det_results = self.det_model(img)
        det_cls, det_conf = self._get_dominant_detection(det_results)
        if det_cls is None:
            print(f"Warning: No detection found in image {img_path}")
            return None

        # 股骨头分割特征（保持原始4个类别）
        seg_fh_results = self.seg_fh_model(img)
        fh_confidences = np.zeros(4)
        if len(seg_fh_results[0].boxes) > 0:
            for box in seg_fh_results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if cls < 4:
                    fh_confidences[cls] = max(fh_confidences[cls], conf)

        # 间隙分割特征
        seg_js_results = self.seg_js_model(img)
        js_confidences = np.zeros(2)
        if len(seg_js_results[0].boxes) > 0:
            for box in seg_js_results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if cls < 2:
                    js_confidences[cls] = max(js_confidences[cls], conf)

        return [
            det_cls, det_conf,
            *fh_confidences,
            *js_confidences
        ]
    """
    def _get_dominant_detection(self, detect_results):
        """
        找到置信度最高的两个框，并返回它们的类别和置信度。
        如果没有检测结果，返回默认值(None, 0.0)。
        """
        boxes = detect_results[0].boxes
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        # print("boxes---:", boxes)
        # print("confs---:", confs)
        # print("cls---:", cls)
        detection_results = []
        for i in range(len(detect_results[0].boxes)):
            detection_results.append((int(clss[i]), confs[i]))
        # print("detection_results---:", detection_results)
        filtered_results = []
        for item in detection_results:
            if len(item) < 2:
                continue  # 忽略不符合格式的项
            try:
                cls, conf = item[0], item[1]
                # print(" cls, conf :", cls, conf)
                if conf >= Config.DET_CONF_THRESH:
                    filtered_results.append((cls, conf))
            except Exception as e:
                    print(f"检测结果格式错误：{e}")
        # print("filtered_results--:",filtered_results)
        # 按置信度降序排序
        sorted_results = sorted(filtered_results, key=lambda x: -x[1])
        # 提取前两个检测结果
        dominant_detections = []
        for i in range(2):
            if i < len(sorted_results):
                dominant_detections.append((sorted_results[i][0], sorted_results[i][1]))
            else:
                dominant_detections.append((None, 0.0))  # 如果没有足够的检测结果，使用默认值
        # det_confs 保持原始4个类别置信度
        det_confs = np.zeros(4)
        for j in range(len(clss)):
            cl = int(clss[j])
            conf = confs[j]
            if cl < 4:
                det_confs[cl] = max(det_confs[cl], conf)
        return dominant_detections, det_confs

    def extract_features(self, img_path):
        """
        提取图像的多类型特征。

        Args:
            img_path (str): 输入图像路径

        Returns:
            list: 包含目标检测、股骨头分割和间隙分割特征的列表
        """
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像文件：{img_path}")

        # 目标检测部分
        det_results = self.det_model(image)
        dominant_detections, det_confs = self._get_dominant_detection(det_results)

        # 股骨头分割特征（保持原始4个类别）
        seg_fh_results = self.seg_fh_model(image)
        print("seg_fh_results:",seg_fh_results)
        fh_confidences = np.zeros(4)
        if len(seg_fh_results[0].boxes) > 0:
            for box in seg_fh_results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if cls < 4:
                    fh_confidences[cls] = max(fh_confidences[cls], conf)
        fh_labels={0:'normal femoral head',1:'abnormal femoral head in stage II', 2:'abnormal femoral head in stage III',
                   3:'femoral head collapse'}
        FH = fh_labels[np.argmax(fh_confidences).item()]
        # print('FH----------:',FH)
        # 间隙分割特征
        seg_js_results = self.seg_js_model(image)
        # print("seg_js_results:", seg_js_results)
        js_confidences = np.zeros(2)
        if len(seg_js_results[0].boxes) > 0:
            for box in seg_js_results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if cls < 2:
                    js_confidences[cls] = max(js_confidences[cls], conf)
        js_labels = {0: 'normal femoral head space', 1: 'abnormal femoral head space'}
        JS = js_labels[np.argmax(js_confidences).item()]
        # print('JS----------:', JS)

        # 组合特征
        features = []
        TOP2 = ''
        #for cls_id, conf in dominant_detections:
        #    features.append(cls_id)
        #    features.append(conf)
        features.extend(det_confs)
        features.extend(fh_confidences)
        features.extend(js_confidences)
        for cls_id, conf in dominant_detections:
            features.append(cls_id)
            features.append(conf)
        # print("features:", features)
        combine_confs = [0] * 4
        combine_confs2 = [0] * 4
        t = 0.5
        t1 = 0.25
        t2 = 0.25
        combine_confs[0] = features[0]*t + features[5]*t1 + features[8]*t2
        combine_confs[1] = features[1]*t + features[6]*t1 + features[8]*t2
        combine_confs[2] = features[2]*t + features[7]*t1 + features[9]*t2
        combine_confs[3] = features[3]*t + features[4]*t1 + features[8]*t2

        features.extend(combine_confs)
        t3, t4, t5, t6 = 0.5, 0.5, 0.5, 0.5
        if features[0]*features[5] == 0:
            t3 = 1
            #print('t3----:', t3)
        if features[1]*features[6] == 0:
            t4 = 1
            #print('t4----:', t4)
        if features[2]*features[7] == 0:
            t5 = 1
            #print('t5----:', t5)
        if features[3]*features[4] == 0:
            t6 = 1
            #print('t6----:', t6)
        combine_confs2[0] = (features[0]+features[5])*t3
        combine_confs2[1] = (features[1]+features[6])*t4
        combine_confs2[2] = (features[2]+features[7])*t5
        combine_confs2[3] = (features[3]+features[4])*t6
        max1, max2 = float('-inf'), float('-inf')
        pos1, pos2 = -1, -1
        for index, num in enumerate(combine_confs2):
            if num > max1:
                max2, max1 = max1, num
                pos2, pos1 = pos1, index
            elif num > max2:
                max2 = num
                pos2 = index
        ficat_labels = {0: 'stage II', 1: 'stage III', 2: 'stage IV', 3: 'nVNFH(stage 0 or I)'}
        if max2 > 0.1:
            TOP2 = str(ficat_labels[pos1])+':'+str(round(max1, 1))+'   '+str(ficat_labels[pos2])+":"+str(round(max2,1))
        else:
            TOP2 = str(ficat_labels[pos1])+':'+str(round(max1, 1))
        # print('TOP2----:', TOP2)
        return TOP2, img_path, FH, JS, seg_fh_results, seg_js_results, features


class DataProcessor:
    @staticmethod
    def load_data(txt_path):
        features = []
        labels = []
        file_names=[]
        FHS = []
        JSS = []
        imgpath=[]
        TOP22 = []

        extractor = FeatureExtractor()

        file_list = DataLoader.load_file_list(txt_path)
        for img_path in tqdm(file_list, desc="Extracting features"):
            if not os.path.exists(img_path):
                continue

            TOP2,img_path,FH,JS,seg_fh_result,seg_js_result,feature = extractor.extract_features(img_path)
            if feature is None:
                continue

            label_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    label = int(lines[0].split()[0])
                    features.append(feature)
                    labels.append(label)
                    file_names.append(os.path.basename(img_path))
                    FHS.append(FH)
                    JSS.append(JS)
                    imgpath.append(img_path)
                    TOP22.append(TOP2)
        # print('top22-----------------:',TOP22)
        return TOP22, imgpath, FHS, JSS, np.array(features), np.array(labels), file_names


class FeatureSelector:
    @staticmethod
    def select_features(X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nFeature ranking:")
        for f in range(X.shape[1]):
            print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]:.4f})")

        selected_indices = np.where(importances > Config.FEATURE_IMPORTANCE_THRESHOLD)[0]
        print(f"\nSelected features: {selected_indices}")

        return selected_indices


class ModelTrainer:
    @staticmethod
    def get_model_configs():
        return [
            {
                'name': 'DecisionTree',
                'model': DecisionTreeClassifier(random_state=42),
                'param_grid': {
                    'max_depth': [5, 8, 10, 12, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced']#[dict,{0:6,1:7,2:3,3:6}] #[2, 2, 1, 2]#
                }
            },
            {
                'name': 'RandomForest',
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': list(range(50, 151, 10)),
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced']
                }
            },
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    objective='multi:softmax',
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=42
                ),
                'param_grid': {
                    'n_estimators': list(range(50, 151, 20)),
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            }
        ]

    @staticmethod
    def train_models():
        _, _, _, _, X_train, y_train, xfilename = DataProcessor.load_data(Config.TRAIN_TXT)
        selected_indices = FeatureSelector.select_features(X_train, y_train)
        X_train_selected = X_train[:, selected_indices]

        best_models = {}
        for config in ModelTrainer.get_model_configs():
            print(f"\n=== Training {config['name']} ===")
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                cv=5,
                n_jobs=-1,
                verbose=2,
                scoring='accuracy'
            )
            grid_search.fit(X_train_selected, y_train)

            model_path = os.path.join(
                Config.MODEL_SAVE_DIR,
                f"{config['name']}_best_model.pkl"
            )
            joblib.dump(grid_search.best_estimator_, model_path)

            print(f"\nBest parameters for {config['name']}:")
            print(grid_search.best_params_)
            best_models[config['name']] = {
                'model': grid_search.best_estimator_,
                'selected_indices': selected_indices
            }

        return best_models


class ModelEvaluator:
    @staticmethod
    def evaluate_all(models, txt_path):
        TOP2,imgpath,FH,JS,X, y, file_names = DataProcessor.load_data(txt_path)
        results = {}
        ficat = []

        for name, model_info in models.items():
            model = model_info['model']
            indices = model_info['selected_indices']
            X_selected = X[:, indices]

            y_pred = model.predict(X_selected)
            # print("y_pred-----------------:", y_pred)

            t1 = 0.6
            t2 = 0.4
            t3 = 0.25
            t4 = 0.5
            t5 = 0.25
            for i in range(len(y_pred)):
                print("y_pred_1:,i:", y_pred[i], i)
                if y_pred[i] == 3:
                    # print("file_names[i]--:", file_names[i])
                    # print("X[i]--:", X[i])
                    # confs = FeatureExtractor.extract_features(file_names[i])==X[i]
                    # X[i]0-9:II\III\IV\nVNFH\NFH\IIFH\AFH\fusion\NJS\AJS
                    #         0    1   2   3   4    5    6    7    8    9
                    confs = [0]*4
                    confs[0] = X[i][0]+X[i][5]
                    confs[1] = X[i][1]+X[i][6]
                    confs[2] = X[i][2]+X[i][7]+X[i][9]
                    max_confs = max(confs)
                    max_cls = confs.index(max_confs)

                    if max_confs > Config.ABNORMAL_P:
                        y_pred[i] = max_cls
                        print("change cls nVNFH to", max_cls)


                if y_pred[i] == 1:
                    # print("file_names[i]--:", file_names[i])
                    # print("X[i]--:", X[i])
                    # confs = FeatureExtractor.extract_features(file_names[i])==X[i]
                    confs = [0]*4
                    confs[0] = X[i][0]*t1+X[i][5]*t2
                    confs[3] = X[i][3]*X[i][4]*X[i][8]
                    confs[2] = X[i][2]*t4+X[i][7]*t5+X[i][9]*t3
                    confs[1] = X[i][1] * t1 + X[i][6] * t2
                    max_confs = max(confs)
                    max_cls = confs.index(max_confs)

                    if max_confs > Config.ABNORMAL_P:
                        #y_pred[i] = max_cls
                        print("no change ")


                if y_pred[i] == 2:
                    #print("file_names[i]--:", file_names[i])
                    #print("X[i]--:", X[i])
                    # confs = FeatureExtractor.extract_features(file_names[i])==X[i]
                    confs = [0]*4
                    confs[0] = X[i][0]*t1+X[i][5]*t2
                    confs[3] = X[i][3]*X[i][4]*X[i][8]
                    confs[1] = X[i][1]*t1+X[i][6]*t2
                    #confs[2] = X[i][2] * t4 + X[i][7] * t5 + X[i][9] * t3
                    max_confs = max(confs)
                    max_cls = confs.index(max_confs)

                    if max_confs > Config.ABNORMAL_P:
                        #y_pred[i] = max_cls
                        print("no change")


                if y_pred[i] == 0:
                    # print("file_names[i]--:", file_names[i])
                    # print("X[i]--:", X[i])
                    # confs = FeatureExtractor.extract_features(file_names[i])==X[i]
                    confs = [0]*4
                    confs[2] = X[i][2]*t4+X[i][7]*t5+X[i][9]*t3
                    confs[3] = X[i][3]*X[i][4]*X[i][8]
                    confs[1] = X[i][1]*t1+X[i][6]*t2
                    confs[0] = X[i][0]*t1+X[i][5]*t2
                    max_confs = max(confs)
                    max_cls = confs.index(max_confs)

                    if max_confs > Config.ABNORMAL_P:
                        #y_pred[i] = max_cls
                        print("no change")

                ficat_labels = {0:'stage II', 1:'stage III', 2:'stage IV', 3:'nVNFH(stage 0 or I)'}
                ficat.append(ficat_labels[y_pred[i]])
                print("y_pred_2:,i:", y_pred[i], i)
                print("ficat_labels[y_pred[i]]:", ficat_labels[y_pred[i]], i)
                '''
                fh_n=FH[i]
                js_n=JS[i]
                ModelEvaluator.draw_results_on_image(ficat,fh_n,js_n,imgpath[i],file_names[i])
                '''
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'confusion_matrix': confusion_matrix(y, y_pred, labels=[0, 1, 2, 3]),
                # 'accuracy_ficat': accuracy_score(y, ficat),
                # 'confusion_matrix_ficat': confusion_matrix(y, ficat, labels=[0, 1, 2, 3]),
                'report': classification_report(y, y_pred, target_names=['II期', 'III期', 'IV期', 'nVNFH'], digits=4),
                # 'report_ficat': classification_report(y, ficat, target_names=['II期', 'III期', 'IV期', 'nVNFH']),
                'y_true': y,
                'y_pred': y_pred,
                'file_names': file_names,
                'feature_fh': FH,
                'feature_js': JS,
                'img_path': imgpath,
                'ficat': ficat,
                'TOP2': TOP2,
            }
        return results

    @staticmethod
    def draw_results_on_model_image(results, img_path, title, class_names):
        # 读取原始图像并转换颜色空间
        # print('boxes----', results[0].boxes)
        # print('masks----', results[0].masks)
        # print('results----', results)
        #if isinstance(results, list):
        #    results = results[0]  # 提取第一个 Results 对象

        # 确保输入是 Results 对象列表
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("输入必须是包含 Results 对象的列表")

        # 提取第一个 Results 对象
        result = results[0]  # 关键修正：使用索引提取

        # 验证必要属性存在
        #if not hasattr(result, "boxes") or not hasattr(result, "masks"):
        #    raise ValueError("模型结果缺少 boxes 或 masks 属性")

        # 检查是否有检测结果
        #if len(result.boxes) == 0 and len(result.masks) == 0:
        #    print(f"警告：{title} 未检测到目标，返回原始图像")
        #    return cv2.imread(img_path)

        # 读取原始图像
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"原始图像未找到: {img_path}")
        h, w = image.shape[:2]

        # 创建叠加层和绘图对象
        overlay = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)

        # 设置字体和颜色（关键修正：使用元组）
        try:
            font = ImageFont.truetype("Arial", 60)
        except OSError:
            font = ImageFont.truetype("DejaVuSans.ttf", 60)
        #colors = {
        #    i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #    for i in range(len(class_names))
        #}
        # 定义固定颜色（按类别顺序）
        fixed_colors = [
            (255, 0, 0),  # 红色 - NFH
            (0, 255, 0),  # 绿色 - IIFH
            (0, 0, 255),  # 蓝色 - AFH
            (255, 255, 0)  # 黄色 - fusion
        ]

        # 确保颜色数量与类别数量匹配
        if len(fixed_colors) < len(class_names):
            raise ValueError(f"固定颜色数量({len(fixed_colors)})少于类别数量({len(class_names)})")

        # 创建颜色字典（类别索引 -> 固定颜色）
        colors = {i: fixed_colors[i] for i in range(len(class_names))}

        # 遍历检测结果
        if result.masks is not None:
            for box, mask in zip(result.boxes, result.masks):
                # 解析边界框坐标
                box_coords = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, box_coords)

                # 解析类别和置信度
                cls = int(box.cls)
                conf = float(box.conf)

                # 获取颜色（确保为元组）
                color = colors.get(cls, (0, 255, 0))

                # 绘制边界框和文本
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                text = f"{class_names[cls]}:{conf:.2f}"
                draw.text((x1, y1 - 55), text, font=font, fill=color)

                # 处理分割掩码（关键修正）
                mask_data = mask.data.cpu().numpy().squeeze()
                if mask_data.ndim != 2:
                    mask_data = mask_data  # 处理冗余维度
                mask_resized = cv2.resize(mask_data, (w, h))  # (width, height)
                binary_mask = (mask_resized > 0.3).astype(np.uint8) * 255  # 调整阈值
                # print("Mask shape:", mask_data.shape)
                # print("Resized mask shape:", mask_resized.shape)
                # print("Binary mask unique values:", np.unique(binary_mask))

                # 创建颜色叠加层（BGR格式）
                color_bgr = color[::-1]  # RGB -> BGR
                color_mask = np.zeros_like(overlay)
                color_mask[:] = color_bgr
                masked_region = cv2.bitwise_and(color_mask, color_mask, mask=binary_mask)

                # 累加叠加效果（保留之前的结果）
                cv2.addWeighted(masked_region, 0.5, overlay, 0.5, 0, overlay)
        elif result.masks is None and result.boxes is not None:
            # 提取所有边界框坐标和分割掩码数据
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()  # 形状: [N]
            clses = result.boxes.cls.cpu().numpy()  # 形状: [N]
            # 遍历所有检测结果
            for i in range(len(boxes_xyxy)):
                # 解析坐标、置信度、类别
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                conf = confs[i]
                cls = int(clses[i])
                color = colors.get(cls, (0, 255, 0))

                # 绘制边界框和文本
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                text = f"{class_names[cls]}:{conf:.2f}"
                draw.text((x1, y1 - 55), text, font=font, fill=color)

        # 合并结果
        result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        result_image = cv2.addWeighted(result_image, 0.5, overlay, 0.5, 0)

        #cv2.putText(result_image, title, (10, 30), 1, 1, (255, 255, 255), 5, 5)
        cv2.putText(
            img=result_image,
            text=title,
            org=(10, 60),  # 文本左下角坐标 (x, y)
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 使用标准字体
            fontScale=2.0,  # 字体缩放比例
            color=(255, 255, 255),  # 白色 (BGR 格式)
            thickness=2,  # 线条粗细
            lineType=cv2.LINE_AA  # 抗锯齿线型
        )

        return result_image


    @staticmethod
    def draw_results_on_image(results, save_path):
        ficat_s = results['ficat']
        FH_s = results['feature_fh']
        JS_s = results['feature_js']
        img_path_s = results['img_path']
        file_name_s = results['file_names']
        TOP2 = results['TOP2']
        # print('TOP222222----:',TOP2)


        seg_fh_model = YOLO(Config.SEG_FH_MODEL_PATH)
        seg_js_model = YOLO(Config.SEG_JS_MODEL_PATH)
        # 目标检测部分
        det_model = YOLO(Config.DET_MODEL_PATH)

        for i in range(len(img_path_s)):
            ficat = ficat_s[i]
            FH = FH_s[i]
            JS = JS_s[i]
            img_path = img_path_s[i]
            file_name = file_name_s[i]
            TOP = TOP2[i]

            final_path = os.path.join(save_path, file_name)

            # 读取原始图像
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"原始图像未找到: {img_path}")

            # 运行分割模型
            seg_fh_result = seg_fh_model(image)
            seg_js_result = seg_js_model(image)
            det_result = det_model(image)
            det_image = ModelEvaluator.draw_results_on_model_image(det_result, img_path, "Combine detection",
                                                                   ['II', 'III', 'IV', 'nVNFH'])
            # 绘制文本到原始图像
            image_rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("Arial", 40)
            except OSError:
                font = ImageFont.truetype("DejaVuSans.ttf", 40)
            text = f"Combine Ficat:{ficat}\nFH:{FH}\nFHS:{JS}\nTOP2:{TOP}"
            # draw.rectangle([])
            text_position = (10, 80)
            '''            
            text_width,text_height=draw.textsize(text,font=font)
            box_position=(text_position[0]-5,text_position[1]-5)
            box_size=(text_width+10,text_height+10)
            draw.rectangle([box_position,(box_position[0]+box_size[0],
                                          box_position[1]+box_size[1])],fill='white')
            draw.text(text_position, text, font=font, fill="black")
            '''
            # 文本尺寸计算
            bbox = draw.textbbox((0, 0), text, font=font)  # 获取边界框
            # print('bbox---:',bbox)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # 绘制背景框
            box_position = (text_position[0] - 5, text_position[1] - 5)
            box_size = (text_width + 10, text_height + 10)
            # gray_color=(128,128,128)
            draw.rectangle(
                [box_position, (box_position[0] + box_size[0], box_position[1] + box_size[1])],
                fill="white"
            )

            # 绘制文本
            draw.text(text_position, text, font=font, fill="red")
            combine_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 生成分割结果图像（确保尺寸与原始图像一致）
            seg_fh_image = ModelEvaluator.draw_results_on_model_image(seg_fh_result, img_path, "FH segmentation",['NFH','IIFH','AFH','fusion'])
            seg_js_image = ModelEvaluator.draw_results_on_model_image(seg_js_result, img_path, "FHS segmentation",['NJS','AJS'])

            # 计算目标尺寸
            h, w = image.shape[:2]
            target_width = int(1.5 * w) - w  # 右侧区域宽度
            target_height_upper = h // 2  # 上半部分高度
            target_height_lower = h - h // 2  # 下半部分高度

            # 调整分割图像尺寸
            seg_fh_image_resized = cv2.resize(seg_fh_image, (target_width, target_height_upper))
            seg_js_image_resized = cv2.resize(seg_js_image, (target_width, target_height_lower))

            # 创建空白画布并拼接图像
            final_image = np.zeros((h, int(1.5 * w), 3), dtype=np.uint8)
            final_image[:h, :w] = combine_image  # 左侧为原始图像+文本
            final_image[:target_height_upper, w: w + target_width] = seg_fh_image_resized  # 右上方为 FH 分割
            final_image[target_height_upper: h, w: w + target_width] = seg_js_image_resized  # 右下方为 JS 分割

            # 显示并保存
            cv2.imshow("final image", final_image)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            cv2.imwrite(final_path, final_image)

    @staticmethod
    def save_results_to_csv(results, csv_path):
        y_true = results['y_true']
        y_pred = results['y_pred']
        file_names = results['file_names']
        results_df = pd.DataFrame({
            'File Name': file_names,
            'True Label': y_true,
            'Predicted Label': y_pred
        })
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    


class DataLoader:
    @staticmethod
    def load_file_list(txt_path):
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

    print("=== Start Training ===")
    models = ModelTrainer.train_models()

    print("\n=== Validation Evaluation ===")
    val_results = ModelEvaluator.evaluate_all(models, Config.VAL_TXT)

    print("\n=== Test Evaluation ===")
    test_results = ModelEvaluator.evaluate_all(models, Config.TEST_TXT)


    def print_results(results, set_name):
        print(f"\n=== {set_name} Results ===")
        for name, metric in results.items():
            print(f"\n{name}:")
            print(f"Accuracy: {metric['accuracy']:.4f}")
            print("Confusion Matrix:")
            print(metric['confusion_matrix'])
            print("Classification Report:")
            print(metric['report'])
            '''            print(f"\n{name}:")
            print(f"Accuracy ficat: {metric['accuracy_ficat']:.4f}")
            print("Confusion Matrix ficat:")
            print(metric['confusion_matrix_ficat'])
            print("Classification Report:")
            print(metric['report_ficat'])'''


    print_results(val_results, "Validation Set")
    print_results(test_results, "Test Set")
    for name, result in test_results.items():
        csv_path = os.path.join(Config.MODEL_SAVE_DIR, f"{name}_test_results.csv")
        ModelEvaluator.save_results_to_csv(result, csv_path)
        save_path = "/home/dyy/sda/dyy/v11/result/178imageresult/combine0410/"+f"{name}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ModelEvaluator.draw_results_on_image(result, save_path)


