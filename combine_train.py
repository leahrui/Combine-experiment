import os
import joblib
import numpy as np
from ultralytics import YOLO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import xgboost as xgb
from tqdm import tqdm
import cv2
import fhsdistance

# 配置参数
class Config:
    # 模型路径
    DET_MODEL_PATH = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/train/yolov11-xndp/weights/best.pt'
    SEG_FH_MODEL_PATH = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/train/segFH-yolo11-xndp2/weights/best.pt'
    SEG_JS_MODEL_PATH = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/train/segJS-yolo11-xndp/weights/best.pt'

    # 数据路径
    TRAIN_TXT = '/home/dyy/sda/dyy/objectdetectdataset/33dataset/newtest0410.txt' #'/home/dyy/sda/dyy/objectdetectdataset/33dataset/dataset20250821/xn-dptrain.txt'
    MODEL_SAVE_DIR = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/combine1111'

    # 检测参数
    DET_CONF_THRESH = 0.1  # 新增检测置信度阈值
    FEATURE_IMPORTANCE_THRESHOLD = 0.05
    ABNORMAL_P = 0.35

class FeatureExtractor:
    def __init__(self):
        self.det_model = YOLO(Config.DET_MODEL_PATH)
        self.seg_fh_model = YOLO(Config.SEG_FH_MODEL_PATH)
        self.seg_js_model = YOLO(Config.SEG_JS_MODEL_PATH)


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
        print("seg_fh_results:", seg_fh_results)
        fh_confidences = np.zeros(4)
        if len(seg_fh_results[0].boxes) > 0:
            for box in seg_fh_results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if cls < 4:
                    fh_confidences[cls] = max(fh_confidences[cls], conf)

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

        fhsdistance_results = fhsdistance.process_image(self.seg_js_model, img_path, None)  # 计算间隙最短距离及面积---20251010
        # 组合特征
        features = []

        features.extend(det_confs)
        features.extend(fh_confidences)
        features.extend(js_confidences)
        for cls_id, conf in dominant_detections:
            features.append(cls_id)
            features.append(conf)
        # print("features:", features)
        combine_confs = [0] * 4
        t = 0.6
        t1 = 0.2
        t2 = 0.2
        combine_confs[0] = features[0]*t + features[5]*t1 + features[8]*t2
        combine_confs[1] = features[1]*t + features[6]*t1 + features[8]*t2
        combine_confs[2] = features[2]*t + features[7]*t1 + features[9]*t2
        combine_confs[3] = features[3]*t + features[4]*t1 + features[8]*t2

        features.extend(combine_confs)

        if fhsdistance_results:
            print(f"间隙面积: {fhsdistance_results['gap_area_pixels']} 像素")
            print(f"最小距离: {fhsdistance_results['min_distance_pixels']:.1f} 像素")
            features.append(fhsdistance_results['gap_area_pixels'])
            features.append(fhsdistance_results['min_distance_pixels'])
        else:
            features.append(None)
            features.append(None)

        return features


class DataProcessor:
    @staticmethod
    def load_data(txt_path):
        features = []
        labels = []
        file_names = []

        extractor = FeatureExtractor()
        file_list = DataLoader.load_file_list(txt_path)

        for img_path in tqdm(file_list, desc="Extracting features"):
            if not os.path.exists(img_path):
                continue

            feature = extractor.extract_features(img_path)
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

        return np.array(features), np.array(labels), file_names


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
                    objective='multi:softmax',  # 'multi:softmax',
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=42
                ),
                'param_grid': {
                    'n_estimators': list(range(50, 151, 20)),  # list(range(50, 151, 20)),
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],  # [0.01, 0.1],
                    'subsample': [0.8, 1.0]  # 'colsample_bytree':[0.8, 1.0]  # zengjia
                }
            }
        ]


    @staticmethod
    def train_models():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        X_train, y_train, _ = DataProcessor.load_data(Config.TRAIN_TXT)
        selected_indices = FeatureSelector.select_features(X_train, y_train)

        X_train_selected = X_train[:, selected_indices]
        best_models = {}
        search_spaces = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'n_estimators': (100, 500),
            'subsample': (0.6, 1.0),
        }

        for config in ModelTrainer.get_model_configs():
            if config['model'] == 'XGBoost':
                print(f"\n=== Training {config['name']} ===")
                bayes_search = BayesSearchCV(
                    estimator=config['model'],
                    search_spaces=search_spaces,
                    cv=5,
                    n_jobs=-1,
                    verbose=2,
                    scoring='precision_macro'  # 'f1_weighted'
                )
                bayes_search.fit(X_train_selected, y_train)

                model_path = os.path.join(
                    Config.MODEL_SAVE_DIR,
                    f"{config['name']}_best_model.pkl"
                )

                joblib.dump(bayes_search.best_estimator_, model_path)

                print(f"\nBest parameters for {config['name']}:")
                print(bayes_search.best_params_)
                best_models[config['name']] = {
                    'model': bayes_search.best_estimator_,
                    'selected_indices': selected_indices
                }
            else:
                print(f"\n=== Training {config['name']} ===")
                grid_search = GridSearchCV(
                    estimator=config['model'],
                    param_grid=config['param_grid'],
                    cv=5,
                    n_jobs=-1,
                    verbose=2,
                    scoring='precision_macro'  # 'f1_weighted'#'accuracy'
                )
                grid_search.fit(X_train_selected, y_train)

                model_path = os.path.join(
                    Config.MODEL_SAVE_DIR,
                    f"{config['name']}_best_model.pkl"
                )

                # 保存模型到文件
                #model_path = f"{config['name']}_model.pkl"
                save_data = {
                    'config_name': config['name'],
                    'model': grid_search.best_estimator_,
                    'selected_indices': selected_indices
                }
                joblib.dump(save_data, model_path)

                #joblib.dump(grid_search.best_estimator_, model_path)

                print(f"\nBest parameters for {config['name']}:")
                print(grid_search.best_params_)
                best_models[config['name']] = {
                    'model': grid_search.best_estimator_,
                    'selected_indices': selected_indices
                }


class DataLoader:
    @staticmethod
    def load_file_list(txt_path):
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    ModelTrainer.train_models()
