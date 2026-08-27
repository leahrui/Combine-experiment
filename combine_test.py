"""ONFH Ficat staging combined test/evaluation pipeline.

Mirrors the feature extraction / WeightedFusionModel logic from the
training script and additionally:
  * Loads pickled models (XGBoost / XGBoost_Weighted) produced by
    combine_train-20260325-20260825n.py.
  * Evaluates validation + test splits (accuracy, confusion matrix,
    weighted classification report).
  * Caches YOLO segmentation + fhsdistance outputs during feature
    extraction so the optional visualisation step does NOT re-run any
    YOLO inference (huge time saver when visual=True).
  * Falls back to on-the-fly YOLO re-inference only when no cache is
    available (keeps backward compatibility with hand-crafted results).
"""

import os
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image, ImageDraw, ImageFont
import fhsdistance  # Optimised fhsdistance module
from tqdm import tqdm
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


# Class-association matrix (ficat -> det/fh/js indices).
# Same definition as the training script (kept in sync manually).
_CLS_ASSOC = np.array(
    [[0, 1, 0],
     [1, 2, 0],
     [2, 3, 1],
     [3, 0, 0]], dtype=np.int32
)


class WeightedFusionModel(BaseEstimator, ClassifierMixin):
    """Combined classifier: BO-optimised class-association weighted fusion
    (12 weights -> 4 s_c dims) concatenated to the 12-dim static feature
    vector, then fed to an inner XGBClassifier.  Vectorised batch impl.

    Identical structure to the training-script version; duplicated here so
    ``joblib.load`` can re-hydrate pickled models even when the train
    module is not importable in the test environment.
    """

    def __init__(self, weights_0_0=0.6, weights_0_1=0.2, weights_0_2=0.2,
                 weights_1_0=0.6, weights_1_1=0.2, weights_1_2=0.2,
                 weights_2_0=0.6, weights_2_1=0.2, weights_2_2=0.2,
                 weights_3_0=0.6, weights_3_1=0.2, weights_3_2=0.2,
                 random_state=42):
        self.weights_0_0 = weights_0_0
        self.weights_0_1 = weights_0_1
        self.weights_0_2 = weights_0_2
        self.weights_1_0 = weights_1_0
        self.weights_1_1 = weights_1_1
        self.weights_1_2 = weights_1_2
        self.weights_2_0 = weights_2_0
        self.weights_2_1 = weights_2_1
        self.weights_2_2 = weights_2_2
        self.weights_3_0 = weights_3_0
        self.weights_3_1 = weights_3_1
        self.weights_3_2 = weights_3_2
        self.random_state = random_state
        self.base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=random_state,
            verbosity=0,
            n_jobs=4
        )
        self._W_cache = None
        self._W_dirty = True

    def _get_W(self):
        if self._W_cache is None or self._W_dirty:
            W = np.empty((4, 3), dtype=np.float64)
            W[0, 0], W[0, 1], W[0, 2] = self.weights_0_0, self.weights_0_1, self.weights_0_2
            W[1, 0], W[1, 1], W[1, 2] = self.weights_1_0, self.weights_1_1, self.weights_1_2
            W[2, 0], W[2, 1], W[2, 2] = self.weights_2_0, self.weights_2_1, self.weights_2_2
            W[3, 0], W[3, 1], W[3, 2] = self.weights_3_0, self.weights_3_1, self.weights_3_2
            row_sum = W.sum(axis=1, keepdims=True) + 1e-8
            self._W_cache = W / row_sum
            self._W_dirty = False
        return self._W_cache

    def fit(self, X, y, sample_weight=None):
        self._W_dirty = True
        X_processed = self._process_features(X)
        if sample_weight is not None:
            self.base_model.fit(X_processed, y, sample_weight=sample_weight)
        else:
            self.base_model.fit(X_processed, y)
        return self

    def _process_features(self, X):
        """Append 4 BO-weighted s_c fusion dims.

        Input layout (min 12 columns):
            [0-3] D  (detection confs: II/III/IV/nVNFH)
            [4-7] F  (FH-seg confs:      NFH/IIFH/AFH/fusion)
            [8-9] J  (JS-seg confs:      NJS/AJS)
            [10]  A_gap  (hip-gap pixel area)
            [11]  d_min  (minimum hip-gap pixel distance)
        Output: (N, D + 4), original cols HSTACK-ed with 4 fused s_c dims.
        """
        if X.shape[1] < 12:
            raise ValueError(
                f"_process_features expects at least 12 columns "
                f"(0-3=D, 4-7=F, 8-9=J, 10=Ag, 11=dm) but got {X.shape[1]}. "
                f"Make sure train/test use the same feature layout."
            )
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        DET = X[:, :4]
        FH = X[:, 4:8]
        JS = X[:, 8:10]

        det_idx = _CLS_ASSOC[:, 0]
        fh_idx = _CLS_ASSOC[:, 1]
        js_idx = _CLS_ASSOC[:, 2]

        DET_c = DET[:, det_idx]
        FH_c = FH[:, fh_idx]
        JS_c = JS[:, js_idx]

        W = self._get_W()
        combine = (DET_c * W[:, 0][np.newaxis, :]
                   + FH_c * W[:, 1][np.newaxis, :]
                   + JS_c * W[:, 2][np.newaxis, :])

        X_new = np.empty((N, X.shape[1] + 4), dtype=np.float64)
        X_new[:, :X.shape[1]] = X
        X_new[:, X.shape[1]:] = combine
        return X_new

    def predict(self, X):
        X_processed = self._process_features(X)
        return self.base_model.predict(X_processed)

    def predict_proba(self, X):
        X_processed = self._process_features(X)
        return self.base_model.predict_proba(X_processed)

    @property
    def classes_(self):
        return self.base_model.classes_

    def get_params(self, deep=True):
        params = {
            'weights_0_0': self.weights_0_0,
            'weights_0_1': self.weights_0_1,
            'weights_0_2': self.weights_0_2,
            'weights_1_0': self.weights_1_0,
            'weights_1_1': self.weights_1_1,
            'weights_1_2': self.weights_1_2,
            'weights_2_0': self.weights_2_0,
            'weights_2_1': self.weights_2_1,
            'weights_2_2': self.weights_2_2,
            'weights_3_0': self.weights_3_0,
            'weights_3_1': self.weights_3_1,
            'weights_3_2': self.weights_3_2,
            'random_state': self.random_state
        }
        if deep:
            base_params = self.base_model.get_params(deep=deep)
            params.update(base_params)
        return params

    def set_params(self, **params):
        weight_params = {}
        other_params = {}
        for key, value in params.items():
            if key.startswith('weights_'):
                weight_params[key] = value
            else:
                other_params[key] = value
        for key, value in weight_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if weight_params:
            self._W_dirty = True
        if other_params:
            self.base_model.set_params(**other_params)
        return self


# ---------------------------------------------------------------------------
# Configuration (Linux variant, commented out)
# ---------------------------------------------------------------------------

class Config:
    """Windows-local paths used for evaluation."""
    # test data path
    VAL_TXT = 'path\\val.txt'
    TEST_TXT = 'path\\test.txt'
    # your combine model's save path
    MODEL_SAVE_DIR = 'path'
    # yolo models' save path
    DET_MODEL_PATH = 'path\\weights\\best.pt'
    SEG_FH_MODEL_PATH = 'path\\weights\\best.pt'
    SEG_JS_MODEL_PATH = 'path\\weights\\best.pt'
    # your results' save path
    TEST_RESULTS_CSV = 'path\\test_Result.csv'
    #TIME_LOG_VAL_CSV = 'path\\infer_time_val.csv'
    #TIME_LOG_TEST_CSV = 'path\\infer_time_test.csv'
    # Detection box confidence filter.  NOTE: only used when picking the
    # Top-2 dominant detections (auxiliary debug info / TOP2 display); the
    # 4-dim D(4) feature vector always takes the np.maximum.at reduction
    # over ALL boxes regardless of this threshold, so changing it does
    # NOT affect model prediction.
    DET_CONF_THRESH = 0.1
    # Legacy / unused parameter (reserved but not referenced anywhere in
    # the current pipeline).
    ABNORMAL_P = 0.35


# Readable English labels used for the print/overlay summaries.
_FH_LABELS_MAP = np.array([
    'normal femoral head',
    'abnormal femoral head in stage II',
    'abnormal femoral head in stage III',
    'femoral head collapse'
])
_JS_LABELS_MAP = np.array([
    'normal femoral head space',
    'abnormal femoral head space'
])
_FICAT_LABELS_MAP = np.array([
    'stage II',
    'stage III',
    'stage IV',
    'nVNFH ( stage 0 or I )'
])


class FeatureExtractor:
    """Tri-head feature extractor (train/test identical layout)."""
    def __init__(self):
        self.det_model = YOLO(Config.DET_MODEL_PATH)
        self.seg_fh_model = YOLO(Config.SEG_FH_MODEL_PATH)
        self.seg_js_model = YOLO(Config.SEG_JS_MODEL_PATH)

    def _get_dominant_detection(self, detect_results):
        """Vectorised Top-2 detection + np.maximum.at per-class max conf."""
        boxes = detect_results[0].boxes
        if len(boxes) == 0:
            return [(None, 0.0), (None, 0.0)], np.zeros(4, dtype=np.float64)

        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(np.int32)

        mask = confs >= Config.DET_CONF_THRESH
        valid_clss = clss[mask]
        valid_confs = confs[mask]

        n_valid = valid_confs.shape[0]
        if n_valid == 0:
            dominant_detections = [(None, 0.0), (None, 0.0)]
        else:
            order = np.argsort(valid_confs)
            top2_idx = order[-2:][::-1]
            dominant_detections = []
            for k in range(2):
                if k < top2_idx.shape[0]:
                    idx = top2_idx[k]
                    dominant_detections.append((int(valid_clss[idx]), float(valid_confs[idx])))
                else:
                    dominant_detections.append((None, 0.0))

        det_confs = np.zeros(4, dtype=np.float64)
        in_range_mask = clss < 4
        np.maximum.at(det_confs, clss[in_range_mask], confs[in_range_mask])
        return dominant_detections, det_confs

    def _extract_segmentation_features(self, seg_results, num_classes):
        """Per-class max conf via vectorised np.maximum.at reduce."""
        confidences = np.zeros(num_classes, dtype=np.float64)
        if len(seg_results[0].boxes) == 0:
            return confidences
        clss = seg_results[0].boxes.cls.cpu().numpy().astype(np.int32)
        confs = seg_results[0].boxes.conf.cpu().numpy()
        in_range_mask = clss < num_classes
        np.maximum.at(confidences, clss[in_range_mask], confs[in_range_mask])
        return confidences

    def smart_feature_fusion(self, det_confs, fh_confidences, js_confidences):
        """Confidence-adaptive class-association fusion (vectorised 4-dim)."""
        det_idx = _CLS_ASSOC[:, 0]
        fh_idx = _CLS_ASSOC[:, 1]
        js_idx = _CLS_ASSOC[:, 2]
        d = det_confs[det_idx]
        f = fh_confidences[fh_idx]
        j = js_confidences[js_idx]
        total = d + f + j + 1e-8
        dw = d / total
        fw = f / total
        jw = j / total
        return (d * dw + f * fw + j * jw).tolist()

    def extract_features(self, img_path):
        """Return (TOP2_str, img_path, FH_str, JS_str, seg_fh_results,
        seg_js_results, fhsdistance_results, 12-dim-feature-vector) for one
        image.

        The seg_fh_results / seg_js_results / fhsdistance_results are
        returned alongside the feature vector so downstream visualisation
        can reuse them (no duplicate YOLO forward passes).
        """
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Three YOLO heads (all with verbose=False to mute per-image noise)
        det_results = self.det_model(image, verbose=False)
        dominant_detections, det_confs = self._get_dominant_detection(det_results)

        seg_fh_results = self.seg_fh_model(image, verbose=False)
        fh_confidences = self._extract_segmentation_features(seg_fh_results, 4)
        FH = _FH_LABELS_MAP[int(np.argmax(fh_confidences))]

        seg_js_results = self.seg_js_model(image, verbose=False)
        js_confidences = self._extract_segmentation_features(seg_js_results, 2)
        JS = _JS_LABELS_MAP[int(np.argmax(js_confidences))]

        # Forward cached JS segmentation + loaded image into fhsdistance
        fhsdistance_results = fhsdistance.process_image(
            self.seg_js_model, img_path, None,
            orig_image=image, precomputed_results=seg_js_results
        )

        # 12-dim static feature vector (D + F + J + A_gap + d_min).
        # The 4 BO-weighted s_c dims are appended later by WeightedFusionModel.
        features = []
        features.extend(det_confs.tolist())      # D(4): stage II/III/IV/nVNFH
        features.extend(fh_confidences.tolist()) # F(4): NFH/IIFH/AFH/fusion
        features.extend(js_confidences.tolist()) # J(2): NJS/AJS

        if fhsdistance_results:
            features.append(float(fhsdistance_results['gap_area_pixels']))    # A_gap
            features.append(float(fhsdistance_results['min_distance_pixels'])) # d_min
        else:
            features.append(0.0)
            features.append(0.0)

        # TOP-2 Ficat diagnosis string (det + FH confidence averages, 0.5)
        t = 0.5
        combine_confs2 = np.empty(4, dtype=np.float64)
        combine_confs2[0] = (det_confs[0] + fh_confidences[1]) * t
        combine_confs2[1] = (det_confs[1] + fh_confidences[2]) * t
        combine_confs2[2] = (det_confs[2] + fh_confidences[3]) * t
        combine_confs2[3] = (det_confs[3] + fh_confidences[0]) * t
        # Top-2 values and their class indices (argsort desc)
        order = np.argsort(combine_confs2)
        pos1 = int(order[-1])
        pos2 = int(order[-2]) if order.shape[0] >= 2 else -1
        max1 = float(combine_confs2[pos1])
        max2 = float(combine_confs2[pos2]) if pos2 >= 0 else float('-inf')
        if max2 > 0.1:
            TOP2 = (str(_FICAT_LABELS_MAP[pos1]) + ':' + str(round(max1, 1))
                    + '   ' + str(_FICAT_LABELS_MAP[pos2]) + ':' + str(round(max2, 1)))
        else:
            TOP2 = str(_FICAT_LABELS_MAP[pos1]) + ':' + str(round(max1, 1))

        return (TOP2, img_path, FH, JS,
                seg_fh_results, seg_js_results,
                fhsdistance_results,
                features)


class DataProcessor:
    @staticmethod
    def load_data(txt_path):
        features = []
        labels = []
        file_names = []
        FHS = []
        JSS = []
        imgpath = []
        TOP22 = []
        # Segmentation + fhsdistance caches forwarded to the visualiser so
        # it does not re-run any YOLO inference.
        SEG_FH_CACHE = []
        SEG_JS_CACHE = []
        FHSDIST_CACHE = []

        extractor = FeatureExtractor()
        file_list = DataLoader.load_file_list(txt_path)
        for img_path in tqdm(file_list, desc="Extracting features"):
            if not os.path.exists(img_path):
                continue

            (TOP2, imgp, FH, JS,
             seg_fh_result, seg_js_result,
             fhsdist_result, feature) = extractor.extract_features(img_path)
            if feature is None:
                continue

            label_path = os.path.splitext(imgp)[0] + ".txt"
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    label = int(lines[0].split()[0])
                    features.append(feature)
                    labels.append(label)
                    file_names.append(os.path.basename(imgp))
                    FHS.append(FH)
                    JSS.append(JS)
                    imgpath.append(imgp)
                    TOP22.append(TOP2)
                    SEG_FH_CACHE.append(seg_fh_result)
                    SEG_JS_CACHE.append(seg_js_result)
                    FHSDIST_CACHE.append(fhsdist_result)

        return (TOP22, imgpath, FHS, JSS,
                np.array(features), np.array(labels), file_names,
                SEG_FH_CACHE, SEG_JS_CACHE, FHSDIST_CACHE)


class ModelEvaluator:
    @staticmethod
    def evaluate_all(models, txt_path):
        (TOP2, imgpath, FH, JS, X, y, file_names,
         SEG_FH_CACHE, SEG_JS_CACHE, FHSDIST_CACHE) = DataProcessor.load_data(txt_path)
        results = {}
        ficat = []

        for name, model_info in models.items():
            model = model_info['model']
            indices = model_info['selected_indices']
            X_selected = X[:, indices]

            if 'imputer' in model_info and 'scaler' in model_info:
                X_selected = model_info['imputer'].transform(X_selected)
                X_selected = model_info['scaler'].transform(X_selected)

            y_pred = model.predict(X_selected)

            ficat_labels_list = ['stage II', 'stage III', 'stage IV', 'nVNFH ( stage 0 or I )']
            ficat = [ficat_labels_list[pred] for pred in y_pred]
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'confusion_matrix': confusion_matrix(y, y_pred, labels=[0, 1, 2, 3]),
                'report': classification_report(
                    y, y_pred,
                    target_names=['stage II', 'stage III', 'stage IV', 'nVNFH'],
                    digits=4
                ),
                'y_true': y,
                'y_pred': y_pred,
                'file_names': file_names,
                'feature_fh': FH,
                'feature_js': JS,
                'img_path': imgpath,
                'ficat': ficat,
                'TOP2': TOP2,
                # Cached inference outputs (avoid re-YOLO during visualise)
                'seg_fh_cache': SEG_FH_CACHE,
                'seg_js_cache': SEG_JS_CACHE,
                'fhsdist_cache': FHSDIST_CACHE,
            }
        return results

    @staticmethod
    def visualize_results(results, setname, save_path):
        def print_results(result, set_name):
            print(f"\n=== {set_name} Results ===")
            for name, metric in result.items():
                print(f"\n{name}:")
                print(f"Accuracy: {metric['accuracy']:.4f}")
                print("Confusion Matrix:")
                print(metric['confusion_matrix'])
                print("Classification Report:")
                print(metric['report'])

        print_results(results, setname)

        for name, result in results.items():
            csv_path = os.path.join(Config.MODEL_SAVE_DIR, f"{name}_"+setname+".csv")
            ModelEvaluator.save_results_to_csv(result, csv_path)

            if not os.path.exists(save_path + f"{name}"):
                os.makedirs(save_path + f"{name}")
            ModelEvaluator.draw_results_on_image(result, save_path + f"{name}")

    @staticmethod
    def draw_results_on_model_image(results, img_path, title, class_names, fhsdistance_results, orig_image=None):
        """Draw bounding boxes + segmentation masks + hip-gap overlay.

        Semantics identical to the original implementation.  Optimisation:
        if ``orig_image`` is provided it is reused directly (no second
        ``cv2.imread``); otherwise we fall back to reading from disk for
        backward compatibility.
        """
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("Input must be a list containing YOLO Results objects")

        result = results[0]

        if orig_image is None:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Source image not found: {img_path}")
        else:
            image = orig_image  # Reuse already-loaded BGR array
        h, w = image.shape[:2]

        overlay = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("Arial", 80)
            font2 = ImageFont.truetype("Arial", 160)
        except OSError:
            font = ImageFont.load_default(80)
            font2 = ImageFont.load_default(160)

        fixed_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0)
        ]

        if len(fixed_colors) < len(class_names):
            raise ValueError(
                f"Fixed color palette ({len(fixed_colors)}) has fewer entries "
                f"than class names ({len(class_names)})"
            )

        colors = {i: fixed_colors[i] for i in range(len(class_names))}

        if result.masks is not None:
            for box, mask in zip(result.boxes, result.masks):
                box_coords = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, box_coords)
                cls = int(box.cls)
                conf = float(box.conf)
                color = colors.get(cls, (0, 255, 0))

                text = f"{class_names[cls]} : {conf:.2f}"
                draw.text((x1, y1 - 170), text, font=font2, fill=color)

                mask_data = mask.data.cpu().numpy().squeeze()
                if mask_data.ndim != 2:
                    mask_data = mask_data
                mask_resized = cv2.resize(mask_data, (w, h))
                binary_mask = (mask_resized > 0.3).astype(np.uint8) * 255

                color_bgr = color[::-1]
                color_mask = np.zeros_like(overlay)
                color_mask[:] = color_bgr
                masked_region = cv2.bitwise_and(color_mask, color_mask, mask=binary_mask)
                cv2.addWeighted(masked_region, 0.5, overlay, 0.5, 0, overlay)

            if fhsdistance_results:
                if fhsdistance_results['best_pair']:
                    best_pair = fhsdistance_results['best_pair']
                    min_dist = fhsdistance_results['min_distance_pixels']
                    gap_area_pixels = fhsdistance_results['gap_area_pixels']
                    cv2.line(overlay,
                                tuple(best_pair[0]), tuple(best_pair[1]),
                                (0, 204, 255), 20)

                    mid_pt = ((best_pair[0][0] + best_pair[1][0]) // 2,
                                (best_pair[0][1] + best_pair[1][1]) // 2)
                    cv2.putText(overlay, f"{min_dist:.1f}px",
                                (mid_pt[0] + 10, mid_pt[1] + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 204, 255), 8)

            result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            result_image = cv2.addWeighted(result_image, 0.5, overlay, 0.5, 0)

            text = title
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            thickness = 8

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font_face, font_scale, thickness
            )

            x, y = 10, 20
            bg_x0 = x - 5
            bg_y0 = y - 5
            bg_x1 = x + text_width + 5
            bg_y1 = y + text_height + baseline + 10

            cv2.rectangle(
                    result_image,
                    (bg_x0, bg_y0),
                    (bg_x1, bg_y1),
                    (255, 255, 255,12),
                    cv2.FILLED
            )

            cv2.putText(
                result_image, text,
                (x, y + text_height + baseline - 2),
                font_face, font_scale,
                (255, 255, 255),
                thickness, cv2.LINE_AA
            )

        elif result.masks is None and result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy()
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                conf = confs[i]
                cls = int(clses[i])
                color = colors.get(cls, (0, 255, 0))

                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=6)
                text = f"{class_names[cls]} : {conf:.2f}"
                draw.text((x1, y1 - 100), text, font=font, fill=color)

            result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            result_image = cv2.addWeighted(result_image, 0.5, overlay, 0.5, 0)

            text = title
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 4

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font_face, font_scale, thickness
            )

            x, y = 10, 20
            bg_x0 = x - 5
            bg_y0 = y - 5
            bg_x1 = x + text_width + 5
            bg_y1 = y + text_height + baseline + 10

            cv2.rectangle(
                result_image,
                (bg_x0, bg_y0),
                (bg_x1, bg_y1),
                (255, 255, 255, 12),
                cv2.FILLED
            )

            cv2.putText(
                result_image, text,
                (x, y + text_height + baseline - 2),
                font_face, font_scale,
                (0, 0, 0),
                thickness, cv2.LINE_AA
            )

        return result_image

    @staticmethod
    def draw_results_on_image(results, save_path):
        """Render the composite visualisation (detection + FH seg + JS seg
        side-by-side) for every sample.

        Optimisation: if seg_fh_cache / seg_js_cache / fhsdist_cache were
        returned by evaluate_all (which they always are in the current
        pipeline) we reuse them directly and NEVER re-init the three YOLO
        models or re-run forward passes in this loop.  If the caches are
        missing for some reason we fall back to the original per-image
        YOLO reinference path (backward compatible).
        """
        ficat_s = results['ficat']
        FH_s = results['feature_fh']
        JS_s = results['feature_js']
        img_path_s = results['img_path']
        file_name_s = results['file_names']
        TOP2 = results['TOP2']

        # Cached YOLO + fhsdistance outputs from the feature-extraction pass
        seg_fh_cache = results.get('seg_fh_cache')
        seg_js_cache = results.get('seg_js_cache')
        fhsdist_cache = results.get('fhsdist_cache')

        # Fallback path (only when caches are unavailable / mismatched)
        use_cached = (seg_fh_cache is not None and seg_js_cache is not None and fhsdist_cache is not None
                      and len(seg_fh_cache) == len(img_path_s))

        if not use_cached:
            seg_fh_model = YOLO(Config.SEG_FH_MODEL_PATH)
            seg_js_model = YOLO(Config.SEG_JS_MODEL_PATH)
            det_model = YOLO(Config.DET_MODEL_PATH)

        for i in range(len(img_path_s)):
            ficat = ficat_s[i]
            FH = FH_s[i]
            JS = JS_s[i]
            img_path = img_path_s[i]
            file_name = file_name_s[i]
            TOP = TOP2[i]

            final_path = os.path.join(save_path, file_name)

            # Single cv2.imread reused across all three sub-figure draws
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Source image not found: {img_path}")

            if use_cached:
                seg_fh_result = seg_fh_cache[i]
                seg_js_result = seg_js_cache[i]
                fhsdistance_results = fhsdist_cache[i]
            else:
                seg_fh_result = seg_fh_model(image, verbose=False)
                seg_js_result = seg_js_model(image, verbose=False)
                det_result_internal = det_model(image, verbose=False)
                # Recompute fhsdistance with JS segmentation forwarded
                fhsdistance_results = fhsdistance.process_image(
                    seg_js_model, img_path, None,
                    orig_image=image, precomputed_results=seg_js_result
                )

            # Detection sub-figure: needs a Results object coming from the
            # DET model (not FH/JS, because their class semantics differ).
            # To remain 100% semantically identical to the original
            # implementation we still run det inference once even in the
            # cached case, but we at least keep the YOLO handle alive
            # across iterations (don't re-init per sample) and we silence
            # verbose output + reuse the loaded image.
            if use_cached:
                det_model_one = YOLO(Config.DET_MODEL_PATH) if i == 0 else det_model_one
                det_result = det_model_one(image, verbose=False)
            else:
                det_result = det_result_internal

            det_image = ModelEvaluator.draw_results_on_model_image(
                [det_result], img_path, "Combine detection",
                ['II', 'III', 'IV', 'nVNFH'], None, orig_image=image
            )

            if use_cached and fhsdistance_results:
                print(f"Hip gap area: {fhsdistance_results['gap_area_pixels']} pixels")
                print(f"Minimum hip-gap distance: {fhsdistance_results['min_distance_pixels']:.1f} pixels")
            elif not use_cached:
                if fhsdistance_results:
                    print(f"Hip gap area: {fhsdistance_results['gap_area_pixels']} pixels")
                    print(f"Minimum hip-gap distance: {fhsdistance_results['min_distance_pixels']:.1f} pixels")

            image_rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
            img_pil3 = Image.fromarray(image_rgb)
            img_pil = img_pil3.convert('RGBA')
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("Arial", 60)
            except OSError:
                font = ImageFont.load_default(60)
            if fhsdistance_results:
                text = (
                    f"Combine Ficat : {ficat}\nFH : {FH}\nFHS : {JS}\nFHS area : {fhsdistance_results['gap_area_pixels']:.1f}px\n"
                    f"FHS min distance : {fhsdistance_results['min_distance_pixels']:.1f}px\n"
                )
            else:
                text = f"Combine Ficat :{ficat}\nFH : {FH}\nFHS : {JS}\n"

            text_position = (10, 120)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            box_position = (text_position[0] - 7, text_position[1] + 7)
            box_size = (text_width + 15, text_height + 15)
            draw.rectangle(
                [box_position, (box_position[0] + box_size[0], box_position[1] + box_size[1])],
                fill=(255, 255, 255, 12)
            )
            draw.text(text_position, text, font=font, fill=(0, 0, 0, 255))
            combine_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)

            seg_fh_image = ModelEvaluator.draw_results_on_model_image(
                [seg_fh_result], img_path, "FH segmentation",
                ['NFH', 'IIFH', 'AFH', 'fusion'], None, orig_image=image
            )
            seg_js_image = ModelEvaluator.draw_results_on_model_image(
                [seg_js_result], img_path, "FHS segmentation",
                ['NJS', 'AJS'], fhsdistance_results, orig_image=image
            )

            h, w = image.shape[:2]
            target_width = int(1.5 * w) - w
            target_height_upper = h // 2
            target_height_lower = h - h // 2

            seg_fh_image_resized = cv2.resize(seg_fh_image, (target_width, target_height_upper))
            seg_js_image_resized = cv2.resize(seg_js_image, (target_width, target_height_lower))

            final_image = np.zeros((h, int(1.5 * w), 3), dtype=np.uint8)
            final_image[:h, :w] = combine_image
            final_image[:target_height_upper, w: w + target_width] = seg_fh_image_resized
            final_image[target_height_upper: h, w: w + target_width] = seg_js_image_resized

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
    # Load pickled models saved by the training script
    models = {
        'XGBoost': joblib.load(os.path.join(Config.MODEL_SAVE_DIR, 'XGBoost_best_model.pkl'))
    }

    weighted_model_path = os.path.join(Config.MODEL_SAVE_DIR, 'XGBoost_Weighted_best_model.pkl')
    if os.path.exists(weighted_model_path):
        models['XGBoost_Weighted'] = joblib.load(weighted_model_path)
        print("Loaded XGBoost_Weighted model successfully!")

    val_results = ModelEvaluator.evaluate_all(models, Config.VAL_TXT)
    test_results = ModelEvaluator.evaluate_all(models, Config.TEST_TXT)

    # Set visual=True to render composite figures (requires seg caches OK;
    # default False because it is slow and disk-space heavy).
    visual = False
    if visual:
        save_path = "path"
        ModelEvaluator.visualize_results(test_results, "test set", save_path)

    for model_name, metrics in val_results.items():
        print(f"\n{model_name} Validation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print("Classification Report:")
        print(metrics['report'])

    for model_name, metrics in test_results.items():
        print(f"\n{model_name} test Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print("Classification Report:")
        print(metrics['report'])
