"""ONFH Ficat staging combined training pipeline.

Tri-head YOLOv11 (detection + FH/JS segmentation) -> hip-gap hand-crafted
features -> XGBoost / XGBoost-Weighted (BayesSearchCV-optimised fusion
weights) classifiers.  Optimised version: vectorised NumPy reductions
(np.maximum.at / argsort) replace per-box Python loops; shared
``orig_image`` + ``precomputed_results`` avoid duplicate I/O and YOLO
forward passes; feature vector fixed to 12 dimensions aligned with the
published method description.
"""

import os
import joblib
import numpy as np
from ultralytics import YOLO
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
from tqdm import tqdm
import cv2
import fhsdistance  # Optimised fhsdistance module

# ---------------------------------------------------------------------------
# Configuration (Linux variant, commented out)
# ---------------------------------------------------------------------------

class Config:
    """Windows-local paths used for training."""
    # yolo models' save path
    DET_MODEL_PATH = 'path\\weights\\best.pt'
    SEG_FH_MODEL_PATH = 'path\\weights\\best.pt'
    SEG_JS_MODEL_PATH = 'path\\weights\\best.pt'

    # train data path
    TRAIN_TXT = "path\\train.txt"
    # your combine model's save path
    MODEL_SAVE_DIR = 'path'

    # Detection box confidence filter.  NOTE: only used when picking the
    # Top-2 dominant detections (auxiliary debug info / TOP2 display); the
    # 4-dim D(4) feature vector always takes the np.maximum.at reduction
    # over ALL boxes regardless of this threshold, so changing it does
    # NOT affect model training / prediction.
    DET_CONF_THRESH = 0.1
    # XGBoost gain-based feature importance cutoff.  In the current
    # 12-dim method-aligned pipeline the hard ``union(arange(12))``
    # guard always retains all base columns, so this threshold only has
    # an effect if the extractor is later extended to emit >12 columns.
    FEATURE_IMPORTANCE_THRESHOLD = 0.05


# Class-association matrix (ficat -> det/fh/js indices).
# Row c = ficat stage; columns = [det_cls, fh_cls, js_cls].  Stored as a
# pre-computed ndarray so fusion code can use pure indexing (no dict
# lookups / if-elif branches).
_CLS_ASSOC = np.array(
    [[0, 1, 0],
     [1, 2, 0],
     [2, 3, 1],
     [3, 0, 0]], dtype=np.int32
)


class FeatureExtractor:
    """Runs the three YOLO heads + hip-gap analysis and builds the 12-dim
    static feature vector aligned with the method description:
    D(4) + F(4) + J(2) + A_gap(1) + d_min(1).
    """
    def __init__(self):
        self.det_model = YOLO(Config.DET_MODEL_PATH)
        self.seg_fh_model = YOLO(Config.SEG_FH_MODEL_PATH)
        self.seg_js_model = YOLO(Config.SEG_JS_MODEL_PATH)

    def _get_dominant_detection(self, detect_results):
        """Vectorised Top-2 detection extraction + per-class max confidences.

        Uses ``np.argsort`` for Top-2 (avoids Python sorted/list building)
        and ``np.maximum.at`` for the per-class reduction.
        """
        boxes = detect_results[0].boxes
        if len(boxes) == 0:
            dominant_detections = [(None, 0.0), (None, 0.0)]
            det_confs = np.zeros(4, dtype=np.float64)
            return dominant_detections, det_confs

        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(np.int32)

        # Threshold filter (boolean mask)
        mask = confs >= Config.DET_CONF_THRESH
        valid_clss = clss[mask]
        valid_confs = confs[mask]

        # Top-2 by confidence (descending)
        n_valid = valid_confs.shape[0]
        if n_valid == 0:
            dominant_detections = [(None, 0.0), (None, 0.0)]
        else:
            order = np.argsort(valid_confs)
            top2_idx = order[-2:][::-1]  # [top1_idx, top2_idx]
            dominant_detections = []
            for k in range(2):
                if k < top2_idx.shape[0]:
                    idx = top2_idx[k]
                    dominant_detections.append((int(valid_clss[idx]), float(valid_confs[idx])))
                else:
                    dominant_detections.append((None, 0.0))

        # Per-class max confidences via vectorised np.maximum.at reduce
        det_confs = np.zeros(4, dtype=np.float64)
        in_range_mask = clss < 4
        np.maximum.at(det_confs, clss[in_range_mask], confs[in_range_mask])
        return dominant_detections, det_confs

    def _extract_segmentation_features(self, seg_results, num_classes):
        """Per-class max confidence (vectorised np.maximum.at reduction)."""
        confidences = np.zeros(num_classes, dtype=np.float64)
        if len(seg_results[0].boxes) == 0:
            return confidences
        clss = seg_results[0].boxes.cls.cpu().numpy().astype(np.int32)
        confs = seg_results[0].boxes.conf.cpu().numpy()
        in_range_mask = clss < num_classes
        np.maximum.at(confidences, clss[in_range_mask], confs[in_range_mask])
        return confidences

    def smart_feature_fusion(self, det_confs, fh_confidences, js_confidences, weights=None):
        """Class-association guided weighted fusion (vectorised, unused in
        the current static 12-dim pipeline but kept for interface stability).
        """
        if weights is None:
            # Confidence-normalised adaptive weights (all 4 classes at once)
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
            combine = d * dw + f * fw + j * jw
            return combine.tolist()
        else:
            weights_arr = np.asarray(weights, dtype=np.float64).reshape(4, 3)
            det_idx = _CLS_ASSOC[:, 0]
            fh_idx = _CLS_ASSOC[:, 1]
            js_idx = _CLS_ASSOC[:, 2]
            d = det_confs[det_idx]
            f = fh_confidences[fh_idx]
            j = js_confidences[js_idx]
            combine = d * weights_arr[:, 0] + f * weights_arr[:, 1] + j * weights_arr[:, 2]
            return combine.tolist()

    def extract_features(self, img_path):
        """Return the 12-dim feature vector for one image.

        Optimisations (unchanged logic):
          - Single ``cv2.imread`` shared across all three YOLO heads and
            ``fhsdistance.process_image`` (avoids repeated disk reads).
          - ``YOLO(..., verbose=False)`` suppresses per-image stdout noise.
          - ``precomputed_results`` forwards the JS-seg output into
            fhsdistance so it does not re-run segmentation.
        """
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Detection head
        det_results = self.det_model(image, verbose=False)
        dominant_detections, det_confs = self._get_dominant_detection(det_results)

        # Femoral-head segmentation (4 classes)
        seg_fh_results = self.seg_fh_model(image, verbose=False)
        fh_confidences = self._extract_segmentation_features(seg_fh_results, 4)

        # Joint-space segmentation (2 classes), result reused below
        seg_js_results = self.seg_js_model(image, verbose=False)
        js_confidences = self._extract_segmentation_features(seg_js_results, 2)

        # Reuse the already-computed JS segmentation + loaded image for the
        # hip-gap analysis (skip a second imread + YOLO forward pass inside
        # fhsdistance).
        fhsdistance_results = fhsdistance.process_image(
            self.seg_js_model, img_path, None,
            orig_image=image, precomputed_results=seg_js_results
        )

        # 12-dim vector: D(4) + F(4) + J(2) + A_gap(1) + d_min(1).
        # The 4 BO-weighted s_c fusion dims are appended later by
        # WeightedFusionModel._process_features (not part of the static
        # extractor; they depend on weights learned at train time).
        features = []
        features.extend(det_confs.tolist())      # D(4): stage II/III/IV/nVNFH
        features.extend(fh_confidences.tolist()) # F(4): NFH/IIFH/AFH/fusion
        features.extend(js_confidences.tolist()) # J(2): NJS/AJS

        # Gap area + minimum distance
        if fhsdistance_results:
            features.append(float(fhsdistance_results['gap_area_pixels']))    # A_gap
            features.append(float(fhsdistance_results['min_distance_pixels'])) # d_min
        else:
            features.append(0.0)
            features.append(0.0)

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
        """Gain-based XGBoost feature selection that FORCE-INCLUDES the first
        12 feature columns (0..11).

        The hard-slice logic in WeightedFusionModel._process_features assumes
        columns 0-3=D, 4-7=F, 8-9=J, 10=A_gap, 11=d_min exist.  Dropping any
        of them via a low importance threshold would silently shift the
        semantics, so we always take the union of the threshold-selected
        set with np.arange(12).
        """
        selector = xgb.XGBClassifier(
            n_estimators=100,
            importance_type='gain',
            random_state=42,
            n_jobs=4,
            verbosity=0
        )
        selector.fit(X, y)

        # Normalise importances to sum 1 (so threshold semantics are stable)
        importances = selector.feature_importances_
        importances = importances / (importances.sum() + 1e-8)
        indices = np.argsort(importances)[::-1]

        print("\nFeature ranking:")
        for f in range(X.shape[1]):
            print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]:.4f})")

        selected_indices = np.where(importances > Config.FEATURE_IMPORTANCE_THRESHOLD)[0]
        # Force-keep the full 12-dim method-aligned base vector
        base_indices = np.arange(12)
        selected_indices = np.union1d(selected_indices, base_indices).astype(int)
        print(f"\nSelected features: {selected_indices}")

        return selected_indices


class ModelTrainer:
    @staticmethod
    def get_model_configs():
        return [
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    objective='multi:softmax',
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=42,
                    verbosity=0,
                    n_jobs=4
                )
            },
            {
                'name': 'XGBoost_Weighted',
                'model': WeightedFusionModel(),
                'search_spaces': {
                    'weights': (0.0, 1.0, 'uniform', 12)  # 12 fusion weight parameters
                }
            }
        ]

    @staticmethod
    def calculate_sample_weights(y):
        from sklearn.utils.class_weight import compute_class_weight
        classes_arr = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes_arr, y=y)
        # Vectorised class->weight map lookup (np.vectorize instead of list
        # comprehension; handles non-sequential class ids safely).
        cw_map = dict(zip(classes_arr.tolist(), class_weights.tolist()))
        sample_weights = np.vectorize(cw_map.get, otypes=[np.float64])(y)
        return sample_weights

    @staticmethod
    def train_models():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        X_train, y_train, _ = DataProcessor.load_data(Config.TRAIN_TXT)
        selected_indices = FeatureSelector.select_features(X_train, y_train)

        X_train_selected = X_train[:, selected_indices]
        X_train_processed, imputer, scaler = DataPreprocessor.preprocess_features(X_train_selected)

        sample_weights = ModelTrainer.calculate_sample_weights(y_train)
        print(f"\nSample weights calculated for {len(np.unique(y_train))} classes")

        best_models = {}

        xgb_search_spaces = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'n_estimators': (100, 500),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0, 0.5),
            'reg_alpha': (1e-6, 10, 'log-uniform'),
            'reg_lambda': (0.1, 20, 'log-uniform')
        }

        for config in ModelTrainer.get_model_configs():
            print(f"\n=== Training {config['name']} ===")

            if config['name'] == 'XGBoost':
                # Bayesian search over XGBoost hyperparameters (9-D space).
                # Outer CV parallelism n_jobs=2 + inner XGB n_jobs=4 avoids
                # thread oversubscription on typical 8-16 core desktops.
                bayes_search = BayesSearchCV(
                    estimator=config['model'],
                    search_spaces=xgb_search_spaces,
                    cv=3,
                    n_jobs=2,
                    verbose=1,
                    scoring='f1_weighted',
                    n_iter=100
                )

                bayes_search.fit(X_train_processed, y_train, sample_weight=sample_weights)

                model_path = os.path.join(
                    Config.MODEL_SAVE_DIR,
                    f"{config['name']}_best_model.pkl"
                )

                save_data = {
                    'config_name': config['name'],
                    'model': bayes_search.best_estimator_,
                    'selected_indices': selected_indices,
                    'imputer': imputer,
                    'scaler': scaler,
                    'best_params': bayes_search.best_params_,
                    'best_score': bayes_search.best_score_
                }
                joblib.dump(save_data, model_path)

                print(f"\nBest parameters for {config['name']}:")
                print(bayes_search.best_params_)
                print(f"Best CV Score: {bayes_search.best_score_:.4f}")
                best_models[config['name']] = save_data

            elif config['name'] == 'XGBoost_Weighted':
                # Bayesian search over the 12 class-association fusion
                # weights (4 classes x 3 modalities).  WeightedFusionModel
                # internally embeds a plain XGBClassifier (default params);
                # the outer BO loop tunes only the 12 fusion weights.
                from skopt.space import Real

                weighted_search_spaces = {}
                for i in range(4):     # 4 Ficat classes
                    for j in range(3): # 3 modalities: det / fh / js
                        weighted_search_spaces[f'weights_{i}_{j}'] = Real(0.05, 0.95, name=f'weight_{i}_{j}')

                base_model = WeightedFusionModel()

                bayes_search = BayesSearchCV(
                    estimator=base_model,
                    search_spaces=weighted_search_spaces,
                    cv=3,
                    n_jobs=2,
                    verbose=1,
                    scoring='f1_weighted',
                    n_iter=100
                )

                bayes_search.fit(X_train_processed, y_train, sample_weight=sample_weights)

                model_path = os.path.join(
                    Config.MODEL_SAVE_DIR,
                    f"{config['name']}_best_model.pkl"
                )

                save_data = {
                    'config_name': config['name'],
                    'model': bayes_search.best_estimator_,
                    'selected_indices': selected_indices,
                    'imputer': imputer,
                    'scaler': scaler,
                    'best_params': bayes_search.best_params_,
                    'best_score': bayes_search.best_score_
                }
                joblib.dump(save_data, model_path)

                print(f"\nBest parameters for {config['name']}:")
                print(bayes_search.best_params_)
                print(f"Best CV Score: {bayes_search.best_score_:.4f}")
                best_models[config['name']] = save_data


class WeightedFusionModel(BaseEstimator, ClassifierMixin):
    """Combined classifier: BO-optimised class-association weighted fusion
    (12 weights -> 4 s_c dims) concatenated to the 12-dim static feature
    vector, then fed to an inner XGBClassifier.

    Vectorised batch implementation: per-sample Python loops are replaced
    with matrix-slice + broadcasting over the whole N batch.
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
        # (4,3) weight-matrix cache (row-L1-normalised).  Rebuilt lazily
        # (dirty-bit pattern) whenever any scalar weight changes.
        self._W_cache = None
        self._W_dirty = True

    def _get_W(self):
        """Pack 12 scalar weights into a (4,3) matrix with L1 row normalisation.

        Uses a dirty cache (avoids re-packing / re-normalising inside every
        ``_process_features`` call when weights have not changed).
        """
        if self._W_cache is None or self._W_dirty:
            W = np.empty((4, 3), dtype=np.float64)
            W[0, 0], W[0, 1], W[0, 2] = self.weights_0_0, self.weights_0_1, self.weights_0_2
            W[1, 0], W[1, 1], W[1, 2] = self.weights_1_0, self.weights_1_1, self.weights_1_2
            W[2, 0], W[2, 1], W[2, 2] = self.weights_2_0, self.weights_2_1, self.weights_2_2
            W[3, 0], W[3, 1], W[3, 2] = self.weights_3_0, self.weights_3_1, self.weights_3_2
            # L1 row-normalise (each class's 3 weights sum to 1)
            row_sum = W.sum(axis=1, keepdims=True) + 1e-8
            self._W_cache = W / row_sum
            self._W_dirty = False
        return self._W_cache

    def fit(self, X, y, sample_weight=None):
        self._W_dirty = True  # Weights may have changed since last fit
        X_processed = self._process_features(X)
        if sample_weight is not None:
            self.base_model.fit(X_processed, y, sample_weight=sample_weight)
        else:
            self.base_model.fit(X_processed, y)
        return self

    def _process_features(self, X):
        """Append 4 BO-weighted s_c fusion dims to the static feature vector.

        Input layout (min 12 columns):
            [0-3] D  (detection confs: II/III/IV/nVNFH)
            [4-7] F  (FH-seg confs:      NFH/IIFH/AFH/fusion)
            [8-9] J  (JS-seg confs:      NJS/AJS)
            [10]  A_gap  (hip-gap pixel area)
            [11]  d_min  (minimum hip-gap pixel distance)
        Output shape: (N, D + 4) -> s_c fused scores are HSTACK-ed on the right.
        """
        if X.shape[1] < 12:
            raise ValueError(
                f"_process_features expects at least 12 columns "
                f"(0-3=D, 4-7=F, 8-9=J, 10=A_gap, 11=d_min) but got {X.shape[1]}. "
                f"Ensure FeatureSelector.force-included indices 0..11 (the full "
                f"method-aligned 12-dim base vector)."
            )
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        # Slice D/F/J views (shapes (N,4), (N,4), (N,2))
        DET = X[:, :4]
        FH = X[:, 4:8]
        JS = X[:, 8:10]

        # Class-association index vectors (shape (4,))
        det_idx = _CLS_ASSOC[:, 0]  # [0,1,2,3]
        fh_idx = _CLS_ASSOC[:, 1]   # [1,2,3,0]
        js_idx = _CLS_ASSOC[:, 2]   # [0,0,1,0]

        # Gather aligned confidences per ficat class -> (N, 4) each
        DET_c = DET[:, det_idx]
        FH_c = FH[:, fh_idx]
        JS_c = JS[:, js_idx]

        # W shape (4, 3), each row L1-normalised; broadcast + sum -> (N, 4)
        W = self._get_W()
        Wd = W[:, 0][np.newaxis, :]  # (1, 4)
        Wf = W[:, 1][np.newaxis, :]
        Wj = W[:, 2][np.newaxis, :]
        combine = DET_c * Wd + FH_c * Wf + JS_c * Wj  # (N, 4)

        # HSTACK: original cols + 4 fused s_c -> (N, D + 4)
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
            self._W_dirty = True  # Weight values changed -> cache stale

        if other_params:
            self.base_model.set_params(**other_params)

        return self


class DataPreprocessor:
    @staticmethod
    def preprocess_features(X):
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled, imputer, scaler


class DataLoader:
    @staticmethod
    def load_file_list(txt_path):
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    ModelTrainer.train_models()
