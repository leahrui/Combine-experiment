# Combine-experiment
# Combined Model: An X-ray Image Ficat Staging Model for ONFH (Osteonecrosis of the Femoral Head) Based on the Fusion of YOLOv11 and Machine Learning

## Project Introduction
This project leverages YOLOv11 object detection and segmentation models, integrated with machine learning algorithms such as Decision Tree, Random Forest, and XGBoost, to achieve multi-stage classification of ONFH (Stage II, Stage III, Stage IV, nVNFH). Core functionalities include hip joint object detection, femoral head segmentation, joint space segmentation, calculation of joint space distance and area, and disease staging classification via multi-feature fusion. It is suitable for medical image-assisted diagnosis scenarios.

## Core Features
- **Multi-model Collaboration**: YOLOv11 handles detection and segmentation, while machine learning models manage feature fusion and classification.
- **Precise Quantitative Analysis**: Supports calculation of key indicators such as minimum joint space distance and joint space area.
- **Visual Output**: Generates comprehensive result images containing detection boxes, segmentation masks, classification results, and quantitative indicators.
- **Full Workflow Support**: Covers the entire pipeline from data loading, feature extraction, model training to evaluation and testing.

## Environment Dependencies
- Python 3.8+
- Core Libraries:
  - ultralytics (for YOLOv11)
  - numpy, pandas
  - scikit-learn, xgboost, scipy
  - opencv-python (cv2)
  - Pillow (PIL)
  - joblib, tqdm, skopt

## Installation Steps
1. Clone the repository
```bash
git clone https://github.com/leahrui/Combine-experiment.git
cd Combine-experiment.
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
(Note: You can create the `requirements.txt` file yourself, including the core library names listed above.)

3. Download pre-trained models
- Download the YOLOv11 detection model (det_model), femoral head segmentation model (seg_fh_model), and joint space segmentation model (seg_js_model).
- Place the models in the paths specified in the configuration file.

## Dataset Preparation
### Data Structure
```
dataset/
    ├── dataset20250315/  # Image folder (contains left/right hip joint images marked with L/R,and the label filesfor each image 
    │   ├── 2019-54-III-96L.jpg
    │   ├── 2019-54-III-96R.jpg
    │   ├── 2019-54-III-96R.jpg
    │   └── ...
    ├── train.txt  # List of image paths for the training set
    ├── val.txt  # List of image paths for the validation set
    └── test.txt  # List of image paths for the test set
```

### Label Description
| Label ID | Disease Stage           |
|----------|-------------------------|
| 0        | Stage II                |
| 1        | Stage III               |
| 2        | Stage IV                |
| 3        | nVNFH (Stage 0 or I)    |

## Usage Instructions
### 1. Model Training (combine_train.py)
- Function: Extract image features, select important features, and train machine learning classification models.
- Steps:
  1. Modify the path configurations in the `Config` class (model paths, training set path, model save path).
  2. Run the training script:
  ```bash
  python combine_train.py
  ```
- Output: Trained models (DecisionTree, RandomForest, XGBoost) are saved to the `MODEL_SAVE_DIR` directory.

### 2. Model Testing and Evaluation (combine_test.py)
- Function: Load trained models, perform predictions on the test set, and output accuracy, confusion matrix, classification report, and visual results.
- Steps:
  1. Modify the test set path, model save path, and result save path in the `Config` class.
  2. Run the test script:
  ```bash
  python combine_test.py
  ```
- Output:
  - Classification metrics (accuracy, confusion matrix, classification report) are printed to the console.
  - Test result CSV file (saved to the `TEST_RESULTS_CSV` path).
  - Visual result images (containing original images, detection boxes, segmentation masks, classification results, and quantitative indicators) are saved to the specified directory.

### 3. Joint Space Calculation (fhsdistance.py)
- Function: Independently calculate joint space area and minimum distance.
- Execution Method:
  1. Modify the model path, image folder path, and result save path in the `main` function.
  2. Run the script:
  ```bash
  python fhsdistance.py
  ```

## Project Structure
```
hip-joint-detection/
├── combine_train.py       # Model training script (feature extraction, feature selection, model training)
├── combine_test.py        # Model testing and evaluation script (prediction, metric calculation, visualization)
├── fhsdistance.py         # Joint space quantitative calculation script (area, minimum distance)
├── test.txt        # List of test set image paths
├── requirements.txt       # List of dependent libraries
└── README.md              # Project documentation
```

## Key Configuration Description (Config Class)
| Configuration Item    | Function                          | Modification Notes                  |
|-----------------------|-----------------------------------|-------------------------------------|
| VAL_TXT/TEST_TXT      | Path files for validation/test set | Replace with custom path files      |
| MODEL_SAVE_DIR        | Model save directory              | Set a custom model storage path     |
| DET_MODEL_PATH, etc.  | Paths for YOLO pre-trained models  | Replace with actual pre-trained model paths |
| DET_CONF_THRESH       | Detection confidence threshold     | Adjust based on data (default: 0.1) |
| ABNORMAL_P            | Abnormality determination threshold | Threshold for adjusting classification results (default: 0.35) |

## Notes
1. **Path Configuration**: All paths in the scripts are absolute paths. Before use, modify the `Config` class or relevant path variables according to your local environment.
2. **Model Dependencies**: Ensure that YOLOv11 pre-trained models (detection and segmentation) have been correctly downloaded and placed in the specified paths.
3. **Dataset Format**: Each image must correspond to a label file (with the same name and `.txt` extension), and the first column of the label file must be the label ID.
4. **Visualization**: When running the test script, result image windows will pop up automatically. Press any key to close the windows and continue execution.
5. **Dependency Versions**: It is recommended to use the specified versions of dependent libraries to avoid version compatibility issues (especially for ultralytics and xgboost).

## License
This project is for academic research use only and not for commercial use.

## Contact Information
For questions or suggestions, please contact: nierui@tmmu.edu.cn
