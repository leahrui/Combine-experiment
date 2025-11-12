import os
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image, ImageDraw, ImageFont
import fhsdistance
from tqdm import tqdm

class Config:
    # data path
    VAL_TXT = '/home/dyy/sda/dyy/objectdetectdataset/33dataset/newtest1112.txt'  # newtest0410.txt' #'/home/dyy/sda/dyy/objectdetectdataset/33dataset/dataset20250821/xn-dptest.txt'
    TEST_TXT = '/home/dyy/sda/dyy/objectdetectdataset/33dataset/newtest1112.txt'  # newtest0410.txt"

    # your models' save path
    MODEL_SAVE_DIR = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/combine1111/'
    # yolo models' save path
    DET_MODEL_PATH = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/train/yolov11-xndp/weights/best.pt'
    SEG_FH_MODEL_PATH = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/train/segFH-yolo11-xndp2/weights/best.pt'
    SEG_JS_MODEL_PATH = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/train/segJS-yolo11-xndp/weights/best.pt'
    # your results' save path
    TEST_RESULTS_CSV = '/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/combine-yolo11-1111/test_Result.csv'

    # 检测参数
    DET_CONF_THRESH = 0.1
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
        fh_labels = {0: 'normal femoral head', 1: 'abnormal femoral head in stage II',
                     2: 'abnormal femoral head in stage III',
                     3: 'femoral head collapse'}
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

        fhsdistance_results = fhsdistance.process_image(self.seg_js_model, img_path, None)  # 计算间隙最短距离及面积---20251010

        # combine features
        features = []
        features.extend(det_confs)
        features.extend(fh_confidences)
        features.extend(js_confidences)
        for cls_id, conf in dominant_detections:
            features.append(cls_id)
            features.append(conf)
        # print("features:", features)
        combine_confs = [0] * 4
        combine_confs2 = [0] * 4
        t = 0.6
        t1 = 0.2
        t2 = 0.2
        combine_confs[0] = features[0] * t + features[5] * t1 + features[8] * t2
        combine_confs[1] = features[1] * t + features[6] * t1 + features[8] * t2
        combine_confs[2] = features[2] * t + features[7] * t1 + features[9] * t2
        combine_confs[3] = features[3] * t + features[4] * t1 + features[8] * t2

        features.extend(combine_confs)

        if fhsdistance_results:
            print(f"间隙面积: {fhsdistance_results['gap_area_pixels']} 像素")
            print(f"最小距离: {fhsdistance_results['min_distance_pixels']:.1f} 像素")
            features.append(fhsdistance_results['gap_area_pixels'])
            features.append(fhsdistance_results['min_distance_pixels'])
        else:
            features.append(None)
            features.append(None)

        t3, t4, t5, t6 = 0.5, 0.5, 0.5, 0.5
        combine_confs2[0] = (features[0] + features[5]) * t3
        combine_confs2[1] = (features[1] + features[6]) * t4
        combine_confs2[2] = (features[2] + features[7]) * t5
        combine_confs2[3] = (features[3] + features[4]) * t6
        max1, max2 = float('-inf'), float('-inf')
        pos1, pos2 = -1, -1
        for index, num in enumerate(combine_confs2):
            if num > max1:
                max2, max1 = max1, num
                pos2, pos1 = pos1, index
            elif num > max2:
                max2 = num
                pos2 = index
        ficat_labels = {0: 'stage II', 1: 'stage III', 2: 'stage IV', 3: 'nVNFH ( stage 0 or I )'}
        if max2 > 0.1:
            TOP2 = str(ficat_labels[pos1]) + ':' + str(round(max1, 1)) + '   ' + str(ficat_labels[pos2]) + ":" + str(
                round(max2, 1))
        else:
            TOP2 = str(ficat_labels[pos1]) + ':' + str(round(max1, 1))
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

            for i in range(len(y_pred)):
                ficat_labels = {0:'stage II', 1:'stage III', 2:'stage IV', 3:'nVNFH ( stage 0 or I )'}
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
                'report': classification_report(y, y_pred, target_names=['II期', 'III期', 'IV期', 'nVNFH'], digits=4),
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

            #save_path = save_path + f"{name}"
            if not os.path.exists(save_path + f"{name}"):
                os.makedirs(save_path + f"{name}")
            ModelEvaluator.draw_results_on_image(result, save_path + f"{name}")

    @staticmethod
    def draw_results_on_model_image(results, img_path, title, class_names, fhsdistance_results):
        # 确保输入是 Results 对象列表
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("输入必须是包含 Results 对象的列表")

        # 提取第一个 Results 对象
        result = results[0]  # 关键修正：使用索引提取

        # 验证必要属性存在
        # if not hasattr(result, "boxes") or not hasattr(result, "masks"):
        #    raise ValueError("模型结果缺少 boxes 或 masks 属性")

        # 检查是否有检测结果
        # if len(result.boxes) == 0 and len(result.masks) == 0:
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
            font = ImageFont.truetype("Arial", 80)
            font2 = ImageFont.truetype("Arial", 160)
        except OSError:
            # font = ImageFont.truetype("DejaVuSans.ttf", 60)
            # 使用默认字体
            font = ImageFont.load_default(80)
            font2 = ImageFont.load_default(160)

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
                # draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                text = f"{class_names[cls]} : {conf:.2f}"
                draw.text((x1, y1 - 170), text, font=font2, fill=color)

                # 处理分割掩码（关键修正）
                mask_data = mask.data.cpu().numpy().squeeze()
                if mask_data.ndim != 2:
                    mask_data = mask_data  # 处理冗余维度
                mask_resized = cv2.resize(mask_data, (w, h))  # (width, height)
                binary_mask = (mask_resized > 0.3).astype(np.uint8) * 255  # 调整阈值

                # 创建颜色叠加层（BGR格式）
                color_bgr = color[::-1]  # RGB -> BGR
                color_mask = np.zeros_like(overlay)
                color_mask[:] = color_bgr
                masked_region = cv2.bitwise_and(color_mask, color_mask, mask=binary_mask)

                # 累加叠加效果（保留之前的结果）
                cv2.addWeighted(masked_region, 0.5, overlay, 0.5, 0, overlay)

            if fhsdistance_results:
                # print(f"间隙面积: {fhsdistance_results['gap_area_pixels']} 像素")
                # print(f"最小距离: {fhsdistance_results['min_distance_pixels']:.1f} 像素")
                if fhsdistance_results['best_pair']:
                    # 绘制上下边缘线
                    # cv2.polylines(overlay, [upper], False, (255, 0, 0), 1)  # 蓝色
                    # cv2.polylines(overlay, [lower], False, (0, 0, 255), 1)  # 红色

                    # 绘制最小距离连线
                    best_pair = fhsdistance_results['best_pair']
                    min_dist = fhsdistance_results['min_distance_pixels']
                    gap_area_pixels = fhsdistance_results['gap_area_pixels']
                    cv2.line(overlay,
                                tuple(best_pair[0]), tuple(best_pair[1]),
                                (0, 204, 255), 20)

                    # 添加测量标注
                    mid_pt = ((best_pair[0][0] + best_pair[1][0]) // 2,
                                (best_pair[0][1] + best_pair[1][1]) // 2)
                    cv2.putText(overlay, f"{min_dist:.1f}px",
                                (mid_pt[0] + 10, mid_pt[1] + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 204, 255), 8)
                    '''                
                    cv2.putText(overlay, f"{gap_area_pixels:.1f}px",
                                (mid_pt[0] + 80, mid_pt[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)'''
            # 合并结果
            result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            result_image = cv2.addWeighted(result_image, 0.5, overlay, 0.5, 0)

            # cv2.putText(result_image, title, (10, 30), 1, 1, (255, 255, 255), 5, 5)
            # 文本参数
            text = title
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            thickness = 8

            # 计算文本尺寸（含边距）
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font_face,
                font_scale,
                thickness
            )

            # 计算背景框位置（带10像素边距）
            x, y = 10, 20  # 文本起始坐标
            bg_x0 = x - 5
            bg_y0 = y - 5
            bg_x1 = x + text_width + 5
            bg_y1 = y + text_height + baseline + 10

            # 绘制白色背景框
            cv2.rectangle(
                    result_image,
                    (bg_x0, bg_y0),
                    (bg_x1, bg_y1),
                    (255, 255, 255,12),  # 白色填充
                    cv2.FILLED  # 填充模式
            )


            # 绘制黑色文本
            cv2.putText(
                result_image,
                text,
                (x, y + text_height + baseline - 2),  # 微调Y坐标避免截断
                font_face,
                font_scale,
                (255, 255, 255),  # 黑色文本
                thickness,
                cv2.LINE_AA
            )


        elif result.masks is None and result.boxes is not None:
            # 提取所有边界框坐标和分割掩码数据
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy()
            # 遍历所有检测结果
            for i in range(len(boxes_xyxy)):
                # 解析坐标、置信度、类别
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                conf = confs[i]
                cls = int(clses[i])
                color = colors.get(cls, (0, 255, 0))

                # 绘制边界框和文本
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=6)
                text = f"{class_names[cls]} : {conf:.2f}"

                draw.text((x1, y1 - 100), text, font=font, fill=color)

            #
            # 合并结果
            result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            result_image = cv2.addWeighted(result_image, 0.5, overlay, 0.5, 0)

            text = title
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 4

            # 计算文本尺寸（含边距）
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font_face,
                font_scale,
                thickness
            )

            # 计算背景框位置（带10像素边距）
            x, y = 10, 20  # 文本起始坐标
            bg_x0 = x - 5
            bg_y0 = y - 5
            bg_x1 = x + text_width + 5
            bg_y1 = y + text_height + baseline + 10

            # 绘制白色背景框
            cv2.rectangle(
                result_image,
                (bg_x0, bg_y0),
                (bg_x1, bg_y1),
                (255, 255, 255, 12),  # 白色填充
                cv2.FILLED  # 填充模式
            )

            # 绘制黑色文本
            cv2.putText(
                result_image,
                text,
                (x, y + text_height + baseline - 2),  # 微调Y坐标避免截断
                font_face,
                font_scale,
                (0, 0, 0),  # 黑色文本
                thickness,
                cv2.LINE_AA
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
                                                                       ['II', 'III', 'IV', 'nVNFH'], None)

            fhsdistance_results = fhsdistance.process_image(seg_js_model, img_path, None)  # 计算间隙最短距离及面积---20251010
            if fhsdistance_results:
                print(f"间隙面积: {fhsdistance_results['gap_area_pixels']} 像素")
                print(f"最小距离: {fhsdistance_results['min_distance_pixels']:.1f} 像素")

            # 绘制文本到原始图像
            image_rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
            img_pil3 = Image.fromarray(image_rgb)
            img_pil = img_pil3.convert('RGBA')
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("Arial", 60)
            except OSError:
                # font = ImageFont.truetype("DejaVuSans.ttf", 40)
                # 使用默认字体
                font = ImageFont.load_default(60)
            if fhsdistance_results:
                text = (
                    f"Combine Ficat : {ficat}\nFH : {FH}\nFHS : {JS}\nFHS area : {fhsdistance_results['gap_area_pixels']:.1f}px\n"
                    f"FHS min distance : {fhsdistance_results['min_distance_pixels']:.1f}px\n")
            else:
                text = f"Combine Ficat :{ficat}\nFH : {FH}\nFHS : {JS}\n"
            # text = f"Combine Ficat:{ficat}\nFH:{FH}\nFHS:{JS}\nTOP2:{TOP}"
            # draw.rectangle([])
            text_position = (10, 120)

            # 文本尺寸计算
            bbox = draw.textbbox((0, 0), text, font=font)  # 获取边界框
            # print('bbox---:',bbox)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # 绘制背景框
            box_position = (text_position[0] - 7, text_position[1] + 7)
            box_size = (text_width + 15, text_height + 15)
            # gray_color=(128,128,128)
            draw.rectangle(
                [box_position, (box_position[0] + box_size[0], box_position[1] + box_size[1])],
                fill=(255, 255, 255, 12)
            )

            # 绘制文本
            draw.text(text_position, text, font=font, fill=(0, 0, 0, 255))  # "red","black"
            combine_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)

            # 生成分割结果图像（确保尺寸与原始图像一致）
            seg_fh_image = ModelEvaluator.draw_results_on_model_image(seg_fh_result, img_path, "FH segmentation",
                                                                          ['NFH', 'IIFH', 'AFH', 'fusion'], None)

            seg_js_image = ModelEvaluator.draw_results_on_model_image(seg_js_result, img_path, "FHS segmentation",
                                                                          ['NJS', 'AJS'], fhsdistance_results)

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
    # 加载训练好的模型
    models = {
        'DecisionTree': joblib.load(os.path.join(Config.MODEL_SAVE_DIR, 'DecisionTree_best_model.pkl')),
        'RandomForest': joblib.load(os.path.join(Config.MODEL_SAVE_DIR, 'RandomForest_best_model.pkl')),
        'XGBoost': joblib.load(os.path.join(Config.MODEL_SAVE_DIR, 'XGBoost_best_model.pkl'))
    }

    # 执行验证集评估
    val_results = ModelEvaluator.evaluate_all(models, Config.VAL_TXT)

    # 执行测试集评估
    test_results = ModelEvaluator.evaluate_all(models, Config.TEST_TXT)

    # 可视化关键结果
    save_path = "/home/dyy/sda/dyy/v11/YOLOv11/runs-wah/combine1111/"
    ModelEvaluator.visualize_results(test_results, "test set", save_path)

    # 保存结果到CSV
    #pd.DataFrame(test_results).to_csv(Config.TEST_RESULTS_CSV)


    # 可视化关键结果
    for model_name, metrics in val_results.items():
        print(f"\n{model_name} Validation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print("Classification Report:")
        print(metrics['report'])
