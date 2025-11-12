import cv2
import numpy as np
from ultralytics import YOLO
import os
from scipy.spatial import KDTree

def load_model(model_path):
    """加载YOLOv11分割模型"""
    try:
        model = YOLO(model_path)   # 假设YOLOv11与YOLOv8接口兼容
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def preprocess_image(image_path, target_size=(640, 640)):
    """预处理图像"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 调整大小并保持宽高比
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        # 计算填充（上下左右）
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # 填充图像
        image_padded = cv2.copyMakeBorder(
            image_resized,
            pad_top, pad_bottom,
            pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        # 保存原始信息用于后处理
        orig_info = {
            'orig_shape': (h, w),
            'pad': (pad_top, pad_bottom, pad_left, pad_right),  # 四个方向的填充
            'scale': scale
        }
        return image_padded, orig_info
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None, None

def postprocess_mask(mask, orig_info):
    """后处理分割掩码"""
    try:
        pad_top, pad_bottom, pad_left, pad_right = orig_info['pad']
        orig_h, orig_w = orig_info['orig_shape']
        scale = orig_info['scale']
        # 确保mask是二维的
        if len(mask.shape) > 2:
            mask = np.argmax(mask, axis=0)
        # 去除填充
        mask = mask[pad_top:mask.shape[0]-pad_bottom,  # 裁剪上下填充
                   pad_left:mask.shape[1]-pad_right]   # 裁剪左右填充
        # 调整回原始尺寸
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception as e:
        print(f"掩码后处理失败: {e}")
        return None

def split_contour_by_two_diagonals(contour, bounding_rect, proportion):
    """
    根据边界矩形的两条对角线分割轮廓点，保留四边形区域内的点
    :param contour: 输入轮廓 (N, 1, 2)
    :param bounding_rect: (x, y, w, h) 边界矩形
    :param proportion: 分割方式，四分之一分割还是二分之一分割，'quarter' or 'half'
    :return: upper_points（上边缘的点）, bottom_points（下边缘的点）
    """
    x, y, w, h = bounding_rect
    contour = contour.squeeze()  # 转换为 (N, 2)

    # 计算两条对角线的斜率和截距
    # 对角线1：左上到右下 (x, y) -> (x + w, y + h)
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    if x2 != x1:
        k1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - k1 * x1
    else:
        k1 = None
        b1 = x1  # 垂直对角线（x = x1）

    # 对角线2：右上到左下 (x + w, y) -> (x, y + h)
    x3, y3 = x + w, y
    x4, y4 = x, y + h
    if x4 != x3:
        k2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - k2 * x3
    else:
        k2 = None
        b2 = x3  # 垂直对角线（x = x3）
        # 计算两条对角线的交点
    if k1 is not None and k2 is not None and k1 != k2:
        xc = (b2 - b1) / (k1 - k2)
        yc = k1 * xc + b1
    elif k1 is None:  # 对角线1是垂直线（x = b1）
        xc = b1
        yc = k2 * xc + b2
    elif k2 is None:  # 对角线2是垂直线（x = b2）
        xc = b2
        yc = k1 * xc + b1
    else:  # 两条对角线平行（k1 == k2）
        xc, yc = None, None  # 无交点，无法定义“之下”区域

    save_points = []
    delete_points = []
    left_open = 0
    right_open = 0
    left_open_save_points = []
    left_open_delete_points = []
    right_open_save_points = []
    right_open_delete_points = []

    for pt in contour:
        px, py = pt
        # 检查点是否在矩形范围内
        # if not (x <= px <= x + w and y <= py <= y + h):
        #     delete_points.append(pt)
        #     continue

        # 计算两条对角线的 y 值
        if k1 is not None:
            y_diag1 = k1 * px + b1
        else:
            y_diag1 = b1  # 垂直对角线，比较 x 值
        if k2 is not None:
            y_diag2 = k2 * px + b2
        else:
            y_diag2 = b2  # 垂直对角线，比较 x 值

        if k1 is None:  # 对角线1是垂直线（x = b1）
            if k2 is not None:
                if py <= y_diag2:
                    save_points.append(pt)
                else:
                    delete_points.append(pt)
            else:  # 两条对角线都是垂直线
                if min(b1, b2) <= px <= max(b1, b2):
                    save_points.append(pt)
                else:
                    delete_points.append(pt)
        elif k2 is None:  # 对角线2是垂直线（x = b2）
            if (px <= b2 and py >= y_diag1) or (px >= b2 and py <= y_diag1):
                save_points.append(pt)
            else:
                delete_points.append(pt)
        # 当分割方式为四分之一上部
        elif k1 is not None and k2 is not None and proportion == 'quarter':
            # 判断点是否在两条对角线之间
            y_min = min(y_diag1, y_diag2)
            # y_max = max(y_diag1, y_diag2)
            if py <= y_min:
                save_points.append(pt)
            else:
                delete_points.append(pt)
        # 当分割方式为二分之一上部
        elif k1 is not None and k2 is not None and proportion == 'half':
            #  首先判断开口方向
            if py <= y_diag1:
                left_open = left_open+1
                left_open_save_points.append(pt)
            else:
                left_open_delete_points.append(pt)
            if py <= y_diag2:
                right_open = right_open+1
                right_open_save_points.append(pt)
            else:
                right_open_delete_points.append(pt)
    if left_open != 0 or right_open != 0:
        print(left_open)
        print(right_open)
        if left_open >= right_open:
            save_points = left_open_save_points
            delete_points = left_open_delete_points
        else:
            save_points = right_open_save_points
            delete_points = right_open_delete_points

    points = np.array(save_points)
    print(points)
    # points = inner_points
    x = points[:, 0]
    y = points[:, 1]

    # 找到最左和最右的点及其索引
    left_idx = np.argmin(x)
    right_idx = np.argmax(x)
    if left_open != 0 or right_open != 0:
        if left_open >= right_open:
            bottom_idx = np.argmax(y)
            # 确保 left_idx <= bottom_idx（否则需要循环处理）
            if left_idx <= bottom_idx:
                bottom = points[left_idx: bottom_idx + 1]
                upper = np.vstack([points[:left_idx], points[bottom_idx + 1:]])
            else:
                # 循环情况（例如，最左点在末尾，最右点在开头）
                bottom = np.vstack([points[left_idx:], points[:bottom_idx + 1]])
                upper = points[bottom_idx + 1: left_idx]
        else:   # 确保 bottom_idx <= right_idx（否则需要循环处理）
            bottom_idx = np.where(y == np.max(y))[0][-1]
            if bottom_idx <= right_idx:
                bottom = points[bottom_idx: right_idx + 1]
                upper = np.vstack([points[:bottom_idx], points[right_idx + 1:]])
            else:
                # 循环情况（例如，最左点在末尾，最右点在开头）
                bottom = np.vstack([points[bottom_idx:], points[:right_idx + 1]])
                upper = points[right_idx + 1: bottom_idx]
    else:
        # 确保 left_idx <= right_idx（否则需要循环处理）
        if left_idx <= right_idx:
            bottom = points[left_idx: right_idx + 1]
            upper = np.vstack([points[:left_idx], points[right_idx + 1:]])
        else:
            # 循环情况（例如，最左点在末尾，最右点在开头）
            bottom = np.vstack([points[left_idx:], points[:right_idx + 1]])
            upper = points[right_idx + 1: left_idx]
    # upper = upper[2:-2]
    bottom = bottom[7:-7]  # 去除一些头尾的点，保证准确性
    all_upper_points = upper.tolist()
    all_bottom_points = bottom.tolist()
    return np.array(all_upper_points), np.array(all_bottom_points)


def analyze_hip_gap(mask, orig_image, visualize=False):

    """
    处理髋关节间隙掩膜，计算上下边缘最小距离
    Args:
        mask: 分割模型输出的二值掩膜
        orig_image: 原始图像（用于可视化）
        visualize: 是否可视化结果
    Returns:
        min_distance: 上下边缘最小距离（像素）
        vis_image: 可视化结果图像
    """
    combined_mask = (mask > 0.0).astype(np.uint8)
    # 1. 形态学处理（去除小噪点）
    gap_area_pixels = np.sum(combined_mask)
    print("gap_area_pixels:",gap_area_pixels)
    print("combined_mask 类型:", combined_mask.dtype)  # 应输出 uint8

    # 2. 提取最大轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("not contours")
        return None, None
    main_contour = max(contours, key=cv2.contourArea)
    gap_mainarea_pixels = np.sum(main_contour)
    print("gap_mainarea_pixels:", gap_mainarea_pixels)

    # 3. 提取上下边缘线

    """将轮廓分解为上下边缘线"""
    # 获取轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(main_contour)
    # print("x,y,w,h:",x,y,w,h)

    # 根据对角线分割轮廓
    upper_points, lower_points = split_contour_by_two_diagonals(main_contour, (x, y, w, h), proportion='half')
    print(upper_points.shape)
    # 提取上下边缘点（排除左右5%区域）
    margin = int(w * 0.05)
    print("margin:", margin)
    upper_edge = []
    lower_edge = []
    #
    for pt in upper_points.squeeze():
        if x + margin <= pt[0] <= x + w - margin:
            upper_edge.append(pt)
    for pt in lower_points.squeeze():
        if x + margin <= pt[0] <= x + w - margin:
            lower_edge.append(pt)

    upper = np.array(upper_edge)
    lower = np.array(lower_edge)

    # print("upper:",upper)
    # 4. 计算最小距离
    if len(upper) == 0 or len(lower) == 0:
        print('upper:',len(upper))
        print('lower:', len(lower))
        return None, None

    # 构建下边缘KD树加速搜索
    lower_tree = KDTree(lower[:, :2])

    min_dist = float('inf')
    best_pair = None

    # 遍历上边缘点寻找最近邻
    for pt_upper in upper:
        dist, idx = lower_tree.query(pt_upper)
        if dist < min_dist:
            min_dist = dist
            best_pair = (pt_upper, lower[idx])

    # 5. 可视化结果
    vis_image = orig_image.copy()

    if visualize and best_pair:
        overlay = orig_image.copy()
        # 绘制掩膜轮廓
        cv2.drawContours(overlay, [main_contour], -1, (0, 255, 0), 2)

        # 绘制上下边缘线
        #cv2.polylines(overlay, [upper], False, (255, 0, 0), 1)  # 蓝色
        #cv2.polylines(overlay, [lower], False, (0, 0, 255), 1)  # 红色

        # 绘制最小距离连线
        cv2.line(overlay,
                 tuple(best_pair[0]), tuple(best_pair[1]),
                 (0, 255, 255), 2)

        # 添加测量标注
        mid_pt = ((best_pair[0][0] + best_pair[1][0]) // 2,
                  (best_pair[0][1] + best_pair[1][1]) // 2)
        cv2.putText(overlay, f"{min_dist:.1f}px",
                    (mid_pt[0] - 30, mid_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        vis_image = overlay

    #return min_dist, vis_image
    return {
               "min_distance_pixels": min_dist,
               "best_pair": best_pair,
                "gap_area_pixels":gap_area_pixels
           }, vis_image

def process_image(model, image_path, save_path=None):
    """处理图像并保存结果"""
    #try:
    # 预处理
    image, orig_info = preprocess_image(image_path)
    if image is None:
        print("preprocess_image is none")
        return None

    # 推理
    results = model(image, conf=0.25, iou=0.5)
    if not results or not hasattr(results[0], 'masks') or results[0].masks is None:
        return None
    else:
        # 处理掩码
        mask_data = results[0].masks.data.cpu().numpy()

        combined_mask = np.max(mask_data, axis=0) if len(mask_data.shape) == 3 else mask_data
        print('combined_mask:', combined_mask.dtype)
        combined_mask = postprocess_mask(combined_mask, orig_info)

        # 分析结果
        orig_image = cv2.imread(image_path)
        analysis, vis_image = analyze_hip_gap(combined_mask, orig_image, visualize=True)
        print("analysis:", analysis)

        # 保存结果
        if save_path and vis_image is not None:
            cv2.imwrite(save_path, vis_image)
            print(f"结果已保存至: {save_path}")
        return analysis


def main():
    #model_path = "your_model.pt"
    #image_path = "test_image.jpg"
    model_path = "D:\\YOLOv11\\runs\\train\\segJS-yolo11-xndp\\weights\\best.pt"  # 替换为实际的模型路径
    image_path = "E:\\1dataset0821\\det-0821\\xq\\minitest\\xq_20131205_04495119_395R.jpg"  # 替换为实际的图像路径
    # 指定文件夹路径
    folder_path = 'E:\\1dataset0821\\det-0821\\xq\\minitest'
    save_filepath = "E:\\1dataset0821\\det-0821\\xq\\TestResult\\"
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):  # 检查扩展名
                file_path = os.path.join(root, filename)
                print(f"正在处理图片: {file_path}")
                #save_filepath = "E:\\1dataset0821\\det-0821\\xq\\TestResult\\"
                save_path = os.path.join(save_filepath, filename)
                model = load_model(model_path)
                if model:
                    results = process_image(model, file_path, save_path)
                    print("results:", results)
                    if results:
                        print(f"间隙面积: {results['gap_area_pixels']} 像素")
                        print(f"最小距离: {results['min_distance_pixels']:.1f} 像素")


if __name__ == "__main__":
    main()
