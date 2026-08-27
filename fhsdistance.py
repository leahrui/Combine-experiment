"""Femoral head joint-space segmentation distance and gap-area computation utility.

Optimized version: vectorized NumPy boolean filters replace Python per-point loops;
a single KDTree.query batch call replaces per-point lookups; callers can pass
``orig_image`` + ``precomputed_results`` to avoid redundant ``cv2.imread`` and
duplicate YOLO inference.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from scipy.spatial import KDTree


def load_model(model_path):
    """Load a YOLOv11 segmentation model; returns None on failure."""
    try:
        return YOLO(model_path)
    except Exception:
        return None


def preprocess_image(image_path, target_size=(640, 640), orig_image=None):
    """Letterbox an image to ``target_size`` and return (padded_array, orig_info).

    Parameters
    ----------
    image_path : str
        Source image path (used only when ``orig_image`` is None).
    target_size : (int, int)
        Target (H, W); defaults to (640, 640) for YOLO.
    orig_image : np.ndarray, optional
        Pre-loaded BGR image; avoids an extra ``cv2.imread`` when provided.

    Returns
    -------
    (np.ndarray, dict) | (None, None)
        RGB-padded image + geometry metadata for mask postprocessing.
    """
    try:
        image = orig_image if orig_image is not None else cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (new_w, new_h))

        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image_padded = cv2.copyMakeBorder(
            image_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        orig_info = {
            'orig_shape': (h, w),
            'pad': (pad_top, pad_bottom, pad_left, pad_right),
            'scale': scale
        }
        return image_padded, orig_info
    except Exception:
        return None, None


def postprocess_mask(mask, orig_info):
    """Undo letterbox padding + resize a segmentation mask to the original image size."""
    try:
        pad_top, pad_bottom, pad_left, pad_right = orig_info['pad']
        orig_h, orig_w = orig_info['orig_shape']
        if mask.ndim > 2:
            mask = np.argmax(mask, axis=0)
        mask = mask[pad_top:mask.shape[0] - pad_bottom, pad_left:mask.shape[1] - pad_right]
        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        return None


def split_contour_by_two_diagonals(contour, bounding_rect, proportion):
    """Split a contour by the two diagonals of its bounding rect (vectorized).

    Parameters
    ----------
    contour : array-like
        Input contour with shape ``(N, 1, 2)`` or ``(N, 2)``.
    bounding_rect : (x, y, w, h)
        Bounding rectangle produced by ``cv2.boundingRect``.
    proportion : {'quarter', 'half'}
        ``'quarter'`` keeps points below both diagonals (intersection);
        ``'half'`` keeps points below whichever diagonal covers more points.

    Returns
    -------
    (upper_points : ndarray (M, 2), lower_points : ndarray (K, 2))
    """
    x, y, w, h = bounding_rect

    # Diagonal 1: TL -> BR
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    if x2 != x1:
        k1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - k1 * x1
    else:
        k1, b1 = None, x1  # vertical

    # Diagonal 2: TR -> BL
    x3, y3 = x + w, y
    x4, y4 = x, y + h
    if x4 != x3:
        k2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - k2 * x3
    else:
        k2, b2 = None, x3  # vertical

    # Intersection of the two diagonals
    if k1 is not None and k2 is not None and k1 != k2:
        xc = (b2 - b1) / (k1 - k2)
        yc = k1 * xc + b1
    elif k1 is None:
        xc, yc = b1, k2 * b1 + b2
    elif k2 is None:
        xc, yc = b2, k1 * b2 + b1
    else:
        xc, yc = None, None  # parallel

    # --- Vectorized point filtering -------------------------------------------------
    contour_arr = np.asarray(contour)
    if contour_arr.ndim == 3:
        contour_arr = contour_arr[:, 0, :] if contour_arr.shape[1] == 1 else contour_arr.reshape(-1, 2)
    elif contour_arr.ndim == 1:
        contour_arr = contour_arr.reshape(-1, 2)
    if contour_arr.size == 0 or contour_arr.ndim != 2 or contour_arr.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    px = contour_arr[:, 0]
    py = contour_arr[:, 1]

    y_diag1 = None if k1 is None else k1 * px + b1
    y_diag2 = None if k2 is None else k2 * px + b2

    if k1 is None and k2 is None:
        save_mask = (px >= min(b1, b2)) & (px <= max(b1, b2))
    elif k1 is None:
        save_mask = py <= y_diag2
    elif k2 is None:
        save_mask = ((px <= b2) & (py >= y_diag1)) | ((px >= b2) & (py <= y_diag1))
    elif proportion == 'quarter':
        save_mask = py <= np.minimum(y_diag1, y_diag2)
    else:  # proportion == 'half'
        m_left = py <= y_diag1
        m_right = py <= y_diag2
        n_left = int(np.count_nonzero(m_left))
        n_right = int(np.count_nonzero(m_right))
        if n_left == 0 and n_right == 0:
            save_mask = m_left
        elif n_left >= n_right:
            save_mask = m_left
        else:
            save_mask = m_right

    points = contour_arr[save_mask]
    if points.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    x_pts = points[:, 0]
    y_pts = points[:, 1]

    # Decide which diagonal was used (for splitting upper/bottom arcs)
    n_left_all = n_left_all2 = n_right_all = n_right_all2 = 0
    if k1 is not None and k2 is not None and proportion == 'half':
        n_left_all = int(np.count_nonzero(py <= y_diag1))
        n_right_all = int(np.count_nonzero(py <= y_diag2))
    use_left = (n_left_all != 0 or n_right_all != 0) and (n_left_all >= n_right_all)
    use_right = (n_left_all != 0 or n_right_all != 0) and (n_right_all > n_left_all)

    left_idx = int(np.argmin(x_pts))
    right_idx = int(np.argmax(x_pts))
    has_open = (n_left_all != 0 or n_right_all != 0)

    if has_open:
        if use_left:
            bottom_idx = int(np.argmax(y_pts))
            if left_idx <= bottom_idx:
                bottom = points[left_idx:bottom_idx + 1]
                upper = np.vstack([points[:left_idx], points[bottom_idx + 1:]])
            else:
                bottom = np.vstack([points[left_idx:], points[:bottom_idx + 1]])
                upper = points[bottom_idx + 1:left_idx]
        else:
            bottom_idx = int(np.where(y_pts == np.max(y_pts))[0][-1])
            if bottom_idx <= right_idx:
                bottom = points[bottom_idx:right_idx + 1]
                upper = np.vstack([points[:bottom_idx], points[right_idx + 1:]])
            else:
                bottom = np.vstack([points[bottom_idx:], points[:right_idx + 1]])
                upper = points[right_idx + 1:bottom_idx]
    else:
        if left_idx <= right_idx:
            bottom = points[left_idx:right_idx + 1]
            upper = np.vstack([points[:left_idx], points[right_idx + 1:]])
        else:
            bottom = np.vstack([points[left_idx:], points[:right_idx + 1]])
            upper = points[right_idx + 1:left_idx]

    # Trim edge artifacts on the bottom arc
    if bottom.shape[0] > 14:
        bottom = bottom[7:-7]
    else:
        bottom = bottom[0:0]
    return np.asarray(upper), np.asarray(bottom)


def analyze_hip_gap(mask, orig_image, visualize=False):
    """Compute the upper/lower hip-gap edge minimum distance and gap pixel area.

    Parameters
    ----------
    mask : np.ndarray
        Binarized hip-space segmentation mask.
    orig_image : np.ndarray
        Original BGR image (used only when ``visualize`` is True).
    visualize : bool
        If True, draw the contour + shortest-distance line on a copy.

    Returns
    -------
    (analysis_dict, vis_image)
        ``analysis_dict`` contains ``min_distance_pixels``, ``best_pair`` and
        ``gap_area_pixels``; ``vis_image`` is None unless ``visualize`` is set.
    """
    combined_mask = (mask > 0.0).astype(np.uint8)
    gap_area_pixels = int(np.sum(combined_mask))

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    main_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(main_contour)
    upper_points, lower_points = split_contour_by_two_diagonals(main_contour, (x, y, w, h), proportion='half')

    # Discard the leftmost/rightmost 5% horizontal regions (vectorized filter)
    margin = int(w * 0.05)

    def _filter(pts):
        if pts is None or pts.size == 0:
            return np.empty((0, 2))
        arr = np.asarray(pts)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        if arr.shape[0] == 0:
            return arr
        keep = (arr[:, 0] >= x + margin) & (arr[:, 0] <= x + w - margin)
        return arr[keep]

    upper = _filter(upper_points)
    lower = _filter(lower_points)
    if len(upper) == 0 or len(lower) == 0:
        return None, None

    # Batch KDTree nearest-neighbour over ALL upper points (single query call)
    lower_tree = KDTree(lower[:, :2])
    dists, idxs = lower_tree.query(upper[:, :2])
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    best_pair = (upper[min_idx], lower[idxs[min_idx]])

    vis_image = orig_image.copy() if orig_image is not None else None
    if visualize and best_pair and vis_image is not None:
        overlay = orig_image.copy()
        cv2.drawContours(overlay, [main_contour], -1, (0, 255, 0), 2)
        cv2.line(overlay, tuple(best_pair[0]), tuple(best_pair[1]), (0, 255, 255), 2)
        mid_pt = ((best_pair[0][0] + best_pair[1][0]) // 2,
                  (best_pair[0][1] + best_pair[1][1]) // 2)
        cv2.putText(overlay, f"{min_dist:.1f}px",
                    (mid_pt[0] - 30, mid_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        vis_image = overlay

    return {
        "min_distance_pixels": min_dist,
        "best_pair": best_pair,
        "gap_area_pixels": gap_area_pixels,
    }, vis_image


def process_image(model, image_path, save_path=None, orig_image=None, precomputed_results=None):
    """End-to-end joint-space analysis for one image.

    When ``precomputed_results`` is provided (a YOLO ``Results`` object from the
    JS segmentation model) both ``preprocess_image`` and the YOLO forward pass are
    skipped entirely; callers that already ran segmentation (e.g. for the
    classification feature pipeline) should pass the object to avoid redundant work.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO segmentation model (used only when ``precomputed_results`` is None).
    image_path : str
        Path to the original image.
    save_path : str | None
        If set, ``analyze_hip_gap`` runs with ``visualize=True`` and the overlay is
        written to ``save_path``.
    orig_image : np.ndarray, optional
        Pre-loaded BGR image; avoids an extra read.
    precomputed_results : list | None
        Pre-computed YOLO ``Results`` for the JS segmentation model.
    """
    if precomputed_results is not None:
        # Reuse cached YOLO outputs -------------------------------------------------------
        if (not precomputed_results
                or not hasattr(precomputed_results[0], 'masks')
                or precomputed_results[0].masks is None):
            return None
        if orig_image is None:
            orig_image = cv2.imread(image_path)
            if orig_image is None:
                return None

        mask_data = precomputed_results[0].masks.data.cpu().numpy()
        mh, mw = mask_data.shape[-2], mask_data.shape[-1]
        oh, ow = orig_image.shape[:2]
        r = min(mh / oh, mw / ow)
        new_h, new_w = int(round(oh * r)), int(round(ow * r))
        pad_h = mh - new_h
        pad_w = mw - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        orig_info = {
            'orig_shape': (oh, ow),
            'pad': (pad_top, pad_bottom, pad_left, pad_right),
            'scale': r,
        }
        combined_mask = np.max(mask_data, axis=0) if mask_data.ndim == 3 else mask_data
        combined_mask = postprocess_mask(combined_mask, orig_info)
        orig_image_use = orig_image
    else:
        # Standard: preprocess -> YOLO inference -> postprocess ---------------------------
        image, orig_info = preprocess_image(image_path, orig_image=orig_image)
        if image is None:
            return None
        results = model(image, conf=0.25, iou=0.5, verbose=False)
        if (not results
                or not hasattr(results[0], 'masks')
                or results[0].masks is None):
            return None
        mask_data = results[0].masks.data.cpu().numpy()
        combined_mask = np.max(mask_data, axis=0) if mask_data.ndim == 3 else mask_data
        combined_mask = postprocess_mask(combined_mask, orig_info)
        orig_image_use = orig_image if orig_image is not None else cv2.imread(image_path)

    try:
        analysis, vis_image = analyze_hip_gap(
            combined_mask, orig_image_use, visualize=(save_path is not None)
        )
        if save_path and vis_image is not None:
            cv2.imwrite(save_path, vis_image)
        return analysis
    except IndexError:
        return None


def main():
    # Standalone smoke-demo entry point. EDIT these paths before running.
    model_path = r"segFHS model path\weights\best.pt"
    image_path = r"xx.jpg"
    folder_path = r"path"
    save_filepath = r"path\\"

    model = load_model(model_path)
    if not model:
        return
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                save_path = os.path.join(save_filepath, filename)
                process_image(model, file_path, save_path)


if __name__ == "__main__":
    main()
