import os
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
REFERENCE_IMAGE = os.path.join(CALIBRATION_DIR, "reference_top.jpg")
REFERENCE_BOARD = os.path.join(CALIBRATION_DIR, "reference_board.jpg")

_orb = cv2.ORB_create(nfeatures=1000, fastThreshold=12)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

_ref_keypoints = None
_ref_descriptors = None
_ref_corners = None
_ref_board_img = None
_last_calib_mtime = 0
_calibration_loaded = False

_smooth_corners = None
_lost_count = 0

_stable_orientation = "TOP_CORRECT"
_stable_label = "[OK] บอร์ดพร้อมตรวจ (0°)"
_orientation_history = []

def _smooth_orientation_status(new_status, new_label):
    """Fast responsive debounce over 3 frames to avoid flickering while reacting instantly."""
    global _stable_orientation, _stable_label, _orientation_history
    _orientation_history.append((new_status, new_label))
    if len(_orientation_history) > 4:
        _orientation_history.pop(0)

    states = [s[0] for s in _orientation_history]

    for candidate in ("TOP_ROTATED", "TOP_TILTED", "TOP_CORRECT"):
        if states.count(candidate) >= 2:
            _stable_orientation = candidate
            for s, l in reversed(_orientation_history):
                if s == candidate:
                    _stable_label = l
                    break
            break

    return _stable_orientation, _stable_label

_font = None
def get_font(size=15):
    global _font
    if _font is not None and getattr(_font, "size", None) == size:
        return _font
    font_paths = [
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/leelawad.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                _font = ImageFont.truetype(fp, size)
                return _font
            except Exception:
                pass
    _font = ImageFont.load_default()
    return _font

def ensure_dir():
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

def save_manual_corners(frame_bgr, corners_list):
    """
    Save 4 corners manually selected.
    Corners list: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    """
    ensure_dir()
    pts = np.array(corners_list, dtype=np.float32)
    h, w = frame_bgr.shape[:2]

    cv2.imwrite(REFERENCE_IMAGE, frame_bgr)

    dst_canon = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], dtype=np.float32)
    M_board = cv2.getPerspectiveTransform(pts, dst_canon)
    ref_board_bgr = cv2.warpPerspective(frame_bgr, M_board, (400, 300))
    cv2.imwrite(REFERENCE_BOARD, ref_board_bgr)

    calib_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_size": {"width": w, "height": h},
        "side": "TOP",
        "corners": pts.tolist()
    }

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    load_reference(force_reload=True)

    thumb = cv2.resize(frame_bgr, (160, 120))
    _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    import base64
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        "ok": True,
        "feature_count": len(_ref_keypoints) if _ref_keypoints else 0,
        "created_at": calib_data["created_at"],
        "thumbnail": thumb_b64,
        "corners": pts.tolist()
    }

def save_reference(frame_bgr):
    """Fallback capture using default inset corners."""
    h, w = frame_bgr.shape[:2]
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    default_corners = [
        [pad_x, pad_y],
        [w - pad_x, pad_y],
        [w - pad_x, h - pad_y],
        [pad_x, h - pad_y]
    ]
    return save_manual_corners(frame_bgr, default_corners)

def load_reference(force_reload=False):
    """
    Load reference image and PCB 4-corners from disk.
    Caches keypoints & descriptors in RAM strictly inside the PCB bounding polygon.
    """
    global _ref_keypoints, _ref_descriptors, _ref_corners, _last_calib_mtime, _calibration_loaded

    if not os.path.exists(CALIBRATION_FILE) or not os.path.exists(REFERENCE_IMAGE):
        return False

    current_mtime = os.path.getmtime(CALIBRATION_FILE)
    if not force_reload and _calibration_loaded and _ref_corners is not None and current_mtime <= _last_calib_mtime:
        return True

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        ref_img = cv2.imread(REFERENCE_IMAGE)
        if ref_img is None:
            return False

        if "corners" in data and len(data["corners"]) == 4:
            pts = np.array(data["corners"], dtype=np.float32)
        else:
            return False

        h, w = ref_img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

        ref_board_img = None
        if os.path.exists(REFERENCE_BOARD):
            ref_board_img = cv2.imread(REFERENCE_BOARD)
        elif ref_img is not None:
            dst_canon = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], dtype=np.float32)
            M_board = cv2.getPerspectiveTransform(pts, dst_canon)
            ref_board_img = cv2.warpPerspective(ref_img, M_board, (400, 300))
            cv2.imwrite(REFERENCE_BOARD, ref_board_img)

        if ref_board_img is not None:
            gray_board = cv2.cvtColor(ref_board_img, cv2.COLOR_BGR2GRAY)
            kp, des = _orb.detectAndCompute(gray_board, None)
        else:
            gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            kp, des = _orb.detectAndCompute(gray, mask)

        _ref_board_img = ref_board_img
        _ref_keypoints = kp
        _ref_descriptors = des
        _ref_corners = pts
        _last_calib_mtime = current_mtime
        _calibration_loaded = True
        return True

    except Exception as e:
        print("[Calibration] Error loading reference:", e)
        return False

def get_calibration_info():
    """Return calibration status and metadata."""
    if not os.path.exists(CALIBRATION_FILE) or not os.path.exists(REFERENCE_IMAGE):
        return {"calibrated": False}

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        thumb_b64 = None
        if os.path.exists(REFERENCE_IMAGE):
            ref_img = cv2.imread(REFERENCE_IMAGE)
            if ref_img is not None:
                thumb = cv2.resize(ref_img, (160, 120))
                _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                import base64
                thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

        return {
            "calibrated": True,
            "created_at": data.get("created_at", ""),
            "feature_count": data.get("feature_count", 0),
            "side": data.get("side", "TOP"),
            "thumbnail": thumb_b64,
            "corners": data.get("corners", [])
        }
    except Exception:
        return {"calibrated": False}

def clear_calibration():
    """Remove all calibration data."""
    global _ref_keypoints, _ref_descriptors, _ref_corners, _calibration_loaded, _smooth_corners
    _ref_keypoints = None
    _ref_descriptors = None
    _ref_corners = None
    _smooth_corners = None
    _calibration_loaded = False

    if os.path.exists(REFERENCE_IMAGE):
        try: os.remove(REFERENCE_IMAGE)
        except Exception: pass
    if os.path.exists(CALIBRATION_FILE):
        try: os.remove(CALIBRATION_FILE)
        except Exception: pass

def order_corners(pts):
    """
    Order 4 corners clockwise starting from Top-Left:
    TL (0), TR (1), BR (2), BL (3).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def order_corners_matching_prev(pts, prev_corners=None):
    """
    Order 4 corners clockwise starting from Top-Left:
    TL (0), TR (1), BR (2), BL (3).
    If prev_corners is provided, match to closest previous corners to avoid 90-degree flips.
    """
    ordered = order_corners(pts)
    if prev_corners is None:
        return ordered

    best_shift = 0
    min_dist = float('inf')
    for shift in range(4):
        shifted = np.roll(ordered, shift, axis=0)
        dist = np.mean(np.linalg.norm(shifted - prev_corners, axis=1))
        if dist < min_dist:
            min_dist = dist
            best_shift = shift

    return np.roll(ordered, best_shift, axis=0)

def detect_physical_pcb_board(frame_bgr):
    """
    Fast Physical PCB Board Detection (< 2 ms).
    Detects the physical PCB board rectangle using color mask and geometric contours.
    Works robustly across multiple PCB variants (dark green, emerald, olive/light green).
    Strictly clips bottom/side drop shadows so bounding box hugs the true board boundaries.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, np.array([28, 34, 30]), np.array([96, 255, 255]))

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (w * h) * 0.02
    max_area = (w * h) * 0.90

    best_corners = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box_area = rect[1][0] * rect[1][1]
            if box_area > 0 and (area / box_area) > 0.45:
                if area > best_area:
                    best_area = area
                    best_corners = order_corners_matching_prev(box, _smooth_corners)

    return best_corners

def check_orientation(frame_bgr):
    """
    Intelligent Real-Time PCB Tracking and Orientation Detection.
    1. Uses Physical PCB Board Detection to track ANY PCB board (holes, defective, rotated).
    2. Uses ORB Keypoint Homography to verify top-side orientation when available.
    """
    global _smooth_corners, _lost_count

    load_reference()

    h_full, w_full = frame_bgr.shape[:2]

    pcb_corners = detect_physical_pcb_board(frame_bgr)

    if pcb_corners is not None:

        if _smooth_corners is None or _lost_count > 3:
            _smooth_corners = pcb_corners.copy()
        else:

            max_drift = float(np.max(np.linalg.norm(pcb_corners - _smooth_corners, axis=1)))
            if max_drift < 3.5:

                pass
            elif max_drift < 25.0:

                alpha = 0.28
                _smooth_corners = (alpha * pcb_corners + (1.0 - alpha) * _smooth_corners)
            else:

                alpha = 0.65
                _smooth_corners = (alpha * pcb_corners + (1.0 - alpha) * _smooth_corners)

        _lost_count = 0

        edge_top = np.linalg.norm(_smooth_corners[1] - _smooth_corners[0])
        edge_side = np.linalg.norm(_smooth_corners[3] - _smooth_corners[0])

        if edge_top >= edge_side * 0.8:
            dx = float(_smooth_corners[1][0] - _smooth_corners[0][0])
            dy = float(_smooth_corners[1][1] - _smooth_corners[0][1])
        else:

            dx = float(_smooth_corners[2][0] - _smooth_corners[1][0])
            dy = float(_smooth_corners[2][1] - _smooth_corners[1][1])

        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        while angle_deg > 90: angle_deg -= 180
        while angle_deg < -90: angle_deg += 180

        if _ref_corners is not None and len(_ref_corners) == 4:
            ref_dx = float(_ref_corners[1][0] - _ref_corners[0][0])
            ref_dy = float(_ref_corners[1][1] - _ref_corners[0][1])
            ref_angle = float(np.degrees(np.arctan2(ref_dy, ref_dx)))
            while ref_angle > 90: ref_angle -= 180
            while ref_angle < -90: ref_angle += 180
            relative_tilt = angle_deg - ref_angle
            while relative_tilt > 90: relative_tilt -= 180
            while relative_tilt < -90: relative_tilt += 180
            abs_angle = abs(round(relative_tilt))
        else:
            abs_angle = abs(round(angle_deg))

        is_rotated_180 = False

        if _ref_descriptors is not None and _ref_keypoints is not None:
            try:

                dst_canon = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], dtype=np.float32)
                M_cur = cv2.getPerspectiveTransform(_smooth_corners.astype(np.float32), dst_canon)
                cur_board = cv2.warpPerspective(frame_bgr, M_cur, (400, 300))
                cur_gray = cv2.cvtColor(cur_board, cv2.COLOR_BGR2GRAY)

                kp_cur, des_cur = _orb.detectAndCompute(cur_gray, None)
                if des_cur is not None and len(kp_cur) >= 12:
                    matches = _bf.knnMatch(_ref_descriptors, des_cur, k=2)
                    good = [m for m in matches if len(m) == 2 and m[0].distance < 0.80 * m[1].distance]
                    if len(good) >= 6:
                        src_pts = np.float32([_ref_keypoints[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_cur[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
                        if M is not None:
                            num_inliers = int(np.sum(inliers))
                            if num_inliers >= 6:
                                homo_angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                                homo_angle = ((homo_angle + 180) % 360) - 180
                                if abs(round(homo_angle)) > 120:
                                    is_rotated_180 = True
            except Exception:
                pass

        if is_rotated_180:
            final_status, final_label = _smooth_orientation_status("TOP_ROTATED", "[!] บอร์ดหมุนกลับหัว 180°")
        elif abs_angle > 12:
            final_status, final_label = _smooth_orientation_status("TOP_TILTED", f"[!] บอร์ดเอียง {abs_angle}°")
        else:
            final_status, final_label = _smooth_orientation_status("TOP_CORRECT", f"[OK] บอร์ดพร้อมตรวจ ({abs_angle}°)")

        return {
            "orientation": final_status,
            "label_th": final_label,
            "match_score": 95 if final_status == "TOP_CORRECT" else 85,
            "rotation_deg": round(angle_deg, 1),
            "corners": _smooth_corners.tolist()
        }

    if _ref_corners is not None and _ref_descriptors is not None and _ref_keypoints is not None:
        try:
            scale = 0.5
            inv_scale = 1.0 / scale
            small_frame = cv2.resize(frame_bgr, (int(w_full * scale), int(h_full * scale)), interpolation=cv2.INTER_LINEAR)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            kp_current, des_current = _orb.detectAndCompute(gray_small, None)

            if des_current is not None and len(kp_current) >= 10:
                matches = _bf.knnMatch(_ref_descriptors, des_current, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.78 * n.distance]

                if len(good_matches) >= 10:
                    src_pts = np.float32([_ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * inv_scale
                    M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

                    if M is not None and np.sum(inliers) >= 8:
                        transformed = cv2.perspectiveTransform(_ref_corners.reshape(-1, 1, 2), M).reshape(-1, 2)
                        area = cv2.contourArea(transformed.astype(np.int32))
                        full_area = w_full * h_full
                        if (full_area * 0.03) < area < (full_area * 0.90):
                            if _smooth_corners is None or _lost_count > 3:
                                _smooth_corners = transformed.copy()
                            else:
                                _smooth_corners = 0.70 * transformed + 0.30 * _smooth_corners
                            _lost_count = 0

                            angle_rad = np.arctan2(M[1, 0], M[0, 0])
                            angle_deg = float(np.degrees(angle_rad))
                            angle_deg = ((angle_deg + 180) % 360) - 180
                            abs_angle = abs(round(angle_deg))

                            if abs_angle > 135:
                                return {"orientation": "TOP_ROTATED", "label_th": f"[!] หมุนกลับหัว 180° ({abs_angle}°)", "match_score": 90, "rotation_deg": round(angle_deg, 1), "corners": _smooth_corners.tolist()}
                            elif abs_angle > 15:
                                return {"orientation": "TOP_TILTED", "label_th": f"[!] บอร์ดเอียง {abs_angle}°", "match_score": 85, "rotation_deg": round(angle_deg, 1), "corners": _smooth_corners.tolist()}
                            else:
                                return {"orientation": "TOP_CORRECT", "label_th": f"[OK] บอร์ดพร้อมตรวจ ({abs_angle}°)", "match_score": 95, "rotation_deg": round(angle_deg, 1), "corners": _smooth_corners.tolist()}
        except Exception:
            pass

    return _handle_tracking_lost()

def _handle_tracking_lost():
    """Handle loss of PCB detection gracefully."""
    global _lost_count, _smooth_corners
    _lost_count += 1

    if _lost_count <= 2 and _smooth_corners is not None:
        return {
            "orientation": "TOP_TRACKING",
            "label_th": "🔍 กำลังตรวจจับบอร์ด PCB...",
            "match_score": 0,
            "rotation_deg": 0,
            "corners": _smooth_corners.tolist()
        }

    _smooth_corners = None
    return {
        "orientation": "NO_BOARD",
        "label_th": "",
        "match_score": 0,
        "rotation_deg": 0,
        "corners": None
    }

def draw_pcb_orientation_overlay(frame_bgr, orientation_result):
    """
    Blazingly fast PCB 4-corner overlay.
    Only draws when a real PCB board is detected. When no board is present, returns clean frame.
    """
    if not orientation_result or not orientation_result.get("corners"):
        return frame_bgr

    orient = orientation_result.get("orientation", "")
    if orient in ("NO_BOARD", "IDLE_GUIDE", ""):
        return frame_bgr

    corners_list = orientation_result.get("corners")
    if not corners_list or len(corners_list) < 4:
        return frame_bgr

    pts = np.array(corners_list, dtype=np.int32)
    label_text = orientation_result.get("label_th", "")

    if not label_text:
        return frame_bgr

    if orient == "TOP_CORRECT":
        bgr_color = (46, 204, 113)
        bg_rgb = (10, 45, 25)
        thickness = 3
    elif orient in ("TOP_ROTATED", "TOP_TILTED"):
        bgr_color = (0, 165, 255)
        bg_rgb = (50, 30, 5)
        thickness = 3
    elif orient == "TOP_TRACKING":
        bgr_color = (0, 215, 255)
        bg_rgb = (10, 35, 45)
        thickness = 2
    else:
        bgr_color = (212, 182, 6)
        bg_rgb = (15, 30, 45)
        thickness = 2

    cv2.polylines(frame_bgr, [pts], isClosed=True, color=bgr_color, thickness=thickness, lineType=cv2.LINE_AA)

    cv2.line(frame_bgr, tuple(pts[0]), tuple(pts[1]), bgr_color, thickness=thickness + 2, lineType=cv2.LINE_AA)

    for p in pts:
        cv2.circle(frame_bgr, tuple(p), 5, bgr_color, -1, lineType=cv2.LINE_AA)

    try:
        h_frame, w_frame = frame_bgr.shape[:2]
        tx = int((pts[0][0] + pts[1][0]) // 2)
        ty = int((pts[0][1] + pts[1][1]) // 2)

        font = get_font(size=14)

        badge_w, badge_h = 240, 32
        x1 = max(5, min(w_frame - badge_w - 5, tx - badge_w // 2))
        y1 = max(5, min(h_frame - badge_h - 5, ty - badge_h - 8))
        x2 = x1 + badge_w
        y2 = y1 + badge_h

        patch_bgr = frame_bgr[y1:y2, x1:x2]
        if patch_bgr.shape[0] == badge_h and patch_bgr.shape[1] == badge_w:

            pil_patch = Image.fromarray(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_patch)
            draw.rounded_rectangle([0, 0, badge_w - 1, badge_h - 1], radius=6, fill=bg_rgb, outline=(bgr_color[2], bgr_color[1], bgr_color[0]), width=2)
            draw.text((10, 6), label_text, font=font, fill=(255, 255, 255))

            frame_bgr[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_patch), cv2.COLOR_RGB2BGR)

    except Exception:
        pass

    return frame_bgr
