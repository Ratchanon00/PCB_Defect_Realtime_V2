import os
import sys
import json
import time
import argparse
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(PROJECT_ROOT, "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
REFERENCE_IMAGE = os.path.join(CALIBRATION_DIR, "reference_top.jpg")

corners = []
frame_copy = None
current_frame = None

CORNER_LABELS = [
    "1: Top-Left",
    "2: Top-Right",
    "3: Bottom-Right",
    "4: Bottom-Left"
]

COLORS = {
    "cyan": (212, 182, 6),
    "green": (94, 197, 34),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "dark_bg": (20, 20, 30),
}

def mouse_callback(event, x, y, flags, param):
    global corners, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append([x, y])
            redraw_overlay()

def redraw_overlay():
    global frame_copy
    if current_frame is None:
        return

    frame_copy = current_frame.copy()
    h, w = frame_copy.shape[:2]

    cv2.rectangle(frame_copy, (0, 0), (w, 50), COLORS["dark_bg"], -1)

    if len(corners) < 4:
        step = len(corners)
        instruction = f"Click corner {CORNER_LABELS[step]}   |   R=Reset  Q=Quit"
        cv2.putText(frame_copy, instruction, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["cyan"], 2, cv2.LINE_AA)
    else:
        instruction = "4 corners marked!  S=Save  R=Reset  Q=Quit"
        cv2.putText(frame_copy, instruction, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["green"], 2, cv2.LINE_AA)

    if len(corners) == 0:
        return

    pts = np.array(corners, dtype=np.int32)

    for i in range(len(corners)):
        if i + 1 < len(corners):
            cv2.line(frame_copy, tuple(pts[i]), tuple(pts[i + 1]), COLORS["cyan"], 2, cv2.LINE_AA)

    if len(corners) == 4:
        cv2.line(frame_copy, tuple(pts[3]), tuple(pts[0]), COLORS["cyan"], 2, cv2.LINE_AA)
        overlay = frame_copy.copy()
        cv2.fillPoly(overlay, [pts], (212, 182, 6))
        cv2.addWeighted(overlay, 0.15, frame_copy, 0.85, 0, frame_copy)
        cv2.line(frame_copy, tuple(pts[0]), tuple(pts[1]), COLORS["green"], 4, cv2.LINE_AA)

    for i, pt in enumerate(corners):
        cv2.circle(frame_copy, tuple(pt), 12, COLORS["cyan"], 2, cv2.LINE_AA)
        cv2.circle(frame_copy, tuple(pt), 6, COLORS["cyan"], -1, cv2.LINE_AA)
        cv2.putText(frame_copy, str(i + 1), (pt[0] + 16, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["white"], 2, cv2.LINE_AA)
        cv2.putText(frame_copy, CORNER_LABELS[i].split(": ")[1], (pt[0] + 16, pt[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["yellow"], 1, cv2.LINE_AA)

def save_calibration():
    global corners, current_frame
    if len(corners) != 4:
        print("[!] Need exactly 4 corners to save.")
        return False

    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    cv2.imwrite(REFERENCE_IMAGE, current_frame)

    pts = np.array(corners, dtype=np.float32)
    h, w = current_frame.shape[:2]

    calib_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_size": {"width": w, "height": h},
        "side": "TOP",
        "corners": pts.tolist()
    }

    try:
        orb = cv2.ORB_create(nfeatures=1500)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, mask)
        calib_data["feature_count"] = len(kp) if kp else 0
    except Exception as e:
        calib_data["feature_count"] = 0

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    print("=" * 55)
    print("  [OK] Calibration saved successfully!")
    print(f"  Corners: {corners}")
    print(f"  Features: {calib_data['feature_count']}")
    print(f"  File: {CALIBRATION_FILE}")
    print(f"  Reference: {REFERENCE_IMAGE}")
    print("=" * 55)

    return True

def show_existing_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("  [i] Existing calibration found:")
            print(f"      Created: {data.get('created_at', '?')}")
            print(f"      Corners: {data.get('corners', [])}")
            print(f"      Features: {data.get('feature_count', 0)}")
            print("      (Will be overwritten if you save new calibration)\n")
        except Exception:
            pass

def try_open_camera(preferred_index=0):
    for idx in [preferred_index, 0, 1, 2]:
        for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, api)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap, idx
                    cap.release()
            except Exception:
                pass
    return None, -1

def pick_image_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="เลือกภาพ PCB สำหรับ Calibrate",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        root.destroy()
        if file_path and os.path.exists(file_path):
            return file_path
    except Exception:
        pass

    if os.path.exists(REFERENCE_IMAGE):
        return REFERENCE_IMAGE
    return None

def main():
    global corners, frame_copy, current_frame

    parser = argparse.ArgumentParser(description="PCB Calibration Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for calibration")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  PCB CALIBRATION TOOL")
    print("  Click 4 corners of the PCB board")
    print("=" * 55 + "\n")

    show_existing_calibration()

    cap = None
    is_image_mode = False

    if args.image and os.path.exists(args.image):
        print(f"  [*] Loading calibration image: {args.image}")
        current_frame = cv2.imread(args.image)
        if current_frame is not None:
            is_image_mode = True

    if not is_image_mode:
        print(f"  [*] Searching for connected camera (Index {args.camera})...")
        cap, actual_idx = try_open_camera(args.camera)
        if cap is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [OK] Camera #{actual_idx} opened: {actual_w}x{actual_h}\n")
        else:
            print("\n  [!] ไม่พบสัญญาณภาพจากกล้อง (Camera Not Found / Disconnected)")
            print("      สาเหตุที่เป็นไปได้:")
            print("      1. ยังไม่ได้เสียบสาย USB ของกล้องเข้ากับคอมพิวเตอร์")
            print("      2. กล้องถูกปิดใช้งาน (Disabled) ใน Windows Device Manager")
            print("      3. มีโปรแกรมอื่นกำลังเปิดใช้งานกล้องอยู่\n")

            if os.path.exists(REFERENCE_IMAGE):
                print(f"  [*] ใช้รูปภาพอ้างอิงเดิม ({REFERENCE_IMAGE}) เพื่อมาร์กจุดใหม่:")
                current_frame = cv2.imread(REFERENCE_IMAGE)
                if current_frame is not None:
                    is_image_mode = True

            if not is_image_mode:
                print("  [*] เปิดหน้าต่างเลือกไฟล์รูปภาพ...")
                img_path = pick_image_file()
                if img_path and os.path.exists(img_path):
                    current_frame = cv2.imread(img_path)
                    if current_frame is not None:
                        is_image_mode = True

            if not is_image_mode:
                print("  [ERROR] ไม่สามารถเริ่ม Calibrate ได้ (ไม่มีทั้งกล้องและไฟล์ภาพ)")
                sys.exit(1)

    print("  Controls:")
    print("    Mouse Click = Mark corner point (1:บนซ้าย, 2:บนขวา, 3:ล่างขวา, 4:ล่างซ้าย)")
    print("    R = Reset all points")
    print("    S = Save calibration (after 4 points)")
    print("    Q / ESC = Quit\n")

    window_name = "PCB Calibration - Click 4 Corners"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        if not is_image_mode and cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            current_frame = frame.copy()

        redraw_overlay()

        if frame_copy is not None:
            cv2.imshow(window_name, frame_copy)
        elif current_frame is not None:
            cv2.imshow(window_name, current_frame)

        key = cv2.waitKey(20 if is_image_mode else 1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("  [*] Exiting calibration tool.")
            break

        if key in (ord('r'), ord('R')):
            corners = []
            print("  [*] Corners reset. Click 4 corners again.")
            redraw_overlay()

        if key in (ord('s'), ord('S')):
            if len(corners) == 4:
                saved = save_calibration()
                if saved:
                    success_frame = current_frame.copy()
                    h, w = success_frame.shape[:2]
                    cv2.rectangle(success_frame, (0, 0), (w, 60), (20, 80, 20), -1)
                    cv2.putText(success_frame, "SAVED! Press Q to quit or R to recalibrate",
                                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (100, 255, 100), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, success_frame)
                    cv2.waitKey(1500)
            else:
                print(f"  [!] Need 4 corners. Currently have {len(corners)}.")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("  [*] Done.")

if __name__ == "__main__":
    main()
