import cv2
import os
import numpy as np
import time
import csv

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# ==========================================
# CẤU HÌNH LOGIC (ĐÃ ĐIỀU CHỈNH)
# ==========================================
# Sau khi sửa lỗi Double Softmax, điểm số sẽ chuẩn hơn.
# Đặt 0.7 là mức an toàn. Nếu > 0.7 thì coi là Real.
REAL_CONFIDENCE_THRESHOLD = 0.70  

# Nếu > 25% số frame bị nghi ngờ (điểm thấp) -> FAKE
# Tăng nhẹ lên 0.25 để tránh bắt nhầm video Real bị rung lắc
FAKE_RATIO_THRESHOLD = 0.25       

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize_image(image):
    height, width, _ = image.shape
    target_width = int(height * (3 / 4))
    if width != target_width:
        image = cv2.resize(image, (target_width, height))
    return image

def ensure_even_dimensions(frame):
    height, width, _ = frame.shape
    if height % 2 != 0 or width % 2 != 0:
        frame = cv2.resize(frame, (width // 2 * 2, height // 2 * 2))
    return frame

def load_model_configs(model_dir):
    configs = []
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        return []
        
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        configs.append({
            "name": model_name,
            "path": os.path.join(model_dir, model_name),
            "h": h_input,
            "w": w_input,
            "scale": scale,
            "param_crop": True if scale is not None else False
        })
    return configs

# ==========================================
# CORE PROCESSING FUNCTION
# ==========================================
def process_video(video_path, model_configs, model_test, image_cropper, device_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": "Cannot open video"}

    total_frames_processed = 0
    suspicious_frames_count = 0 
    avg_real_score = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip bớt frame để tăng tốc
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 != 0:
            continue

        try:
            frame = normalize_image(frame)
            frame = ensure_even_dimensions(frame)
            image_bbox = model_test.get_bbox(frame)
            
            if image_bbox is None or image_bbox[2] == 0: 
                continue

            prediction = np.zeros((1, 3))
            
            for config in model_configs:
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": config["scale"],
                    "out_w": config["w"],
                    "out_h": config["h"],
                    "crop": config["param_crop"],
                }
                img = image_cropper.crop(**param)
                # Model gốc đã trả về Softmax rồi, KHÔNG cần làm lại
                prediction += model_test.predict(img, config["path"])

            # Lấy trung bình cộng
            prediction = prediction / len(model_configs)
            
            # --- FIX: LẤY TRỰC TIẾP GIÁ TRỊ ---
            # Class 1 là Real. 
            real_prob = prediction[0][1] 
            
            avg_real_score += real_prob
            total_frames_processed += 1

            # Logic đánh giá
            if real_prob < REAL_CONFIDENCE_THRESHOLD:
                suspicious_frames_count += 1
            
        except Exception:
            pass

    cap.release()
    
    if total_frames_processed == 0:
        return None, {"error": "No face detected"}

    fake_ratio = suspicious_frames_count / total_frames_processed
    avg_real_score /= total_frames_processed

    # Quyết định
    is_spoof_detected = fake_ratio > FAKE_RATIO_THRESHOLD

    return is_spoof_detected, {
        "fake_ratio": fake_ratio,
        "avg_real_score": avg_real_score
    }

# ==========================================
# BATCH RUNNER
# ==========================================
def run_batch_evaluation_final(csv_path, video_dir, model_dir, device_id, limit=100):
    print(f"--- Final Debug Run (Limit: {limit}) ---")
    print(f"Thresholds: Real_Score > {REAL_CONFIDENCE_THRESHOLD} | Fake_Ratio > {FAKE_RATIO_THRESHOLD}")
    
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    model_configs = load_model_configs(model_dir)

    if not model_configs: return

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) 
            all_rows = list(reader)
    except FileNotFoundError:
        return

    count = 0
    correct_count = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    print(f"\n{'IDX':<5} | {'VIDEO':<10} | {'GT':<5} | {'PRED':<5} | {'RESULT':<8} | {'INFO (Ratio | AvgReal)'}")
    print("-" * 75)

    for row in all_rows:
        if count >= limit: break
            
        fname = row[0]
        # CSV: 1=Real, 0=Fake
        is_actual_fake = (int(row[1]) == 0)
        gt_str = "FAKE" if is_actual_fake else "REAL"

        video_path = os.path.join(video_dir, fname)
        if not os.path.exists(video_path): continue
        
        is_predicted_fake, details = process_video(
            video_path, model_configs, model_test, image_cropper, device_id
        )

        if is_predicted_fake is None:
            print(f"{count+1:<5} | {fname:<10} | {gt_str:<5} | ERROR | {details['error']}")
            continue

        pred_str = "FAKE" if is_predicted_fake else "REAL"
        
        if is_predicted_fake == is_actual_fake:
            res_str = "CORRECT"
            correct_count += 1
            if is_actual_fake: tp += 1
            else: tn += 1
        else:
            res_str = "WRONG"
            if is_predicted_fake: fp += 1 
            else: fn += 1 

        # Highlight dòng WRONG bằng dấu *
        marker = "<<" if res_str == "WRONG" else ""
        print(f"{count+1:<5} | {fname:<10} | {gt_str:<5} | {pred_str:<5} | {res_str:<8} | R:{details['fake_ratio']:.2f} | S:{details['avg_real_score']:.2f} {marker}")

        count += 1

    accuracy = (correct_count / count) * 100 if count > 0 else 0
    print("\n" + "="*40)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{count})")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*40)

if __name__ == "__main__":
    ROOT_DIR = "images/train/train"
    CSV_FILE = os.path.join(ROOT_DIR, "label.csv")
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos") 
    MODEL_DIR = "./resources/anti_spoof_models"
    DEVICE_ID = 0

    run_batch_evaluation_final(CSV_FILE, VIDEO_DIR, MODEL_DIR, DEVICE_ID, limit=100)