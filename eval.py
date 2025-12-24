import cv2
import os
import numpy as np
import time
import csv  # Thêm thư viện đọc CSV

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# --- HELPER FUNCTIONS ---
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

# --- MODIFIED PROCESS FUNCTION ---
def process_video(video_path, output_path, model_configs, model_test, image_cropper, device_id, silent=True):
    """
    Trả về:
    - is_spoof (bool): True nếu phát hiện giả mạo, False nếu là thật
    - message (str): Thông báo lỗi nếu có
    """
    
    # Setup Video Input
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Error: Unable to open video {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Setup Video Output Dimensions
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None, "Error: Cannot read the first frame."

    processed_first = normalize_image(first_frame)
    processed_first = ensure_even_dimensions(processed_first)
    out_h, out_w, _ = processed_first.shape

    # Tắt việc ghi video output nếu chỉ muốn test nhanh (bỏ comment dòng dưới nếu muốn ghi file)
    out = None 
    # try:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    # except Exception:
    #     pass

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -- Logic Variables --
    fake_frame_count = 0
    consecutive_fake = 0
    max_consecutive_fake = 0
    spoof_threshold = 5 # Alert if 5 consecutive frames are fake

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            frame = normalize_image(frame)
            frame = ensure_even_dimensions(frame)
            image_bbox = model_test.get_bbox(frame)
            
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
                prediction += model_test.predict(img, config["path"])

            label = np.argmax(prediction)
            
            # Label 1 is Real, others are Spoof
            is_fake = (label != 1) 
            
            if is_fake:
                consecutive_fake += 1
                max_consecutive_fake = max(max_consecutive_fake, consecutive_fake)
            else:
                consecutive_fake = 0 # Reset counter

            # Nếu muốn ghi video output thì uncomment đoạn vẽ UI và out.write(frame) ở đây
            # ... (code vẽ UI cũ) ...
            # if out: out.write(frame)

        except Exception as e:
            pass

    cap.release()
    if out: out.release()
    
    # FINAL VERDICT
    is_spoof_detected = max_consecutive_fake >= spoof_threshold
    return is_spoof_detected, "Success"

# --- BATCH EVALUATION FUNCTION ---
def run_batch_evaluation(csv_path, video_dir, output_root, model_dir, device_id, limit=100):
    print(f"--- Starting Batch Evaluation (Limit: {limit} videos) ---")
    
    # 1. Load Models Once (Optimization)
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    model_configs = load_model_configs(model_dir)

    if not model_configs:
        return

    # 2. Read CSV
    results = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            all_rows = list(reader)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 3. Process Loop
    count = 0
    correct_count = 0
    
    # Statistics
    tp = 0 # True Positive (Fake detected as Fake)
    tn = 0 # True Negative (Real detected as Real)
    fp = 0 # False Positive (Real detected as Fake)
    fn = 0 # False Negative (Fake detected as Real)

    start_total_time = time.time()

    for row in all_rows:
        if count >= limit:
            break
            
        fname = row[0]
        # Label trong CSV: 1 là Real (thật), 0 là Fake (giả)
        ground_truth_score = int(row[1]) 
        is_actual_fake = (ground_truth_score == 0)

        video_path = os.path.join(video_dir, fname)
        output_path = os.path.join(output_root, fname)

        if not os.path.exists(video_path):
            print(f"Skipping {fname}: File not found.")
            continue

        print(f"[{count+1}/{limit}] Processing: {fname}...", end=" ", flush=True)
        
        # Run prediction
        is_predicted_fake, msg = process_video(
            video_path, output_path, model_configs, model_test, image_cropper, device_id, silent=True
        )

        if is_predicted_fake is None:
            print(f"Failed: {msg}")
            continue

        # Evaluate
        if is_predicted_fake == is_actual_fake:
            print("CORRECT")
            correct_count += 1
            if is_actual_fake: tp += 1
            else: tn += 1
        else:
            print(f"WRONG (Pred: {'Fake' if is_predicted_fake else 'Real'} | GT: {'Fake' if is_actual_fake else 'Real'})")
            if is_predicted_fake: fp += 1 # Đoán Fake nhưng thật ra là Real
            else: fn += 1 # Đoán Real nhưng thật ra là Fake
            
        count += 1

    # 4. Final Report
    total_time = time.time() - start_total_time
    accuracy = (correct_count / count) * 100 if count > 0 else 0
    
    print("\n" + "="*40)
    print(f"EVALUATION REPORT (Sample Size: {count})")
    print(f"Total Time: {total_time:.2f}s")
    print("="*40)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{count})")
    print(f"True Positives (Fake caught): {tp}")
    print(f"True Negatives (Real passed): {tn}")
    print(f"False Positives (Real blocked): {fp}")
    print(f"False Negatives (Fake passed): {fn}")
    print("="*40)

if __name__ == "__main__":
    # Cấu hình đường dẫn
    ROOT_DIR = "images/train/train"
    CSV_FILE = os.path.join(ROOT_DIR, "label.csv")
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos") # Giả sử video nằm trong folder 'videos'
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output_results")
    
    MODEL_DIR = "./resources/anti_spoof_models"
    DEVICE_ID = 0

    run_batch_evaluation(CSV_FILE, VIDEO_DIR, OUTPUT_DIR, MODEL_DIR, DEVICE_ID, limit=100)