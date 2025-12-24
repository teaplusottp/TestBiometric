import cv2
import os
import numpy as np
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# --- HELPER FUNCTIONS ---
def normalize_image(image):
    """Resize image to 3:4 aspect ratio to match model training requirements."""
    height, width, _ = image.shape
    target_width = int(height * (3 / 4))
    if width != target_width:
        image = cv2.resize(image, (target_width, height))
    return image

def ensure_even_dimensions(frame):
    """Ensure dimensions are even for video encoding compatibility."""
    height, width, _ = frame.shape
    if height % 2 != 0 or width % 2 != 0:
        frame = cv2.resize(frame, (width // 2 * 2, height // 2 * 2))
    return frame

def load_model_configs(model_dir):
    """
    OPTIMIZATION: Pre-load model configurations once.
    Avoids parsing strings and listing directories inside the video loop.
    """
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

def process_video(video_path, output_path, model_dir, device_id):
    # 1. Initialize Models & Tools
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Pre-load model configs (Performance Boost)
    model_configs = load_model_configs(model_dir)
    if not model_configs:
        return

    # 2. Setup Video Input
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 3. Setup Video Output Dimensions (Based on first frame)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    processed_first = normalize_image(first_frame)
    processed_first = ensure_even_dimensions(processed_first)
    out_h, out_w, _ = processed_first.shape
    
    print(f"Original Size: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"Output Size: {out_w}x{out_h}")

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    except Exception as e:
        print(f"Error: Unable to create VideoWriter. {e}")
        cap.release()
        return

    # Reset pointer to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 4. Processing Loop
    frame_count = 0
    
    # -- Aggregation Logic Variables --
    total_frames = 0
    fake_frame_count = 0
    consecutive_fake = 0
    max_consecutive_fake = 0
    spoof_threshold = 5 # Alert if 5 consecutive frames are fake

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        try:
            # Pre-processing
            frame = normalize_image(frame)
            frame = ensure_even_dimensions(frame)
            
            # Detect face
            image_bbox = model_test.get_bbox(frame)
            
            prediction = np.zeros((1, 3))
            
            # Prediction Loop (Optimized: using pre-loaded configs)
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

            # Post-processing results
            # Note: prediction is sum of 2 models, so divide by 2 for average
            label = np.argmax(prediction)
            value = prediction[0][label] / 2 
            
            # --- AGGREGATION LOGIC ---
            is_fake = (label != 1) # Assuming label 1 is Real, others are Spoof
            
            if is_fake:
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255) # Red
                fake_frame_count += 1
                consecutive_fake += 1
                max_consecutive_fake = max(max_consecutive_fake, consecutive_fake)
            else:
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0) # Blue
                consecutive_fake = 0 # Reset counter

            # Warning text if consecutive fakes detected
            if consecutive_fake >= spoof_threshold:
                 cv2.putText(frame, "WARNING: SPOOF DETECTED!", (50, 50), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Draw UI
            cv2.rectangle(frame, 
                         (image_bbox[0], image_bbox[1]), 
                         (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), 
                         color, 2)
            
            cv2.putText(frame, result_text, 
                       (image_bbox[0], image_bbox[1] - 5), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames... (Current Max Consecutive Fake: {max_consecutive_fake})")

        except Exception as e:
            # In production, you might want to log this properly
            # print(f"Warning: Frame {frame_count} skipped. {e}")
            pass

    # 5. Cleanup & Final Report
    cap.release()
    out.release()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 30)
    print(f"Processing Complete: {output_path}")
    print(f"Time taken: {duration:.2f}s ({total_frames/duration:.2f} fps)")
    print(f"Total Frames: {total_frames}")
    print(f"Fake Frames: {fake_frame_count} ({fake_frame_count/total_frames*100:.1f}%)")
    print(f"Max Consecutive Fake Frames: {max_consecutive_fake}")
    
    # FINAL VERDICT
    if max_consecutive_fake >= spoof_threshold:
        print(">>> FINAL RESULT: FAKE (Spoofing Attack Detected)")
    else:
        print(">>> FINAL RESULT: REAL")
    print("-" * 30)

if __name__ == "__main__":
    video_path = "images/train/train/videos/1.mp4"
    output_path = "images/train/train/videos/1__res.mp4"
    model_dir = "./resources/anti_spoof_models"
    device_id = 0

    if os.path.exists(video_path):
        process_video(video_path, output_path, model_dir, device_id)
    else:
        print(f"Error: Input file not found at {video_path}")