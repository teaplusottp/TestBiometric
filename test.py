# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"
RESULTS_PATH = os.path.join(SAMPLE_IMAGE_PATH, "results/")

# Ensure the results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)


# Automatically resize image to 3:4 aspect ratio
def normalize_image(image):
    height, width, _ = image.shape
    target_width = int(height * (3 / 4))
    if width != target_width:
        print("Resizing image to 3:4 aspect ratio.")
        image = cv2.resize(image, (target_width, height))
    return image


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    image = normalize_image(image)
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    result_image_path = os.path.join(RESULTS_PATH, result_image_name)
    cv2.imwrite(result_image_path, image)


def test_all_images(model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    for image_name in os.listdir(SAMPLE_IMAGE_PATH):
        if not image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
        image = normalize_image(image)
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0

        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time() - start

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)

        format_ = os.path.splitext(image_name)[-1]
        result_image_name = image_name.replace(format_, "_result" + format_)
        result_image_path = os.path.join(RESULTS_PATH, result_image_name)
        cv2.imwrite(result_image_path, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="image used to test")
    args = parser.parse_args()

    if args.image_name:
        test(args.image_name, args.model_dir, args.device_id)
    else:
        test_all_images(args.model_dir, args.device_id)