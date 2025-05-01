import time
import torch
import cv2
import numpy as np
import os
from hawp.fsl.model.models import WireframeDetector
import yaml
from easydict import EasyDict as edict
import argparse
# =============================================================================
# Usage:
#   python custom_inference_time.py \
#     --input_dir /path/to/your/images/ \
#     --output_dir /path/to/save/results/ \
#     --threshold 0.7 \
#     --config /path/to/configs/plnet.yaml \
#     --weights /path/to/plnet.pth
#
# Arguments:
#   --input_dir   Directory containing input images (png/jpg/jpeg)
#   --output_dir  Directory to save visualized outputs (will be created if needed)
#   --threshold   Confidence threshold for line drawing (default: 0.5)
#   --config      Path to the plnet.yaml config file
#   --weights     Path to the trained model checkpoint (.pth)
# =============================================================================

def load_cfg(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = edict(yaml.safe_load(f))
    return cfg

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1,1,512,512]

def visualize_results(image, keypoints, lines, scores, threshold=0.5, output_path=None):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for pt in keypoints:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
    for line, score in zip(lines, scores):
        if score > threshold:
            pt1 = (int(line[0]), int(line[1]))
            pt2 = (int(line[2]), int(line[3]))
            cv2.line(vis, pt1, pt2, (0,0,255), 2)
    if output_path:
        cv2.imwrite(output_path, vis)
    return vis

def main():
    parser = argparse.ArgumentParser(description='PLNet Custom Dataset Inference')
    parser.add_argument('--config', type=str, default='configs/plnet.yaml')
    parser.add_argument('--weights', type=str, default='plnet.pth')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WireframeDetector(cfg).to(device).eval()

    ckpt = torch.load(args.weights, map_location=device)
    if 'model' in ckpt:
        sd = ckpt['model']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(args.input_dir) 
                   if f.lower().endswith(('.png','.jpg','.jpeg'))]
    image_files = image_files[:50]  # 첫 50개만 처리

    times = []
    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        img_tensor = preprocess_image(image_path).to(device)

        # 추론 시간 측정 시작
        start = time.time()
        with torch.no_grad():
            outputs, _ = model(img_tensor, annotations=[{
                "filename": image_file, "width":512, "height":512
            }])
        if device.type=='cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{image_file}: inference time = {elapsed:.3f} s")

        # 시각화
        keypoints = outputs['juncs_pred'].cpu().numpy()
        lines     = outputs['lines_pred'].cpu().numpy()
        scores    = outputs['lines_score'].cpu().numpy()
        orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orig = cv2.resize(orig, (512,512))
        out_path = os.path.join(args.output_dir, f"result_{image_file}")
        visualize_results(orig, keypoints, lines, scores, args.threshold, out_path)

    avg_time = sum(times) / len(times)
    print(f"\nProcessed {len(times)} images")
    print(f"Average inference time: {avg_time:.3f} s")

if __name__ == '__main__':
    main()
