"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
"""
import os
import cv2
import numpy as np
import torch
import time
from PIL import Image
from pathlib import Path

from .models.with_mobilenet import PoseEstimationWithMobileNet
from .modules.keypoints import extract_keypoints, group_keypoints
from .modules.load_state import load_state
from .modules.pose import Pose, track_poses
from .val import normalize, pad_width


class PoseEstimator(object):
    def __init__(
        self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.device = device
        self.module_path = os.path.abspath(os.getcwd())
        self.save_model_path = str(
            Path(__file__).parent / "models/checkpoint_iter_370000.pth"
        )
        self.model = PoseEstimationWithMobileNet()
        checkpoint = torch.load(self.save_model_path, map_location=self.device)
        load_state(self.model, checkpoint)
        self.model.to(self.device)
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.height_size = 256

    def predict_path(self, image_path):
        img = cv2.imread(image_path)
        return self.predict_numpy(img=img)

    def predict_numpy(self, img):
        output = np.zeros((self.num_keypoints, 2), dtype=np.int32)
        heatmaps, pafs, scale, pad = self._infer_fast(img=img)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )
        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs, demo=True
        )
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]
            ) / scale
        pose_keypoints = np.zeros((self.num_keypoints, 3), dtype=np.int32)
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != 0.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    )
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    )
                    pose_keypoints[kpt_id, 2] = 1.0
        pose_keypoints = pose_keypoints.reshape(-1)
        keypoint_dict = {"people": [{"pose_keypoints": pose_keypoints.tolist()}]}
        return keypoint_dict

    def _infer_fast(
        self, img, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256
    ):
        height, width, _ = img.shape
        scale = self.height_size / height

        scaled_img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [self.height_size, max(scaled_img.shape[1], self.height_size)]
        padded_img, pad = pad_width(scaled_img, self.stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

        tensor_img = tensor_img.to(self.device)

        stages_output = self.model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(
            heatmaps,
            (0, 0),
            fx=self.upsample_ratio,
            fy=self.upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(
            pafs,
            (0, 0),
            fx=self.upsample_ratio,
            fy=self.upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        return heatmaps, pafs, scale, pad
