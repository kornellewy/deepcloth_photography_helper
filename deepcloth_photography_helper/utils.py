from typing import List
from os import listdir, path
from pathlib import Path
from shutil import copy2
import json
import cv2
import numpy as np

from deepcloth_photography_helper.full_body_segmentaion import FullBodySegmentaion
from deepcloth_photography_helper.lightweight_human_pose_estimation import PoseEstimator
from deepcloth_photography_helper.constans import (
    SEGMENTATION_IMAGE_NAME,
    POSE_POINTS_IMAGE_NAME,
)


def load_images(dir_path: str) -> List[str]:
    images = []
    valid_images = [".jpeg", ".jpg", ".png"]
    for f in listdir(dir_path):
        ext = path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(path.join(dir_path, f))
    return images


def create_image_cases(dir_path):
    radius = 3
    color = (0, 0, 255)
    thickness = -1
    body_segmenter = FullBodySegmentaion(output_shape=None)
    key_point_predictor = PoseEstimator()
    # image_cases = []
    image_paths = load_images(dir_path)
    dir_path = Path(dir_path).joinpath("start")
    dir_path.mkdir(exist_ok=True, parents=True)
    for image_path in image_paths:
        image_path = Path(image_path)
        case_name = image_path.stem
        case_path = dir_path.joinpath(case_name)
        case_path.mkdir(exist_ok=True, parents=True)
        new_image_path = case_path.joinpath(image_path.name)
        copy2(image_path, new_image_path)
        # create segmentaion image
        segmentaion_path = case_path.joinpath(
            f"{SEGMENTATION_IMAGE_NAME}_{image_path.name}"
        ).as_posix()
        segmentaion_image = body_segmenter.predict_path(new_image_path)
        image = cv2.imread(image_path.as_posix())
        height, width, _ = image.shape
        segmentaion_image = cv2.resize(
            segmentaion_image, (width, height), cv2.INTER_AREA
        )
        cv2.imwrite(segmentaion_path, segmentaion_image)
        # pose points
        pose_points_path = case_path.joinpath(
            f"{POSE_POINTS_IMAGE_NAME}_{image_path.name}"
        ).as_posix()
        pose_points = key_point_predictor.predict_path(new_image_path.as_posix())
        empty_image = np.zeros_like(image)
        for pose_point in pose_points:
            if pose_point[0] != 0 and pose_point[1] != 0:
                empty_image = cv2.circle(
                    empty_image,
                    (pose_point[0], pose_point[1]),
                    radius,
                    color,
                    thickness,
                )
        cv2.imwrite(pose_points_path, empty_image)
