from pathlib import Path
from pprint import pprint

import cv2
import numpy as np

from deepcloth_photography_helper.pose_photographer import PosePhotographer
from deepcloth_photography_helper.utils import create_image_cases

if __name__ == "__main__":
    # create_image_cases("production_utils/deepcloth_photography_helper/pose_images")
    pose_photographer = PosePhotographer()
    # frame = pose_photographer.create_first_image()
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    pose_photographer.open_preview()

    # camera = cv2.VideoCapture(0)
    # while True:
    #     _, frame = camera.read()
    #     # agumented_frame = frame
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) and 0xFF == ord('q'):
    #         break
