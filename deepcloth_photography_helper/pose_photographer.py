from pathlib import Path
import os
import time
from datetime import datetime

import cv2
import numpy as np

from deepcloth_photography_helper.lightweight_human_pose_estimation import PoseEstimator
from deepcloth_photography_helper.utils import load_images
from deepcloth_photography_helper.constans import (
    SEGMENTATION_IMAGE_NAME,
    POSE_POINTS_IMAGE_NAME,
)


class PosePhotographer:
    BASE_DIR = Path(__file__).parent.parent
    CAMERA_INDEX = 0
    POSE_IMAGES_DIR_PATH = BASE_DIR.joinpath("pose_images").as_posix()
    PAUSE_TIME_LENGHT_IN_SECONDS = 20

    def __init__(self) -> None:
        self.camera = cv2.VideoCapture(self.CAMERA_INDEX)
        self.pose_image_paths = load_images(self.POSE_IMAGES_DIR_PATH)
        self.pose_series = self.open_series()
        self.key_point_predictor = PoseEstimator()

    def open_series(self) -> list:
        series = [p for p in Path(self.POSE_IMAGES_DIR_PATH).iterdir() if p.is_dir()]
        series = [p for p in series if any(p.iterdir())]
        return series

    def get_newes_series(self) -> list:
        return max(self.pose_series, key=os.path.getmtime)

    def load_series(self, series_path: Path) -> list:
        image_dirs = [p for p in Path(series_path).iterdir() if p.is_dir()]
        images_data = []
        for image_dir in image_dirs:
            image_data = {}
            images_paths = image_dir.glob("*.png")
            for images_path in images_paths:
                # pose image
                if images_path.stem.startswith(POSE_POINTS_IMAGE_NAME):
                    image_data[POSE_POINTS_IMAGE_NAME] = cv2.imread(
                        images_path.as_posix()
                    )
                # sagemntion
                elif images_path.stem.startswith(SEGMENTATION_IMAGE_NAME):
                    seg_image = cv2.imread(images_path.as_posix(), 0)
                    _, thresh = cv2.threshold(seg_image, 10, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(
                        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
                    )
                    cv2.drawContours(
                        image=np.zeros_like(seg_image),
                        contours=contours,
                        contourIdx=-1,
                        color=255,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
                    image_data[SEGMENTATION_IMAGE_NAME] = seg_image
                # normal image
                else:
                    image_data["image"] = cv2.imread(images_path.as_posix())
            images_data.append(image_data)
        return images_data

    def open_preview(self) -> None:
        now = datetime.now()
        now = now.strftime("%S_%M_%H___%d_%m_%Y")
        new_series_path = Path(self.POSE_IMAGES_DIR_PATH).joinpath(now)
        new_series_path.mkdir(exist_ok=True, parents=True)
        radius = 3
        thickness = -1
        font = cv2.FONT_HERSHEY_PLAIN
        serie_path = self.get_newes_series()
        series = self.load_series(serie_path)
        _, frame = self.camera.read()
        height, width, _ = frame.shape
        for serie_idx, serie in enumerate(series):
            start_time = time.time()
            end_time = start_time + self.PAUSE_TIME_LENGHT_IN_SECONDS
            image = serie["image"]
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
            target_pose_points = serie[POSE_POINTS_IMAGE_NAME]
            target_pose_points = cv2.resize(
                target_pose_points, (width, height), cv2.INTER_AREA
            )
            target_mask = serie[SEGMENTATION_IMAGE_NAME]
            target_mask = cv2.resize(target_mask, (width, height), cv2.INTER_AREA)
            while time.time() < end_time:
                _, frame = self.camera.read()
                agumented_frame = frame.copy()
                # add time to frame
                frame_time = int(end_time - time.time())
                cv2.putText(
                    agumented_frame,
                    str(frame_time),
                    (20, 40),
                    font,
                    2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                # add points to frame
                # real time
                pose_points = self.key_point_predictor.predict_numpy(frame)
                for pose_point in pose_points:
                    if pose_point[0] != 0 and pose_point[1] != 0:
                        agumented_frame = cv2.circle(
                            agumented_frame,
                            (pose_point[0], pose_point[1]),
                            radius,
                            (0, 255, 0),
                            thickness,
                        )
                # targegs poits
                agumented_frame = cv2.addWeighted(
                    agumented_frame, 0.8, target_pose_points, 0.2, 0.0
                )
                # add target obrys
                agumented_frame = cv2.addWeighted(
                    agumented_frame, 0.8, target_mask, 0.2, 0.0
                )
                # add image to frame
                agumented_frame = cv2.hconcat([agumented_frame, image])
                if int(end_time - time.time()) < self.PAUSE_TIME_LENGHT_IN_SECONDS // 2:
                    image_path = new_series_path.joinpath(
                        f"{serie_idx}_{int(end_time-time.time())}.png"
                    ).as_posix()
                    cv2.imwrite(image_path, frame)
                # agumented_frame = frame
                cv2.imshow("agumented_frame", agumented_frame)
                if cv2.waitKey(1) and 0xFF == ord("q"):
                    break
            else:
                pass
        # cv2.destroyAllWindows()
        print(f"Prosze wkleić zdjecie ubrania do folderu {str(new_series_path)}")
        print("Zdjecie topu ma być na biłym tle najlepiej bez zagiec.")
        # return series

    def max_seconds(self, max_seconds, *, interval=1):
        interval = int(interval)
        start_time = time.time()
        end_time = start_time + max_seconds
        yield 0
        while time.time() < end_time:
            if interval > 0:
                next_time = start_time
                while next_time < time.time():
                    next_time += interval
                # time.sleep(int(round(next_time - time.time())))
            yield int(round(time.time() - start_time))
            if int(round(time.time() + interval)) > int(round(end_time)):
                return
