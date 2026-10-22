
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor

from capsules.HoughTransform.src.models.PackageModel import PackageModel
from capsules.HoughTransform.src.utils.response import build_response


class HoughTransform(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.image = self.request.get_param("inputImage")
        self.task = self.request.get_param("Task")

        self.canny_low = self.request.get_param("CannyLow")
        self.canny_high = self.request.get_param("CannyHigh")
        self.hough_threshold = self.request.get_param("HoughThreshold")
        self.min_line_length = self.request.get_param("MinLineLength")
        self.max_line_gap = self.request.get_param("MaxLineGap")

        self.dp = self.request.get_param("DP")
        self.min_dist = self.request.get_param("MinDist")
        self.param1 = self.request.get_param("Param1")
        self.param2 = self.request.get_param("Param2")
        self.min_radius = self.request.get_param("MinRadius")
        self.max_radius = self.request.get_param("MaxRadius")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    @staticmethod
    def _to_gray(frame):
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input image is empty or invalid.")

        if frame.dtype != np.uint8:
            raise ValueError("Input image must use uint8 pixel values.")

        if frame.ndim == 2:
            return frame

        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        raise ValueError("Expected a grayscale, BGR, or BGRA image.")

    def detect_lines(self, frame):
        gray = self._to_gray(frame)

        if not (0 <= self.canny_low < self.canny_high <= 255):
            raise ValueError("CannyLow must be less than CannyHigh.")

        if self.hough_threshold <= 0:
            raise ValueError("HoughThreshold must be positive.")

        if self.min_line_length <= 0 or self.max_line_gap < 0:
            raise ValueError("Invalid line length or gap.")

        edges = cv2.Canny(
            gray,
            self.canny_low,
            self.canny_high,
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        output = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2
            else frame[:, :, :3].copy()
        )

        detections = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])

                cv2.line(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )

                detections.append({
                    "classId": 0,
                    "classLabel": "line",
                    "bbox": [
                        min(x1, x2),
                        min(y1, y2),
                        max(x1, x2),
                        max(y1, y2),
                    ],
                    "geometry": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                })

        return output, detections

    def detect_circles(self, frame):
        gray = self._to_gray(frame)

        if min(
            self.dp,
            self.min_dist,
            self.param1,
            self.param2,
        ) <= 0:
            raise ValueError(
                "DP, MinDist, Param1, and Param2 must be positive."
            )

        if self.min_radius < 0 or self.max_radius < 0:
            raise ValueError("Circle radii cannot be negative.")

        if self.max_radius and self.max_radius < self.min_radius:
            raise ValueError("MaxRadius must be >= MinRadius.")

        blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        output = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2
            else frame[:, :, :3].copy()
        )

        detections = []

        if circles is not None:
            height, width = gray.shape

            for circle in np.round(circles[0]).astype(int):
                x, y, radius = map(int, circle)

                cv2.circle(
                    output,
                    (x, y),
                    radius,
                    (0, 255, 0),
                    2,
                )

                detections.append({
                    "classId": 1,
                    "classLabel": "circle",
                    "bbox": [
                        max(0, x - radius),
                        max(0, y - radius),
                        min(width - 1, x + radius),
                        min(height - 1, y + radius),
                    ],
                    "geometry": {
                        "centerX": x,
                        "centerY": y,
                        "radius": radius,
                    },
                })

        return output, detections

    def run(self):
        img = Image.get_frame(
            img=self.image,
            redis_db=self.redis_db,
        )

        if self.task == "LineDetection":
            output, self.detect = self.detect_lines(img.value)

        elif self.task == "CircleDetection":
            output, self.detect = self.detect_circles(img.value)

        else:
            raise ValueError(f"Unsupported Hough task: {self.task}")

        img.value = output

        self.image = Image.set_frame(
            img=img,
            package_uID=self.uID,
            redis_db=self.redis_db,
        )

        self.image = self.image
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
