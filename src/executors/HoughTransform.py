
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.helper.executor import Executor

try:  # NovaVision's capsule loader uses this qualified package path.
    from capsules.HoughTransform.src.models.PackageModel import (
        HoughDetection,
        PackageModel,
    )
    from capsules.HoughTransform.src.utils.response import build_response
except ModuleNotFoundError:  # Allow execution from an ordinary source installation.
    from src.models.PackageModel import HoughDetection, PackageModel
    from src.utils.response import build_response


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

    def _make_detection(self, class_id, label, bounds, geometry):
        left, top, right, bottom = bounds
        return HoughDetection(
            boundingBox=BoundingBox(
                left=float(left),
                top=float(top),
                width=float(max(1, right - left)),
                height=float(max(1, bottom - top)),
            ),
            confidence=1.0,
            classId=class_id,
            classLabel=label,
            imgUID=self.uID,
            keyPoints=[],
            metadata={"geometry": geometry},
        )

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

                detections.append(
                    self._make_detection(
                        class_id=0,
                        label="line",
                        bounds=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                        geometry={
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    )
                )

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

                left = max(0, x - radius)
                top = max(0, y - radius)
                right = min(width - 1, x + radius)
                bottom = min(height - 1, y + radius)
                detections.append(
                    self._make_detection(
                        class_id=1,
                        label="circle",
                        bounds=(left, top, right, bottom),
                        geometry={
                            "centerX": x,
                            "centerY": y,
                            "radius": radius,
                        },
                    )
                )

        return output, detections

    def run(self):
        if self.task not in ("LineDetection", "CircleDetection"):
            raise ValueError(f"Unsupported Hough task: {self.task}")

        input_is_list = isinstance(self.image, list)
        input_images = self.image if input_is_list else [self.image]
        if not input_images or any(image is None for image in input_images):
            raise ValueError("At least one input image is required.")
        output_images = []
        self.detections = []

        for input_image in input_images:
            img = Image.get_frame(img=input_image, redis_db=self.redis_db)
            if self.task == "LineDetection":
                output, detections = self.detect_lines(img.value)
            else:
                output, detections = self.detect_circles(img.value)
            self.detections.extend(detections)
            img.value = output
            output_images.append(
                Image.set_frame(
                    img=img,
                    package_uID=self.uID,
                    redis_db=self.redis_db,
                )
            )

        self.image = output_images if input_is_list else output_images[0]

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
