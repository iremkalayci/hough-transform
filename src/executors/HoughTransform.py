
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox

from capsules.HoughTransform.src.models.PackageModel import Detection, PackageModel
from capsules.HoughTransform.src.utils.response import build_response
from capsules.HoughTransform.src.utils.line_ops import (
    clip_segments,
    filter_by_angle,
    filter_circles_by_edge_support,
    merge_segments,
    road_roi_polygon,
)


MAX_OUTPUT_LINES = 25


class HoughTransform(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.image = self.request.get_param("inputImage")
        self.detections = []

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

        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]

        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        raise ValueError("Expected a grayscale, BGR, or BGRA image.")

    @staticmethod
    def _prepare_frame(frame):
        """Convert supported NovaVision numeric frames to OpenCV's uint8 format."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input image is empty or invalid.")
        if frame.dtype == np.uint8:
            return frame, False

        if np.issubdtype(frame.dtype, np.floating):
            if not np.all(np.isfinite(frame)):
                raise ValueError("Input image contains NaN or infinite pixel values.")
            minimum = float(np.min(frame))
            maximum = float(np.max(frame))
            if minimum < 0 or maximum > 255:
                raise ValueError(
                    "Floating-point input pixels must be in [0, 1] or [0, 255]; "
                    "received range [{}, {}].".format(minimum, maximum)
                )
            normalized = maximum <= 1.0
            scale = 255.0 if normalized else 1.0
            frame_u8 = np.rint(frame * scale).astype(np.uint8)
            return frame_u8, normalized

        if np.issubdtype(frame.dtype, np.integer):
            minimum = int(np.min(frame))
            maximum = int(np.max(frame))
            if minimum < 0 or maximum > 255:
                raise ValueError(
                    "Integer input pixels must be in [0, 255]; received range "
                    "[{}, {}].".format(minimum, maximum)
                )
            return frame.astype(np.uint8), False

        raise ValueError(
            "Unsupported input image dtype '{}'; expected uint8 or floating-point "
            "pixels in [0, 1] or [0, 255].".format(frame.dtype)
        )

    @staticmethod
    def _restore_frame_dtype(output, source_frame, normalized):
        """Return annotations in the input frame's dtype and numeric range."""
        if source_frame.dtype == np.uint8:
            return output
        restored = output.astype(source_frame.dtype)
        if normalized:
            restored /= 255.0
        return restored

    @staticmethod
    def _to_color(frame):
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        return frame[:, :, :3].copy()

    def _make_detection(self, class_id, label, bounds):
        left, top, right, bottom = bounds
        return Detection(
            boundingBox=BoundingBox(
                left=float(left),
                top=float(top),
                width=float(max(1, right - left)),
                height=float(max(1, bottom - top)),
            ),
            # Hough's accumulator threshold is not a calibrated confidence.
            confidence=None,
            classId=class_id,
            classLabel=label,
            imgUID=self._current_image_uid,
        )

    @staticmethod
    def _image_uid(processed_image, source_image):
        """Read the image UID without confusing it with Capsule.uID."""
        for image in (processed_image, source_image):
            if isinstance(image, dict):
                image_uid = image.get("uID")
            else:
                image_uid = getattr(image, "uID", None)
            if image_uid is not None:
                return image_uid
        raise ValueError(
            "The processed image does not expose its NovaVision uID; "
            "cannot create correctly associated detections."
        )

    def detect_lines(self, frame):
        gray = self._to_gray(frame)
        self._validate_line_params()
        min_dimension = min(gray.shape)
        min_line_length = max(self.min_line_length, int(round(min_dimension * 0.20)))
        max_line_gap = max(self.max_line_gap, int(round(min_dimension * 0.04)))
        merge_distance = max(self.merge_distance, min_dimension * 0.02)

        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            smoothed,
            self.canny_low,
            self.canny_high,
        )

        # Keep the road-focused ROI internal so users do not need to tune a mask.
        roi = np.zeros(edges.shape, dtype=np.uint8)
        cv2.fillPoly(roi, [road_roi_polygon(gray.shape)], 255)
        edges = cv2.bitwise_and(edges, roi)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        output = self._to_color(frame)

        detections = []

        if lines is not None:
            segments = np.asarray(lines).reshape(-1, 4)
            segments = filter_by_angle(segments)
            segments = merge_segments(
                segments,
                max_gap=max_line_gap,
                merge_distance=merge_distance,
            )
            segments = filter_by_angle(segments)
            # Avoid overwhelming the output with low-value fragments on dense textures.
            segments = segments[:MAX_OUTPUT_LINES]
            height, width = gray.shape
            segments = clip_segments(segments, width=width, height=height)
            for line in segments:
                x1, y1, x2, y2 = map(int, line)

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
                    )
                )

        return output, detections

    def _validate_line_params(self):
        if not (0 <= self.canny_low < self.canny_high <= 255):
            raise ValueError("CannyLow must be less than CannyHigh.")
        if self.hough_threshold <= 0:
            raise ValueError("HoughThreshold must be positive.")
        if self.min_line_length <= 0 or self.max_line_gap < 0:
            raise ValueError("Invalid line length or gap.")

    def detect_circles(self, frame):
        gray = self._to_gray(frame)

        if min(
            self.dp,
            self.param1,
            self.param2,
        ) <= 0:
            raise ValueError(
                "DP, Param1, and Param2 must be positive."
            )

        if self.min_radius < 0 or self.max_radius < 0:
            raise ValueError("Circle radii cannot be negative.")

        if self.max_radius and self.max_radius < self.min_radius:
            raise ValueError("MaxRadius must be >= MinRadius.")

        blurred = cv2.medianBlur(gray, 5)
        min_dimension = min(gray.shape)
        # Avoid nearby shadow/texture arcs becoming extra circles in common
        # inspection images, while keeping the spacing automatic for users.
        min_dist = self.min_dist or max(1.0, min_dimension * 0.25)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        output = self._to_color(frame)

        detections = []

        if circles is not None:
            height, width = gray.shape
            edge_map = cv2.Canny(blurred, self.param1 / 2.0, self.param1)
            candidates = filter_circles_by_edge_support(
                circles[0], edge_map, min_support=0.30
            )

            accepted_circles = []
            for circle in np.round(candidates).astype(int):
                x, y, radius = map(int, circle)
                if radius <= 0:
                    continue
                if any(np.hypot(x - old_x, y - old_y) < 0.90 * (radius + old_radius)
                       for old_x, old_y, old_radius in accepted_circles):
                    continue
                accepted_circles.append((x, y, radius))

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
                    )
                )

        return output, detections

    def run(self):
        if self.task == "LineDetection":
            self._validate_line_params()
        elif min(self.dp, self.param1, self.param2) <= 0:
            raise ValueError("DP, Param1, and Param2 must be positive.")
        elif self.min_radius < 0 or self.max_radius < 0:
            raise ValueError("Circle radii cannot be negative.")
        elif self.max_radius and self.max_radius < self.min_radius:
            raise ValueError("MaxRadius must be >= MinRadius.")

        input_is_list = isinstance(self.image, list)
        input_images = self.image if input_is_list else [self.image]
        if any(image is None for image in input_images):
            raise ValueError("Input image entries cannot be null.")
        output_images = []
        self.detections = []

        for input_image in input_images:
            img = Image.get_frame(img=input_image, redis_db=self.redis_db)
            self._current_image_uid = self._image_uid(img, input_image)
            source_frame = np.asarray(img.value)
            frame, normalized = self._prepare_frame(source_frame)
            if self.task == "LineDetection":
                output, detections = self.detect_lines(frame)
            else:
                output, detections = self.detect_circles(frame)
            self.detections.extend(detections)
            img.value = self._restore_frame_dtype(output, source_frame, normalized)
            output_images.append(
                Image.set_frame(
                    img=img,
                    package_uID=self.uID,
                    redis_db=self.redis_db,
                )
            )

        self.image = output_images if input_is_list else output_images[0]

        return build_response(context=self)
