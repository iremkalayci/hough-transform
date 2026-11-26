import os
import sys

import re

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox

from capsules.HoughTransform.src.models.PackageModel import Detection, PackageModel
from capsules.HoughTransform.src.utils.response import build_response
from capsules.HoughTransform.src.utils import vision_ops


MAX_WORKING_SIDE = 1280

LINE_COLOR = (0, 0, 255)
CIRCLE_COLOR = (0, 200, 0)
ELLIPSE_COLOR = (255, 0, 255)
SHAPE_COLOR = (255, 160, 0)

# classId used in the detection output
CLASS_IDS = {"line": 0, "circle": 1, "ellipse": 2}
SHAPE_CLASS_ID = 3

TASKS = ("LineDetection", "CircleDetection", "NoiseDetection", "BrokenDetection",
         "PartialEdgeDetection", "EllipseDetection", "ShapeDetection")


class HoughTransform(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.image = self.request.get_param("inputImage")
        self.detections = []

    def _user_settings(self):
        """Values of the active executor's fields, keyed like ``settings.DEFAULTS``."""
        configs = self.request.model.configs.executor.value.value.configs
        values = {}
        for name in type(configs).model_fields if hasattr(type(configs), "model_fields") \
                else configs.__fields__:
            field = getattr(configs, name, None)
            if field is not None and hasattr(field, "value"):
                values[re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()] = field.value
        return values

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

    # ------------------------------------------------------------------
    # Task pipelines. Each returns lines (N,4), circles (M,3),
    # ellipses [((cx,cy),(w,h),angle)] and shapes [(label, polygon, bbox)].
    # ------------------------------------------------------------------

    def analyze(self, gray, cfg=None):
        """Run the pipeline of the active task on a grayscale uint8 frame.

        ``cfg`` holds the task's settings (see ``settings.DEFAULTS``); when it
        is omitted the defaults are used.
        """
        task = self.task
        if task not in TASKS:
            raise ValueError("Unsupported Hough detection task: {!r}".format(task))
        cfg = dict(cfg or {})
        empty_circles = np.empty((0, 3))
        lines, circles, ellipses, shapes = np.empty((0, 4)), empty_circles, [], []

        ed = vision_ops.prepare_edges(
            gray,
            strong_denoise=(task == "NoiseDetection"),
            sensitivity=cfg.get("edge_sensitivity_pct", 100.0),
            denoise_strength=cfg.get("denoise_strength_pct", 100.0),
        )

        if task in ("LineDetection", "NoiseDetection", "BrokenDetection", "PartialEdgeDetection"):
            broken = task in ("BrokenDetection", "PartialEdgeDetection")
            # Explain away clear circles first, so rim chords are not reported as lines.
            solid = vision_ops.detect_circles(ed, "full")
            # Partial edges reuse the broken-line gap rules (fixed: they are part of
            # what makes an edge "partial", not a user preference).
            line_cfg = cfg if task != "PartialEdgeDetection" else dict(
                cfg, min_dash_fill_pct=22.0, max_gap_pct=20.0)
            lines = vision_ops.detect_lines(ed.without_circles(solid), broken=broken, cfg=line_cfg)
            circles = solid
            if task in ("BrokenDetection", "PartialEdgeDetection"):
                mode = "broken" if task == "BrokenDetection" else "partial"
                extra = vision_ops.detect_circles(ed.without_circles(solid), mode, cfg=cfg)
                # Chords of dashed rims / arcs are not lines; circles that merely
                # trace a straight line are not circles.
                lines = vision_ops.drop_lines_on_circles(lines, extra)
                extra = vision_ops.drop_circles_on_lines(ed, extra, lines)
                circles = vision_ops.merge_circles(solid, extra)
            if task == "LineDetection":
                circles = empty_circles
            if task == "NoiseDetection":
                solid = vision_ops.detect_circles(ed, "full", cfg=cfg)
                circles = solid
        elif task == "CircleDetection":
            circles = vision_ops.detect_circles(ed, "full", cfg=cfg)
        elif task == "EllipseDetection":
            ellipses = vision_ops.detect_ellipses(ed, cfg=cfg)
        elif task == "ShapeDetection":
            shapes = vision_ops.detect_shapes(ed, cfg=cfg)

        if len(lines) and len(circles):
            lines = vision_ops.drop_lines_on_circles(lines, circles)
            circles = vision_ops.drop_circles_on_lines(ed, circles, lines)
        return lines, circles, ellipses, shapes

    def detect(self, frame, cfg=None):
        """Detect on ``frame`` (uint8, any channel layout); return annotated BGR + detections."""
        gray = self._to_gray(frame)
        height, width = gray.shape[:2]
        scale = min(1.0, MAX_WORKING_SIDE / float(max(height, width)))
        work = gray
        if scale < 1.0:
            work = cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))),
                              interpolation=cv2.INTER_AREA)
        lines, circles, ellipses, shapes = self.analyze(work, cfg)
        inv = 1.0 / scale

        output = self._to_color(frame)
        detections = []
        thickness = max(2, int(round(min(height, width) / 300)))

        for x1, y1, x2, y2 in np.asarray(lines, dtype=np.float64).reshape(-1, 4):
            x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
            cv2.line(output, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                     LINE_COLOR, thickness, cv2.LINE_AA)
            detections.append(self._make_detection(
                CLASS_IDS["line"], "line",
                (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))

        for cx, cy, r in np.asarray(circles, dtype=np.float64).reshape(-1, 3):
            cx, cy, r = cx * inv, cy * inv, r * inv
            cv2.circle(output, (int(round(cx)), int(round(cy))), int(round(r)),
                       CIRCLE_COLOR, thickness, cv2.LINE_AA)
            cv2.circle(output, (int(round(cx)), int(round(cy))), max(2, thickness),
                       CIRCLE_COLOR, -1, cv2.LINE_AA)
            detections.append(self._make_detection(
                CLASS_IDS["circle"], "circle",
                (max(0, cx - r), max(0, cy - r), min(width, cx + r), min(height, cy + r))))

        for (cx, cy), (w, h), angle in ellipses:
            cx, cy, w, h = cx * inv, cy * inv, w * inv, h * inv
            cv2.ellipse(output, ((cx, cy), (w, h), angle), ELLIPSE_COLOR, thickness, cv2.LINE_AA)
            box = cv2.boxPoints(((cx, cy), (w, h), angle))
            detections.append(self._make_detection(
                CLASS_IDS["ellipse"], "ellipse",
                (max(0, box[:, 0].min()), max(0, box[:, 1].min()),
                 min(width, box[:, 0].max()), min(height, box[:, 1].max()))))

        for label, polygon, (bx1, by1, bx2, by2) in shapes:
            poly = np.rint(np.asarray(polygon, dtype=np.float64) * inv).astype(np.int32)
            cv2.polylines(output, [poly.reshape(-1, 1, 2)], True, SHAPE_COLOR, thickness, cv2.LINE_AA)
            # Black text with a thin white halo stays readable on any background.
            text_origin = (int(bx1 * inv), max(12, int(by1 * inv) - 4))
            text_scale = max(0.4, min(height, width) / 900.0)
            text_width = max(1, thickness - 1)
            cv2.putText(output, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                        (255, 255, 255), text_width + 2, cv2.LINE_AA)
            cv2.putText(output, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                        (0, 0, 0), text_width, cv2.LINE_AA)
            detections.append(self._make_detection(
                SHAPE_CLASS_ID, label, (bx1 * inv, by1 * inv, bx2 * inv, by2 * inv)))

        return output, detections

    def run(self):
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
            output, detections = self.detect(frame, self._user_settings())
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