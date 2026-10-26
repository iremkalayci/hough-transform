import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class LineDetection(HoughTransform):
    task = "LineDetection"

    PRESETS = {
        "GeneralLines": {
            "line_mode": "general",
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 50,
            "min_line_length": 30,
            "max_line_gap": 20,
            "merge_distance": 4.0,
            "max_output_lines": 100,
        },
        "BrokenLines": {
            "line_mode": "general",
            "canny_low": 35,
            "canny_high": 110,
            "hough_threshold": 25,
            "min_line_length": 15,
            "max_line_gap": 40,
            "merge_distance": 8.0,
            "max_output_lines": 150,
        },
        "StrictLines": {
            "line_mode": "general",
            "canny_low": 75,
            "canny_high": 200,
            "hough_threshold": 80,
            "min_line_length": 60,
            "max_line_gap": 10,
            "merge_distance": 2.0,
            "max_output_lines": 50,
        },
        "RoadLaneMarkings": {"line_mode": "road_lanes"},
    }

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        profile = self._active_task_configs().lineProfile.value

        if profile.name == "CustomLineSettings":
            self.line_mode = "general"
            self.canny_low = profile.cannyLow.value
            self.canny_high = profile.cannyHigh.value
            self.hough_threshold = profile.houghThreshold.value
            self.min_line_length = profile.minLineLength.value
            self.max_line_gap = profile.maxLineGap.value
            self.merge_distance = profile.mergeDistance.value
            self.max_output_lines = profile.maxOutputLines.value
        else:
            settings = self.PRESETS[profile.name]
            for name, value in settings.items():
                setattr(self, name, value)

            if profile.name == "RoadLaneMarkings":
                self.canny_low = profile.cannyLow.value
                self.canny_high = profile.cannyHigh.value
                self.hough_threshold = profile.houghThreshold.value
                self.min_line_length = profile.minLineLength.value
                self.max_line_gap = profile.maxLineGap.value
                self.merge_distance = profile.mergeDistance.value
                self.max_output_lines = profile.maxOutputLines.value
                self.road_roi_top_ratio = profile.roiTopRatio.value
                self.road_roi_top_left = profile.roiTopLeft.value
                self.road_roi_top_right = profile.roiTopRight.value
                self.road_white_lightness_min = profile.whiteLightnessMin.value
                self.road_white_saturation_max = profile.whiteSaturationMax.value


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
