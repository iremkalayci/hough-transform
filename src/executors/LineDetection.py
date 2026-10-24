import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class LineDetection(HoughTransform):
    task = "LineDetection"

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        # Keep OpenCV tuning internal so users only choose the detection task.
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10
        self.merge_distance = 4.0


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
