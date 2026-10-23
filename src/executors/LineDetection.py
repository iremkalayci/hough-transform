import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class LineDetection(HoughTransform):
    task = "LineDetection"

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.canny_low = self.request.get_param("CannyLow")
        self.canny_high = self.request.get_param("CannyHigh")
        self.hough_threshold = self.request.get_param("HoughThreshold")
        self.min_line_length = self.request.get_param("MinLineLength")
        self.max_line_gap = self.request.get_param("MaxLineGap")


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
