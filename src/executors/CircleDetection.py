import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class CircleDetection(HoughTransform):
    task = "CircleDetection"

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        # Keep OpenCV tuning internal so users only choose the detection task.
        self.dp = 1.2
        self.min_dist = 0.0
        self.param1 = 100.0
        self.param2 = 30.0
        self.min_radius = 0
        self.max_radius = 0


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
