import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class CircleDetection(HoughTransform):
    task = "CircleDetection"

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.dp = self.request.get_param("DP")
        self.min_dist = self.request.get_param("MinDist")
        self.param1 = self.request.get_param("Param1")
        self.param2 = self.request.get_param("Param2")
        self.min_radius = self.request.get_param("MinRadius")
        self.max_radius = self.request.get_param("MaxRadius")


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
