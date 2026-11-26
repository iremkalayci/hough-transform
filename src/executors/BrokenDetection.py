import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class BrokenDetection(HoughTransform):
    task = "BrokenDetection"


if __name__ == "__main__":
    Executor(sys.argv[1]).run()