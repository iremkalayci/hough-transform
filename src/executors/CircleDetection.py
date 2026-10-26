import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.HoughTransform.src.executors.HoughTransform import HoughTransform


class CircleDetection(HoughTransform):
    task = "CircleDetection"

    PRESETS = {
        "GeneralCircles": {
            "dp": 1.2,
            "min_dist": 0.0,
            "param1": 100.0,
            "param2": 30.0,
            "min_radius": 0,
            "max_radius": 0,
            "max_output_circles": 100,
        },
        "CoinCircles": {
            "dp": 1.2,
            "min_dist": 0.0,
            "param1": 100.0,
            "param2": 35.0,
            "min_radius": 0,
            "max_radius": 0,
            "max_output_circles": 50,
        },
    }

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        profile = self._active_task_configs().circleProfile.value

        if profile.name == "CustomCircleSettings":
            self.dp = profile.dp.value
            self.min_dist = profile.minDist.value
            self.param1 = profile.param1.value
            self.param2 = profile.param2.value
            self.min_radius = profile.minRadius.value
            self.max_radius = profile.maxRadius.value
            self.max_output_circles = profile.maxOutputCircles.value
        else:
            settings = self.PRESETS[profile.name]
            for name, value in settings.items():
                setattr(self, name, value)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
