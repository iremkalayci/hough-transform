import importlib.util
import pathlib
import time

import numpy as np
import pytest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "src" / "utils" / "line_ops.py"
SPEC = importlib.util.spec_from_file_location("line_ops", MODULE_PATH)
line_ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(line_ops)


def test_merges_collinear_segments_across_gap_and_overlap():
    segments = np.array([[0, 10, 20, 10], [18, 10, 40, 10], [47, 10, 60, 10]])
    result = line_ops.merge_segments(segments, max_gap=8, merge_distance=2)
    np.testing.assert_array_equal(result, [[0, 10, 60, 10]])


def test_does_not_merge_segments_that_diverge_beyond_angle_tolerance():
    segments = np.array([[0, 0, 100, 0], [0, 0, 100, 7]])
    result = line_ops.merge_segments(segments, max_gap=10, merge_distance=20)
    assert len(result) == 2


def test_merge_distance_controls_parallel_stripes():
    stripes = np.array([[0, 10, 100, 10], [0, 22, 100, 22]])
    assert len(line_ops.merge_segments(stripes, 0, 14)) == 1
    assert len(line_ops.merge_segments(stripes, 0, 5)) == 2


def test_distant_parallel_segments_do_not_merge():
    segments = np.array([[0, 0, 10, 0], [100, 0, 110, 0]])
    assert len(line_ops.merge_segments(segments, max_gap=5, merge_distance=2)) == 2


def test_merge_is_order_independent():
    segments = np.array([[0, 10, 20, 10], [18, 10, 40, 10], [47, 10, 60, 10]])
    expected = line_ops.merge_segments(segments, 8, 2)
    for seed in range(5):
        np.testing.assert_array_equal(
            line_ops.merge_segments(np.random.default_rng(seed).permutation(segments), 8, 2),
            expected,
        )


def test_merge_off_keeps_raw_segments_longest_first_and_drops_zero_length():
    segments = np.array([[0, 0, 2, 0], [0, 0, 10, 0], [4, 4, 4, 4]])
    np.testing.assert_array_equal(
        line_ops.merge_segments(segments, 0, 0), [[0, 0, 10, 0], [0, 0, 2, 0]]
    )


def test_empty_and_single_segment_inputs():
    assert line_ops.merge_segments([], 0, 1).shape == (0, 4)
    np.testing.assert_array_equal(
        line_ops.merge_segments([[3, 4, 15, 20]], 0, 10), [[3, 4, 15, 20]]
    )
    with pytest.raises(ValueError):
        line_ops.merge_segments([], -1, 10)


def test_clip_segments_clamps_to_image_bounds_without_mutating_source():
    source = np.array([[-5.0, 2.0, 20.0, 30.0]])
    original = source.copy()
    np.testing.assert_array_equal(line_ops.clip_segments(source, 10, 8), [[0, 2, 9, 7]])
    np.testing.assert_array_equal(source, original)


def test_three_thousand_segments_complete_in_reasonable_time():
    rng = np.random.default_rng(7)
    starts = rng.integers(0, 1000, size=(3000, 2))
    ends = starts + rng.integers(20, 100, size=(3000, 2))
    segments = np.column_stack((starts, ends))
    started = time.perf_counter()
    result = line_ops.merge_segments(segments, max_gap=10, merge_distance=4)
    elapsed = time.perf_counter() - started
    assert result.ndim == 2 and result.shape[1] == 4
    assert elapsed < 8


def test_circle_filter_keeps_edge_supported_circle_and_rejects_texture_candidate():
    edges = np.zeros((100, 100), dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    x = np.rint(50 + 20 * np.cos(angles)).astype(int)
    y = np.rint(50 + 20 * np.sin(angles)).astype(int)
    edges[y, x] = 255

    result = line_ops.filter_circles_by_edge_support(
        [[50, 50, 20], [20, 20, 12]], edges, min_support=0.30
    )

    np.testing.assert_array_equal(result, [[50, 50, 20]])


def test_circle_filter_accepts_empty_candidates_and_validates_edge_map():
    assert line_ops.filter_circles_by_edge_support([], np.zeros((5, 5))).shape == (0, 3)
    with pytest.raises(ValueError):
        line_ops.filter_circles_by_edge_support([[2, 2, 1]], np.zeros((0, 0)))


def test_road_roi_is_an_internal_lower_trapezoid():
    np.testing.assert_array_equal(
        line_ops.road_roi_polygon((100, 200)),
        [[80, 36], [119, 36], [199, 99], [0, 99]],
    )
    with pytest.raises(ValueError):
        line_ops.road_roi_polygon((0, 200))


def test_road_angle_filter_removes_horizontal_and_near_vertical_segments():
    segments = np.asarray(
        [[0, 0, 100, 0], [0, 100, 100, 0], [50, 0, 52, 100]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        line_ops.filter_by_angle(segments), [[0, 100, 100, 0]]
    )


def test_vanishing_point_filter_keeps_road_aligned_lines():
    segments = np.asarray([[0, 70, 199, 0], [0, 95, 199, 60]])
    np.testing.assert_array_equal(
        line_ops.filter_toward_vanishing_point(segments, width=200, height=100),
        [[0, 70, 199, 0]],
    )


def test_estimated_vanishing_point_keeps_converging_lane_candidates():
    segments = np.asarray(
        [
            [0, 199, 98, 70],
            [199, 199, 102, 70],
            [50, 199, 99, 70],
            [150, 199, 101, 70],
            [0, 190, 190, 180],  # road texture: does not converge at the VP
        ],
        dtype=np.float64,
    )
    point, support = line_ops.estimate_vanishing_point(
        segments, width=200, height=200
    )

    assert point is not None
    assert 90 <= point[0] <= 110
    assert 60 <= point[1] <= 90
    np.testing.assert_array_equal(support, [True, True, True, True, False])


def test_clip_above_horizon_drops_and_trims_segments():
    segments = np.asarray(
        [[0, 20, 20, 80], [0, 10, 20, 30], [0, 10, 20, 15]], dtype=np.float64
    )
    result = line_ops.clip_above_horizon(segments, horizon_y=25)

    np.testing.assert_allclose(result, [[1.6666667, 25, 20, 80], [15, 25, 20, 30]])


def test_line_marking_scores_prefer_bright_painted_stripe_over_dark_seam():
    image = np.full((100, 100), 70, dtype=np.uint8)
    image[:, 20:27] = 255
    image[:, 70:77] = 105
    scores = line_ops.line_marking_scores(
        image, [[23, 5, 23, 95], [73, 5, 73, 95]]
    )

    assert scores[0] > 0.95
    assert scores[1] < 0.45


def test_select_road_markings_keeps_outer_lanes_and_short_centre_dashes():
    image = np.full((200, 200), 70, dtype=np.uint8)
    segments = np.asarray(
        [
            [0, 199, 95, 70],
            [199, 199, 105, 70],
            [100, 95, 100, 125],
            [100, 140, 100, 170],
            [25, 180, 100, 80],  # dark asphalt seam
        ],
        dtype=np.float64,
    )

    for x1, y1, x2, y2 in segments[:4]:
        samples = np.linspace(0.0, 1.0, 200)
        xs = np.rint(x1 + (x2 - x1) * samples).astype(int)
        ys = np.rint(y1 + (y2 - y1) * samples).astype(int)
        image[np.clip(ys, 0, 199), np.clip(xs, 0, 199)] = 255

    selected = line_ops.select_road_markings(
        image, segments, width=200, height=200, vanishing_point=(100, 65)
    )

    np.testing.assert_array_equal(selected, segments[:4])
