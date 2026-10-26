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


def test_road_lane_filter_keeps_left_right_and_dashed_centre_markings():
    segments = np.array(
        [
            [480, 360, 120, 790],  # left lane boundary
            [520, 360, 880, 790],  # right lane boundary
            [500, 500, 504, 700],  # centre dash
        ]
    )
    result = line_ops.filter_road_lane_segments(segments, width=1000, height=800)
    assert len(result) == 3


def test_road_lane_filter_keeps_steep_perspective_lane_boundaries():
    # Coordinates mirror the perspective in the tested forward-facing road image.
    segments = np.array(
        [
            [448, 166, 0, 392],    # left lane boundary
            [448, 166, 872, 392],  # right lane boundary
            [448, 166, 444, 551],  # centre marking
        ]
    )
    result = line_ops.filter_road_lane_segments(segments, width=872, height=552)
    assert len(result) == 3


def test_road_lane_filter_rejects_lower_right_asphalt_texture_diagonals():
    segments = np.array(
        [
            [530, 350, 835, 534],
            [545, 344, 810, 520],
            [598, 410, 840, 550],
        ]
    )
    result = line_ops.filter_road_lane_segments(segments, width=872, height=552)
    assert result.shape == (0, 4)


def test_road_lane_filter_rejects_horizon_trees_and_asphalt_cracks():
    segments = np.array(
        [
            [0, 350, 1000, 350],  # horizon / tree line
            [100, 680, 850, 720],  # near-horizontal asphalt crack
            [80, 450, 80, 760],  # vertical non-centre object
            [50, 400, 110, 780],  # wrong direction on left side
        ]
    )
    result = line_ops.filter_road_lane_segments(segments, width=1000, height=800)
    assert result.shape == (0, 4)


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
