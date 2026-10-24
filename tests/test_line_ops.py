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


def test_does_not_merge_segments_outside_angle_filter():
    segments = np.array([[0, 0, 100, 0], [0, 0, 100, 20]])
    np.testing.assert_array_equal(line_ops.filter_by_angle(segments, 10, 90), [segments[1]])


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


def test_segment_angles_and_inclusive_filtering():
    segments = np.array([[0, 0, 10, 0], [0, 0, 10, 10], [10, 10, 0, 0]])
    np.testing.assert_allclose(line_ops.segment_angles_deg(segments), [0, 45, 45])
    np.testing.assert_array_equal(line_ops.filter_by_angle(segments, 45, 45), segments[1:])


def test_roi_rectangle_mask_keeps_requested_area():
    mask = line_ops.roi_mask((100, 200), 25, 20, 75, 80)
    assert mask.dtype == np.uint8
    assert mask[50, 100] == 255
    assert mask[10, 100] == 0
    assert mask[50, 10] == 0


def test_roi_trapezoid_mask_uses_top_inset():
    mask = line_ops.roi_mask((100, 100), 0, 0, 100, 100, 20)
    assert mask[10, 5] == 0
    assert mask[50, 50] == 255
    assert mask[50, 15] == 255


def test_full_roi_is_all_included_and_invalid_roi_raises():
    np.testing.assert_array_equal(line_ops.roi_mask((4, 5), 0, 0, 100, 100), 255)
    with pytest.raises(ValueError):
        line_ops.roi_mask((20, 20), 50, 0, 50, 100)
    with pytest.raises(ValueError):
        line_ops.roi_mask((20, 20), 0, 0, 100, 100, 51)


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
