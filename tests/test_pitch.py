"""Coordinate systems, zones and pitch geometry."""

import numpy as np
import pytest

from src.config import STD_PITCH_LENGTH, STD_PITCH_WIDTH
from src.utils.pitch import (STANDARD_PITCH, Pitch, angle_to_goal, channels,
                             distance_to_goal, from_normalised, grid, in_penalty_area,
                             penalty_spots, pitch_line_segments, thirds, to_attacking,
                             to_normalised, to_standard)


def test_standard_pitch_dimensions():
    assert STANDARD_PITCH.length == STD_PITCH_LENGTH == 105.0
    assert STANDARD_PITCH.width == STD_PITCH_WIDTH == 68.0
    assert STANDARD_PITCH.half_length == 52.5
    assert STANDARD_PITCH.area == pytest.approx(7140.0)


def test_to_standard_rescales_corners_onto_the_reference_pitch():
    p = Pitch(106.0, 68.0)
    x, y = to_standard([p.half_length, -p.half_length], [p.half_width, -p.half_width], p)
    assert np.allclose(x, [52.5, -52.5])
    assert np.allclose(y, [34.0, -34.0])


def test_to_standard_is_identity_on_a_standard_pitch():
    x, y = to_standard([10.0, -3.0], [4.0, -8.0], STANDARD_PITCH)
    assert np.allclose(x, [10.0, -3.0])
    assert np.allclose(y, [4.0, -8.0])


def test_to_attacking_flips_both_axes_and_round_trips():
    x0, y0 = np.array([12.0, -7.0]), np.array([-4.0, 9.0])
    x1, y1 = to_attacking(x0, y0, -1)
    assert np.allclose(x1, -x0) and np.allclose(y1, -y0)
    x2, y2 = to_attacking(x1, y1, -1)
    assert np.allclose(x2, x0) and np.allclose(y2, y0)


def test_normalised_round_trip():
    x = np.array([-52.5, 0.0, 52.5])
    y = np.array([-34.0, 0.0, 34.0])
    nx, ny = to_normalised(x, y)
    assert np.allclose(nx, [0, 50, 100])
    assert np.allclose(ny, [0, 50, 100])
    bx, by = from_normalised(nx, ny)
    assert np.allclose(bx, x) and np.allclose(by, y)


def test_thirds_boundaries():
    assert list(thirds([-52.4, -17.6, -17.4, 0.0, 17.4, 17.6, 52.4])) == [
        "defensive_third", "defensive_third", "middle_third", "middle_third",
        "middle_third", "attacking_third", "attacking_third",
    ]


def test_channels_are_symmetric_and_ordered():
    got = list(channels([-30.0, -15.0, 0.0, 15.0, 30.0]))
    assert got == ["wide_right", "half_space_right", "centre", "half_space_left", "wide_left"]
    assert list(channels([-1e-9]))[0] == "centre"


def test_penalty_area_membership():
    assert bool(in_penalty_area(50.0, 0.0))
    assert not bool(in_penalty_area(30.0, 0.0))
    assert not bool(in_penalty_area(50.0, 25.0))
    assert bool(in_penalty_area(-50.0, 0.0, attacking=False))


def test_distance_and_angle_to_goal():
    assert float(distance_to_goal(52.5, 0.0)) == pytest.approx(0.0)
    assert float(distance_to_goal(42.5, 0.0)) == pytest.approx(10.0)
    # Directly in front of goal subtends a wider angle than out wide at equal distance.
    assert float(angle_to_goal(42.5, 0.0)) > float(angle_to_goal(42.5, 20.0))
    # Angle is symmetric in y.
    assert float(angle_to_goal(40.0, 12.0)) == pytest.approx(float(angle_to_goal(40.0, -12.0)))


def test_pitch_lines_stay_inside_a_small_margin():
    for xs, ys in pitch_line_segments():
        assert max(abs(np.asarray(xs))) <= STANDARD_PITCH.half_length + 2.01
        assert max(abs(np.asarray(ys))) <= STANDARD_PITCH.half_width + 0.01


def test_penalty_spots_are_eleven_metres_out():
    spots = penalty_spots()
    assert spots[0][0] == pytest.approx(-(52.5 - 11.0))
    assert spots[1][0] == pytest.approx(52.5 - 11.0)


def test_grid_covers_the_pitch_and_has_the_requested_shape():
    cx, cy, (X, Y) = grid(10, 8)
    assert X.shape == (8, 10)
    assert cx.min() > -52.5 and cx.max() < 52.5
    assert np.allclose(np.diff(cx), np.diff(cx)[0])
