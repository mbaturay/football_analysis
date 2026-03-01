import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from view_transformer import ViewTransformer


def _make_tracks(position_adjusted):
    """Build a minimal tracks dict with one player at the given position."""
    return {
        "players": [
            {1: {"position_adjusted": position_adjusted}}
        ],
    }


def test_inside_polygon_yields_two_numbers():
    vt = ViewTransformer()  # default vertices
    # Point well inside the default pixel polygon
    tracks = _make_tracks((500, 500))
    vt.add_transformed_position_to_tracks(tracks)

    result = tracks["players"][0][1]["position_transformed"]
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


def test_outside_polygon_yields_none():
    vt = ViewTransformer()
    # Point far outside the default pixel polygon
    tracks = _make_tracks((0, 0))
    vt.add_transformed_position_to_tracks(tracks)

    result = tracks["players"][0][1]["position_transformed"]
    assert result is None
