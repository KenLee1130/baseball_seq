"""Shared, user-facing pitch-location definitions.

Horizontal coordinates use ``plate_x_bv``: batter-relative plate width where
``-1`` is the inner edge, ``0`` is the centre, and ``+1`` is the outer edge.
Vertical coordinates use ``plate_z_norm``: ``0`` is the bottom and ``1`` the
top of that batter's strike zone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


IN_ZONE_NAMES = (
    ("內角高", "中間高", "外角高"),
    ("內角中", "正中", "外角中"),
    ("內角低", "中間低", "外角低"),
)
OUT_ZONE_NAMES = ("帶外高", "帶外低", "帶外內角", "帶外外角")


def region(df: pd.DataFrame) -> pd.Series:
    """Map every located pitch to one of 9 strike-zone or 4 chase regions.

    Savant's official ``zone`` decides whether the pitch is in the zone.  The
    3x3 label is then derived from batter-relative normalized coordinates, so
    inner/outer means the same thing for left- and right-handed hitters.
    Pitches outside the zone use high/low first, then inner/outer.
    """
    known = df["zone"].notna() & df["plate_x_bv"].notna() & df["plate_z_norm"].notna()
    inside = df["zone"].between(1, 9) & known

    x = df["plate_x_bv"]
    z = df["plate_z_norm"]
    col = np.select([x < -1 / 3, x <= 1 / 3], [0, 1], 2)
    row = np.select([z > 2 / 3, z >= 1 / 3], [0, 1], 2)
    names = np.asarray(IN_ZONE_NAMES, dtype=object)

    result = pd.Series("未知", index=df.index, dtype=object)
    result.loc[inside] = names[row[inside].astype(int), col[inside].astype(int)]

    outside = known & ~inside
    outside_names = np.select(
        [z > 1.0, z < 0.0, x < 0],
        ["帶外高", "帶外低", "帶外內角"],
        default="帶外外角",
    )
    result.loc[outside] = pd.Series(outside_names, index=df.index).loc[outside]
    return result

