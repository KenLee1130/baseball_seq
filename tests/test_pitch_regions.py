import pandas as pd

from analysis.pitch_regions import region


def test_region_names_all_nine_strike_zone_cells_in_batter_coordinates():
    frame = pd.DataFrame({
        "zone": [1] * 9,
        "plate_x_bv": [-0.8, 0.0, 0.8] * 3,
        "plate_z_norm": [0.85] * 3 + [0.5] * 3 + [0.15] * 3,
    })
    assert region(frame).tolist() == [
        "內角高", "中間高", "外角高",
        "內角中", "正中", "外角中",
        "內角低", "中間低", "外角低",
    ]


def test_region_names_four_chase_areas_and_unknown():
    frame = pd.DataFrame({
        "zone": [11, 12, 13, 14, None],
        "plate_x_bv": [0.0, 0.0, -1.4, 1.4, 0.0],
        "plate_z_norm": [1.2, -0.2, 0.5, 0.5, 0.5],
    })
    assert region(frame).tolist() == ["帶外高", "帶外低", "帶外內角", "帶外外角", "未知"]
