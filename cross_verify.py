import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ── Match radius in meters ───────────────────────────────────
# If a Redfin property and Airbnb listing are within 150 meters
# of each other, we consider them the same property
MATCH_RADIUS_METERS = 150


def haversine_meters(lat1, lon1, lat2, lon2):
    """
    Calculate distance in meters between two coordinates.
    Uses haversine formula for accurate geographic distance.
    """
    R = 6_371_000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def cross_verify(properties, airbnb):
    """
    Cross-verify Redfin properties against Airbnb listings
    using coordinate proximity matching.

    Logic:
    - For each Redfin property, check if any Airbnb listing
      exists within MATCH_RADIUS_METERS of its coordinates
    - If yes → property is Airbnb-verified (confirmed second home)
    - If no  → potential second home based on property signals only

    Returns properties DataFrame with added columns:
    - airbnb_verified: True/False
    - airbnb_match_distance: closest Airbnb listing distance in meters
    - airbnb_source: which Airbnb file matched
    """

    if airbnb.empty:
        print("  No Airbnb data — skipping cross-verification")
        properties["airbnb_verified"]       = False
        properties["airbnb_match_distance"] = None
        properties["airbnb_source"]         = None
        return properties

    print(f"\n── Cross-verifying {len(properties)} properties "
          f"against {len(airbnb)} Airbnb listings ──")
    print(f"  Match radius: {MATCH_RADIUS_METERS} meters")

    # Convert to numpy arrays for fast computation
    prop_lats  = properties["latitude"].values
    prop_lons  = properties["longitude"].values
    airbnb_lats = airbnb["latitude"].values
    airbnb_lons = airbnb["longitude"].values
    airbnb_srcs = airbnb["airbnb_source"].values

    verified        = []
    match_distances = []
    match_sources   = []

    for i, (plat, plon) in enumerate(zip(prop_lats, prop_lons)):
        # Quick bounding box filter first (fast)
        # 150 meters ≈ 0.00135 degrees latitude
        lat_margin = 0.00135
        lon_margin = 0.00180

        mask = (
            (np.abs(airbnb_lats - plat) < lat_margin) &
            (np.abs(airbnb_lons - plon) < lon_margin)
        )
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            verified.append(False)
            match_distances.append(None)
            match_sources.append(None)
            continue

        # Precise haversine distance for candidates only
        min_dist   = float("inf")
        min_source = None

        for j in candidates:
            dist = haversine_meters(plat, plon,
                                    airbnb_lats[j], airbnb_lons[j])
            if dist < min_dist:
                min_dist   = dist
                min_source = airbnb_srcs[j]

        if min_dist <= MATCH_RADIUS_METERS:
            verified.append(True)
            match_distances.append(round(min_dist, 1))
            match_sources.append(min_source)
        else:
            verified.append(False)
            match_distances.append(round(min_dist, 1))
            match_sources.append(None)

    properties = properties.copy()
    properties["airbnb_verified"]       = verified
    properties["airbnb_match_distance"] = match_distances
    properties["airbnb_source"]         = match_sources

    # Summary
    n_verified = sum(verified)
    pct        = round(n_verified / len(properties) * 100, 1)
    print(f"\n  Results:")
    print(f"  Airbnb-verified properties : {n_verified} ({pct}%)")
    print(f"  Potential second homes     : {len(properties) - n_verified}")
    print(f"\n  Verified by market:")

    verified_by_market = (
        properties[properties["airbnb_verified"]]
        .groupby("market")
        .size()
        .sort_values(ascending=False)
    )
    for market, count in verified_by_market.items():
        total = len(properties[properties["market"] == market])
        print(f"    {market:<25} {count:>4} / {total} verified")

    return properties


if __name__ == "__main__":
    from data import load_all_data
    props, airbnb = load_all_data()
    if not props.empty:
        result = cross_verify(props, airbnb)
        print("\nSample verified properties:")
        verified = result[result["airbnb_verified"] == True]
        print(verified[["address", "market", "property_value",
                         "airbnb_match_distance"]].head(10))