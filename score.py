import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Scoring weights ──────────────────────────────────────────
WEIGHTS = {
    "property_value":     0.30,  # Higher value = better quality = brand fit
    "lifestyle_score":    0.20,  # Market lifestyle pull
    "regulatory_safety":  0.15,  # Low regulatory risk
    "price_per_sqft":     0.15,  # Luxury signal
    "airbnb_bonus":       0.20,  # Airbnb verification bonus
}


def build_airbnb_bonus(df):
    """
    Airbnb verification adds a bonus score on top of base scoring.
    Verified = strong bonus, closer match = higher bonus.
    Unverified = small base score (not zero — property signals still matter)
    """
    df = df.copy()

    def compute_bonus(row):
        if not row["airbnb_verified"]:
            return 0.10  # small base — property signals still matter
        dist = row["airbnb_match_distance"]
        if dist is None:    return 0.80
        if dist <= 20:      return 1.00
        if dist <= 50:      return 0.95
        if dist <= 100:     return 0.90
        if dist <= 150:     return 0.85
        return 0.80

    df["airbnb_bonus"] = df.apply(compute_bonus, axis=1)
    return df


def clean_property_values(df):
    """
    Remove rows with clearly bad property values.
    $0 or $1 means the price didn't load correctly.
    """
    df = df.copy()
    before = len(df)
    df = df[df["property_value"] >= 100_000]
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed} records with invalid property values")
    return df


def score_properties(df):
    """
    Score and rank all properties for The Joes network.
    Returns DataFrame with composite score, rank, and tier.
    """
    df = df.copy()
    df = clean_property_values(df)
    df = build_airbnb_bonus(df)

    features = list(WEIGHTS.keys())

    # Fill missing values with column median
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            print(f"  Warning: '{col}' not found — filling with 0.5")
            df[col] = 0.5

    # Normalize all features to [0, 1]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    # Weighted composite score
    df["score"] = sum(
        df_scaled[col] * weight
        for col, weight in WEIGHTS.items()
    )
    df["score"] = df["score"].round(4)

    # Rank — 1 = best candidate
    df["rank"] = df["score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Tier — based on score percentile for better distribution
    p33 = df["score"].quantile(0.33)
    p66 = df["score"].quantile(0.66)

    def assign_tier(score):
        if score >= p66:  return "A"
        if score >= p33:  return "B"
        return "C"

    df["tier"] = df["score"].apply(assign_tier)

    # Human readable labels
    tier_map = {
        "A": "🟢 Hot Lead",
        "B": "🟡 Warm Lead",
        "C": "🔴 Low Priority"
    }
    df["tier_label"] = df["tier"].map(tier_map)

    # Sort by score descending
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def score_summary(df):
    """Print a clean summary of scoring results."""
    print("\n── Scoring Summary ─────────────────────────────────")
    print(f"  Total properties scored  : {len(df)}")
    print(f"  🟢 Hot Leads   (A tier)  : {len(df[df['tier']=='A'])}")
    print(f"  🟡 Warm Leads  (B tier)  : {len(df[df['tier']=='B'])}")
    print(f"  🔴 Low Priority (C tier) : {len(df[df['tier']=='C'])}")
    print(f"\n  Airbnb-verified leads    : {df['airbnb_verified'].sum()}")
    print(f"  Unverified candidates    : {(~df['airbnb_verified']).sum()}")

    print("\n  Top 10 candidates:")
    top = df[[
        "rank", "owner_name", "address", "market",
        "property_value", "score", "tier_label", "airbnb_verified"
    ]].head(10).copy()
    top["property_value"] = top["property_value"].apply(
        lambda x: f"${x:,.0f}"
    )
    print(top.to_string(index=False))

    print("\n  Average score by market:")
    for market in df["market"].unique():
        mdf      = df[df["market"] == market]
        score    = mdf["score"].mean()
        verified = mdf["airbnb_verified"].sum()
        total    = len(mdf)
        print(f"    {market:<25} score: {score:.3f}  "
              f"verified: {verified}/{total}")


if __name__ == "__main__":
    from data import load_all_data
    from cross_verify import cross_verify

    props, airbnb = load_all_data()
    if not props.empty:
        verified = cross_verify(props, airbnb)
        scored   = score_properties(verified)
        score_summary(scored)