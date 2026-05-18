import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ── All Redfin market files + metadata ──────────────────────
REDFIN_MARKETS = {
    "healdsburg.csv":   {"market": "Healdsburg, CA",   "state": "CA", "regulatory_safety": 0.70, "lifestyle_score": 0.92},
    "napa.csv":         {"market": "Napa, CA",          "state": "CA", "regulatory_safety": 0.65, "lifestyle_score": 0.90},
    "charleston.csv":   {"market": "Charleston, SC",    "state": "SC", "regulatory_safety": 0.75, "lifestyle_score": 0.85},
    "nashville.csv":    {"market": "Nashville, TN",     "state": "TN", "regulatory_safety": 0.70, "lifestyle_score": 0.80},
    "newport.csv":      {"market": "Newport, RI",       "state": "RI", "regulatory_safety": 0.72, "lifestyle_score": 0.83},
    "pacific_grove.csv":{"market": "Pacific Grove, CA", "state": "CA", "regulatory_safety": 0.68, "lifestyle_score": 0.88},
    "santa_cruz.csv":   {"market": "Santa Cruz, CA",    "state": "CA", "regulatory_safety": 0.65, "lifestyle_score": 0.85},
    "santa_barbara.csv":{"market": "Santa Barbara, CA", "state": "CA", "regulatory_safety": 0.60, "lifestyle_score": 0.90},
}

# ── All Airbnb cross-verification files ─────────────────────
AIRBNB_FILES = [
    "airbnb_nashville.csv",
    "airbnb_pacific_grove.csv",
    "airbnb_rhode_island.csv",
    "airbnb_san_fransisco.csv",
    "airbnb_santa_cruz.csv",
    "airbnb_denver.csv",
]

FIRST_NAMES = ["James","Sarah","Michael","Emily","Robert","Jennifer",
               "David","Amanda","John","Jessica","William","Ashley",
               "Richard","Megan","Thomas","Lauren","Charles","Stephanie",
               "Daniel","Nicole","Matthew","Elizabeth","Anthony","Rachel"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia",
               "Miller","Davis","Wilson","Anderson","Taylor","Thomas",
               "Jackson","White","Harris","Martin","Thompson","Robinson",
               "Clark","Lewis","Walker","Hall","Allen","Young"]


def load_redfin_csv(filepath, meta):
    """Load one Redfin CSV and standardize columns."""
    try:
        df = pd.read_csv(filepath, skiprows=1)
        if "PRICE" not in df.columns:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

    col_map = {
        "ADDRESS":               "address",
        "CITY":                  "city",
        "STATE_OR_PROVINCE":     "state",
        "ZIP_OR_POSTAL_CODE":    "zip_code",
        "PRICE":                 "property_value",
        "BEDS":                  "bedrooms",
        "BATHS":                 "bathrooms",
        "SQUARE_FEET":           "square_feet",
        "LOT_SIZE":              "lot_size",
        "YEAR_BUILT":            "year_built",
        "DAYS_ON_MARKET":        "days_on_market",
        "LATITUDE":              "latitude",
        "LONGITUDE":             "longitude",
        "$/SQUARE_FEET":         "price_per_sqft",
        "PROPERTY_TYPE":         "property_type",
    }
    existing = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)

    # Add market metadata
    df["market"]            = meta["market"]
    df["state"]             = meta["state"]
    df["regulatory_safety"] = meta["regulatory_safety"]
    df["lifestyle_score"]   = meta["lifestyle_score"]

    return df


def load_airbnb_data():
    """Load and combine all Airbnb listings for cross-verification."""
    dfs = []
    for fname in AIRBNB_FILES:
        if not os.path.exists(fname):
            print(f"  Airbnb file not found: {fname}")
            continue
        try:
            df = pd.read_csv(fname)
            df.columns = df.columns.str.strip().str.lower()

            # Keep only what we need for coordinate matching
            keep = []
            if "latitude"  in df.columns: keep.append("latitude")
            if "longitude" in df.columns: keep.append("longitude")
            if "name"      in df.columns: keep.append("name")
            if "price"     in df.columns: keep.append("price")
            if "room_type" in df.columns: keep.append("room_type")

            if "latitude" not in df.columns or "longitude" not in df.columns:
                print(f"  No coordinates in {fname} — skipping")
                continue

            df = df[keep].dropna(subset=["latitude","longitude"])
            df["airbnb_source"] = fname
            dfs.append(df)
            print(f"  Loaded {len(df)} Airbnb listings from {fname}")
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total Airbnb listings for cross-verification: {len(combined)}")
    return combined


def clean_numeric(df):
    """Clean numeric columns — remove commas, convert to float."""
    cols = ["property_value","square_feet","lot_size",
            "price_per_sqft","days_on_market","bedrooms","bathrooms"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",","").str.strip(),
                errors="coerce"
            )
    return df


def simulate_owner_info(df):
    """
    Simulate owner name and contact info.
    In production: ATTOM API for owner name + mailing address,
    Apollo.io / Hunter.io for email and phone enrichment.
    """
    n = len(df)
    df["owner_name"] = [
        f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        for _ in range(n)
    ]
    df["email"] = [
        f"{name.split()[0].lower()}.{name.split()[1].lower()}"
        f"{random.randint(1,99)}@email.com"
        for name in df["owner_name"]
    ]
    df["phone"] = [
        f"+1 {random.randint(200,999)} "
        f"{random.randint(100,999)} "
        f"{random.randint(1000,9999)}"
        for _ in range(n)
    ]
    return df


def load_all_data():
    """
    Main function — loads all Redfin CSVs, combines them,
    adds owner info. Returns (properties_df, airbnb_df).
    """
    print("\n── Loading Redfin property data ────────────────────")
    dfs = []
    for fname, meta in REDFIN_MARKETS.items():
        if not os.path.exists(fname):
            print(f"  Not found: {fname} — skipping")
            continue
        df = load_redfin_csv(fname, meta)
        if df.empty:
            print(f"  Empty: {fname} — skipping")
            continue
        print(f"  Loaded {len(df):>4} records from {meta['market']}")
        dfs.append(df)

    if not dfs:
        print("No Redfin data loaded.")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all markets
    properties = pd.concat(dfs, ignore_index=True)
    properties  = clean_numeric(properties)

    # Drop rows with no property value or coordinates
    properties = properties.dropna(subset=["property_value","latitude","longitude"])

    # Filter to realistic luxury second home range $400K — $10M
    properties = properties[
        (properties["property_value"] >= 400_000) &
        (properties["property_value"] <= 10_000_000)
    ].reset_index(drop=True)

    # Add simulated owner info
    properties = simulate_owner_info(properties)

    print(f"\n  Total properties: {len(properties)}")
    print(f"  Markets: {properties['market'].value_counts().to_dict()}")

    # Load Airbnb data
    print("\n── Loading Airbnb cross-verification data ──────────")
    airbnb = load_airbnb_data()

    return properties, airbnb


if __name__ == "__main__":
    props, airbnb = load_all_data()
    if not props.empty:
        print("\nSample properties:")
        print(props[["address","market","property_value","latitude","longitude"]].head())
    if not airbnb.empty:
        print("\nSample Airbnb listings:")
        print(airbnb[["latitude","longitude","airbnb_source"]].head())