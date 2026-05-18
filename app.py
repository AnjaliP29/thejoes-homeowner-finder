import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import load_all_data
from cross_verify import cross_verify
from score import score_properties
from enrich import generate_outreach

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="The Joes — Homeowner Finder",
    page_icon="🏡",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 4px;
    }
    .verified-badge {
        background: #ECFDF5;
        color: #065F46;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .unverified-badge {
        background: #F1F5F9;
        color: #475569;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load and cache data ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    props, airbnb = load_all_data()
    verified      = cross_verify(props, airbnb)
    scored        = score_properties(verified)
    return scored


# ── Header ───────────────────────────────────────────────────
st.markdown("## 🏡 The Joes — Second Home Owner Finder")
st.markdown(
    "Identify, score, and reach out to potential homeowners "
    "for The Joes luxury midterm rental network."
)
st.divider()

# ── Load data ────────────────────────────────────────────────
with st.spinner("Loading property data and running cross-verification..."):
    df = load_data()

# ── Top metrics ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Properties", f"{len(df):,}")
with col2:
    st.metric("🟢 Hot Leads", f"{len(df[df['tier']=='A']):,}")
with col3:
    st.metric("🟡 Warm Leads", f"{len(df[df['tier']=='B']):,}")
with col4:
    st.metric("✅ Airbnb Verified", f"{df['airbnb_verified'].sum():,}")
with col5:
    st.metric("Markets Covered", f"{df['market'].nunique()}")

st.divider()

# ── Sidebar filters ──────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/home.png",
    width=60
)
st.sidebar.title("Filters")

# Market filter
markets = ["All Markets"] + sorted(df["market"].unique().tolist())
selected_market = st.sidebar.selectbox("Market", markets)

# Tier filter
tiers = st.sidebar.multiselect(
    "Lead Tier",
    options=["A", "B", "C"],
    default=["A", "B"],
    format_func=lambda x: {
        "A": "🟢 Hot Lead",
        "B": "🟡 Warm Lead",
        "C": "🔴 Low Priority"
    }[x]
)

# Airbnb verified filter
verified_filter = st.sidebar.radio(
    "Airbnb Verification",
    options=["All", "Verified Only", "Unverified Only"],
)

# Price range
min_val = int(df["property_value"].min())
max_val = int(df["property_value"].max())
price_range = st.sidebar.slider(
    "Property Value Range",
    min_value=min_val,
    max_value=max_val,
    value=(min_val, max_val),
    format="$%d"
)

# ── Apply filters ────────────────────────────────────────────
filtered = df.copy()

if selected_market != "All Markets":
    filtered = filtered[filtered["market"] == selected_market]

if tiers:
    filtered = filtered[filtered["tier"].isin(tiers)]

if verified_filter == "Verified Only":
    filtered = filtered[filtered["airbnb_verified"] == True]
elif verified_filter == "Unverified Only":
    filtered = filtered[filtered["airbnb_verified"] == False]

filtered = filtered[
    (filtered["property_value"] >= price_range[0]) &
    (filtered["property_value"] <= price_range[1])
]

st.markdown(f"### Showing {len(filtered):,} properties")

# ── Two column layout ────────────────────────────────────────
left, right = st.columns([1.2, 1])

# ── Left — Map ───────────────────────────────────────────────
with left:
    st.markdown("#### 📍 Property Map")

    map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
    map_df["verified_label"] = map_df["airbnb_verified"].map(
        {True: "✅ Airbnb Verified", False: "🔍 Potential Second Home"}
    )
    map_df["value_fmt"] = map_df["property_value"].apply(
        lambda x: f"${x:,.0f}"
    )

    color_map = {
        "✅ Airbnb Verified":        "#059669",
        "🔍 Potential Second Home":  "#2563EB",
    }

    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="verified_label",
        color_discrete_map=color_map,
        size="score",
        size_max=14,
        hover_name="address",
        hover_data={
            "market":        True,
            "value_fmt":     True,
            "score":         True,
            "tier_label":    True,
            "latitude":      False,
            "longitude":     False,
            "verified_label":False,
        },
        zoom=4,
        height=450,
        mapbox_style="carto-positron",
    )
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ── Right — Charts ───────────────────────────────────────────
with right:
    st.markdown("#### 📊 Market Breakdown")

    market_counts = (
        filtered.groupby(["market", "tier"])
        .size()
        .reset_index(name="count")
    )

    tier_colors = {"A": "#059669", "B": "#D97706", "C": "#DC2626"}

    fig_bar = px.bar(
        market_counts,
        x="market",
        y="count",
        color="tier",
        color_discrete_map=tier_colors,
        labels={"market": "", "count": "Properties", "tier": "Tier"},
        height=220,
    )
    fig_bar.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        legend_title="Tier",
        xaxis_tickangle=-30,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Verified vs unverified donut
    st.markdown("#### ✅ Verification Status")
    verified_counts = filtered["airbnb_verified"].value_counts()
    fig_donut = go.Figure(data=[go.Pie(
        labels=["Airbnb Verified", "Potential Second Home"],
        values=[
            verified_counts.get(True, 0),
            verified_counts.get(False, 0)
        ],
        hole=0.55,
        marker_colors=["#059669", "#2563EB"],
    )])
    fig_donut.update_layout(
        height=200,
        margin={"r":0,"t":10,"l":0,"b":0},
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# ── Lead table ───────────────────────────────────────────────
st.markdown("### 🏆 Ranked Lead List")

display_cols = [
    "rank", "owner_name", "address", "market",
    "property_value", "score", "tier_label", "airbnb_verified",
    "email", "phone"
]
display_df = filtered[
    [c for c in display_cols if c in filtered.columns]
].copy()

display_df["property_value"] = display_df["property_value"].apply(
    lambda x: f"${x:,.0f}"
)
display_df["score"] = display_df["score"].apply(lambda x: f"{x:.3f}")
display_df["airbnb_verified"] = display_df["airbnb_verified"].map(
    {True: "✅ Yes", False: "🔍 No"}
)
display_df.columns = [
    c.replace("_", " ").title()
    for c in display_df.columns
]

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    hide_index=True,
)

# Download button
csv = filtered.to_csv(index=False)
st.download_button(
    label="⬇️ Download Full Lead List as CSV",
    data=csv,
    file_name="thejoes_leads.csv",
    mime="text/csv",
)

st.divider()

# ── Outreach message generator ───────────────────────────────
st.markdown("### ✉️ AI Outreach Message Generator")
st.markdown(
    "Select a property to generate a personalized outreach "
    "message powered by Groq LLM."
)

top_leads = filtered.head(50)
if len(top_leads) == 0:
    st.warning("No properties match your current filters.")
else:
    options = [
        f"#{row['rank']} — {row['address']} | "
        f"{row['market']} | ${row['property_value']:,.0f}"
        for _, row in top_leads.iterrows()
        if pd.notna(row.get("address")) and str(row.get("address")) != "nan"
    ]

    if options:
        selected = st.selectbox("Select a property", options)
        selected_rank = int(selected.split("#")[1].split("—")[0].strip())
        selected_row  = filtered[filtered["rank"] == selected_rank].iloc[0]

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Property Details**")
            st.write(f"📍 **Address:** {selected_row.get('address', 'N/A')}")
            st.write(f"🏙️ **Market:** {selected_row.get('market', 'N/A')}")
            st.write(f"💰 **Value:** ${selected_row.get('property_value', 0):,.0f}")
            st.write(f"🎯 **Score:** {selected_row.get('score', 0):.3f}")
            st.write(f"✅ **Airbnb Verified:** "
                     f"{'Yes' if selected_row.get('airbnb_verified') else 'No'}")
            st.write(f"📧 **Email:** {selected_row.get('email', 'N/A')}")
            st.write(f"📞 **Phone:** {selected_row.get('phone', 'N/A')}")

        with col_b:
            st.markdown("**AI Generated Outreach Message**")
            if st.button("✨ Generate Message", type="primary"):
                with st.spinner("Generating personalized message..."):
                    message = generate_outreach(selected_row)
                st.text_area(
                    "Copy and send:",
                    value=message,
                    height=200,
                    label_visibility="collapsed"
                )
                st.success("Message ready to send!")
    else:
        st.info("No properties with addresses available in current filters.")

st.divider()

# ── Footer ───────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#94A3B8; font-size:0.8rem;'>"
    "Built for The Joes · "
    "Data: Redfin MLS + Inside Airbnb · "
    "Production would integrate ATTOM API for owner verification"
    "</div>",
    unsafe_allow_html=True
)