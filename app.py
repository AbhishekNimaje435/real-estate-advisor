"""Streamlit app: Predict, Explore and Model Performance tabs."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src import config as C
from src.data_utils import add_features, add_targets, clean, load_data
from src.predict import predict_property

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top, #1e293b 0%, #0f172a 60%); color: #e2e8f0; }
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; color: white; margin-bottom: 1rem;
        background: linear-gradient(120deg, #6366f1, #06b6d4); box-shadow: 0 10px 30px rgba(0,0,0,.4); }
.card { padding: 1rem 1.2rem; border-radius: 14px; background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.12); }
.verdict-good { color: #34d399; font-size: 1.6rem; font-weight: 800; }
.verdict-bad { color: #f87171; font-size: 1.6rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    # Use a small sample when present (needed for cloud deployment size limits).
    if C.APP_SAMPLE_PATH.exists():
        raw = load_data(C.APP_SAMPLE_PATH)
    else:
        raw = load_data()
    return add_targets(add_features(clean(raw)))


@st.cache_data
def read_metrics():
    path = C.REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def models_ready():
    return (C.MODELS_DIR / "classifier.joblib").exists() and (C.MODELS_DIR / "regressor.joblib").exists()


st.markdown(
    '<div class="hero"><h1 style="margin:0">🏠 Real Estate Investment Advisor</h1>'
    '<p style="margin:.3rem 0 0">Check if a property is a Good Investment and see its estimated price after 5 years.</p></div>',
    unsafe_allow_html=True,
)

if not models_ready():
    st.error("Models nahi mile. Pehle `python -m src.train` chalayein.")
    st.stop()

df = get_data()
tab_predict, tab_explore, tab_model = st.tabs(["🔮 Predict", "📊 Explore", "🧠 Model Performance"])


def options(col):
    return sorted(df[col].dropna().unique().tolist())


# ================= PREDICT =================
with tab_predict:
    st.subheader("Property details")
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", options("State"))
    city = c2.selectbox("City", sorted(df.loc[df["State"] == state, "City"].unique()))

    with st.form("property_form"):
        a, b, c = st.columns(3)
        property_type = a.selectbox("Property type", options("Property_Type"))
        bhk = b.number_input("BHK", min_value=1, max_value=10, value=2)
        size = c.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1200)

        d, e, f = st.columns(3)
        price = d.number_input("Current price (Lakhs)", min_value=1.0, value=80.0)
        year_built = e.number_input("Year built", min_value=1950, max_value=2026, value=2015)
        floor_no = f.number_input("Floor number", min_value=0, max_value=100, value=3)

        g, h, i = st.columns(3)
        total_floors = g.number_input("Total floors", min_value=1, max_value=100, value=10)
        schools = h.number_input("Nearby schools", min_value=0, max_value=50, value=3)
        hospitals = i.number_input("Nearby hospitals", min_value=0, max_value=50, value=2)

        j, k, l = st.columns(3)
        parking = j.selectbox("Parking", options("Parking_Space"))
        furnished = k.selectbox("Furnished status", options("Furnished_Status"))
        transport = l.selectbox("Public transport", options("Public_Transport_Accessibility"))

        m, n, o = st.columns(3)
        facing = m.selectbox("Facing", options("Facing"))
        owner = n.selectbox("Owner type", options("Owner_Type"))
        availability = o.selectbox("Availability", options("Availability_Status"))

        p, q = st.columns(2)
        security = p.selectbox("Security", options("Security"))
        amenities_count = q.number_input("Number of amenities", min_value=0, max_value=20, value=3)

        submitted = st.form_submit_button("Analyse property ✨")

    if submitted:
        payload = {
            "State": state, "City": city, "Property_Type": property_type,
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
            "Year_Built": year_built, "Floor_No": floor_no, "Total_Floors": total_floors,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Parking_Space": parking, "Furnished_Status": furnished,
            "Public_Transport_Accessibility": transport, "Facing": facing,
            "Owner_Type": owner, "Availability_Status": availability,
            "Security": security, "Amenities_Count": amenities_count,
        }
        result = predict_property(payload)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown('<div class="card"><small>Is this a Good Investment?</small><br>'
                        + ('<span class="verdict-good">✅ Yes</span>' if result["good_investment"]
                           else '<span class="verdict-bad">❌ Not really</span>')
                        + f'<br><small>Confidence: {result["good_probability"]*100:.1f}%</small></div>',
                        unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="card"><small>Estimated price after 5 years</small><br>'
                        f'<span style="font-size:1.6rem;font-weight:800">₹{result["future_price_lakhs"]:.1f} L</span>'
                        f'<br><small>Growth rate used: {result["growth_rate_used"]*100:.0f}% / year</small></div>',
                        unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="card"><small>Expected gain</small><br>'
                        f'<span style="font-size:1.6rem;font-weight:800">{result["expected_gain_pct"]:.1f}%</span>'
                        f'<br><small>Current price: ₹{result["current_price_lakhs"]:.1f} L</small></div>',
                        unsafe_allow_html=True)

        # Compare with similar properties in the same city and BHK
        peers = df[(df["City"] == city) & (df["BHK"] == bhk)]
        if len(peers):
            my_ppsqft = price * 1e5 / size
            peer_median = peers["Price_per_SqFt"].median()
            st.info(f"Is city aur {bhk} BHK ke {len(peers):,} properties mein median price per sq ft "
                    f"₹{peer_median:,.0f} hai. Is property ka ₹{my_ppsqft:,.0f} hai.")

# ================= EXPLORE =================
with tab_explore:
    f1, f2, f3 = st.columns(3)
    sel_states = f1.multiselect("State", options("State"))
    sel_bhk = f2.multiselect("BHK", options("BHK"))
    pmin, pmax = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
    price_range = f3.slider("Price (Lakhs)", pmin, pmax, (pmin, pmax))

    view = df
    if sel_states:
        view = view[view["State"].isin(sel_states)]
    if sel_bhk:
        view = view[view["BHK"].isin(sel_bhk)]
    view = view[view["Price_in_Lakhs"].between(*price_range)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Properties", f"{len(view):,}")
    m2.metric("Avg price (L)", f"{view['Price_in_Lakhs'].mean():.1f}" if len(view) else "-")
    m3.metric("Avg ₹/sq ft", f"{view['Price_per_SqFt'].mean():,.0f}" if len(view) else "-")
    m4.metric("Good investment rate", f"{view[C.TARGET_CLS].mean()*100:.1f}%" if len(view) else "-")

    if view.empty:
        st.warning("Filters se koi property nahi mili.")
    else:
        left, right = st.columns(2)
        left.subheader("Avg ₹/sq ft by city (top 15)")
        left.bar_chart(view.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False).head(15))
        right.subheader("Good investment rate by property type")
        right.bar_chart(view.groupby("Property_Type")[C.TARGET_CLS].mean())

        st.subheader("Heatmap: avg ₹/sq ft by city and BHK")
        pivot = view.pivot_table(index="City", columns="BHK", values="Price_per_SqFt", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(pivot.head(15), annot=True, fmt=".0f", cmap="viridis", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Properties")
        cols = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
                "Price_in_Lakhs", "Price_per_SqFt", C.TARGET_CLS]
        st.dataframe(view[cols].head(200), use_container_width=True)

# ================= MODEL PERFORMANCE =================
with tab_model:
    report = read_metrics()
    if report is None:
        st.info("metrics.json nahi mila. `python -m src.train` chalayein.")
    else:
        st.success(f"Best classifier: **{report['best_classifier']}** | "
                   f"Best regressor: **{report['best_regressor']}**")
        c1, c2 = st.columns(2)
        c1.subheader("Classification (test set)")
        c1.dataframe(pd.DataFrame(report["classification"]).T.round(4))
        c2.subheader("Regression (test set)")
        c2.dataframe(pd.DataFrame(report["regression"]).T.round(2))

        c3, c4 = st.columns(2)
        c3.subheader("Confusion matrix (best classifier)")
        cm = pd.DataFrame(report["classification_confusion_matrix"],
                          index=["Actual: Not good", "Actual: Good"],
                          columns=["Pred: Not good", "Pred: Good"])
        c3.dataframe(cm)
        c4.subheader("Top features (classifier)")
        c4.bar_chart(pd.Series(report["top_features_classifier"]).sort_values())

        st.subheader("Top features (regressor)")
        st.bar_chart(pd.Series(report["top_features_regressor"]).sort_values())

    figs = sorted(C.FIGURES_DIR.glob("*.png")) if C.FIGURES_DIR.exists() else []
    if figs:
        with st.expander("EDA figures"):
            for fig_path in figs:
                st.image(str(fig_path), caption=fig_path.stem)
