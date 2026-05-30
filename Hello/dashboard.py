import requests
import streamlit as st
import pandas as pd

# Physical and financial constants used throughout the calculations
DAYS_PER_YEAR             = 365
KG_PER_TON                = 1_000
CARBON_FACTOR_KG_PER_KWH  = 0.82       # India grid average (CEA 2023)
COST_PER_KW_INR           = 50_000     # ₹50,000 per kWp, India average install cost
PANEL_POWER_DENSITY_KW_M2 = 0.20       # 0.2 kW per m² — standard installer basis
PANEL_DEGRADATION_RATE    = 0.005      # panels lose ~0.5% efficiency each year
MAINTENANCE_COST_PER_KW   = 500        # ₹500/kWp/year for O&M
DEFAULT_IRRADIANCE        = 4.8        # kWh/m²/day — conservative India-wide fallback
MIN_SAVINGS_THRESHOLD     = 1.0        # below ₹1/year net savings, payback is treated as infinite
NOMINATIM_USER_AGENT      = "SolarPVAnalyzer/5.0"
RETRYABLE_STATUS_CODES    = {429, 500, 502, 503, 504}
API_MAX_RETRIES           = 2
API_BASE_DELAY_S          = 0.8        # short enough not to visibly block the UI


# ── Pure utilities — no Streamlit, safe to unit-test independently ────────────

def format_inr(amount: float) -> str:
    """Display rupee amounts in Crores, Lakhs, or plain — whichever fits."""
    if amount >= 1e7:
        return f"₹ {amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹ {amount / 1e5:.2f} L"
    return f"₹ {amount:,.0f}"


def calculate_solar_metrics(
    df: pd.DataFrame,
    irradiance: float,
    efficiency: float,
    loss_factor: float,
    usage_factor: float,
    elec_rate: float,
) -> pd.DataFrame:
    """
    Add solar output and financial columns to the buildings DataFrame.

    usage_factor scales the installed system size (how much roof is usable).
    Payback accounts for annual panel degradation and O&M costs, not just
    a naive cost-divided-by-savings ratio.
    """
    df = df.copy()

    conversion_factor = efficiency * (1 - loss_factor) * usage_factor * DAYS_PER_YEAR
    kwh_per_m2_year   = irradiance * conversion_factor

    area                                = df["Area"]
    df["System Size (kW)"]              = area * PANEL_POWER_DENSITY_KW_M2 * usage_factor
    df["Estimated Annual Energy (kWh)"] = area * kwh_per_m2_year
    df["CO2 Offset (tCO2e)"]            = df["Estimated Annual Energy (kWh)"] * CARBON_FACTOR_KG_PER_KWH / KG_PER_TON
    df["System Cost (₹)"]               = df["System Size (kW)"] * COST_PER_KW_INR
    annual_om                           = df["System Size (kW)"] * MAINTENANCE_COST_PER_KW
    df["Financial Savings (₹)"]         = (df["Estimated Annual Energy (kWh)"] * elec_rate) - annual_om

    # Year-by-year payback: cumulate degraded savings until they cover install cost.
    # Cap at 50 years; mark as NaN if savings are too small to ever break even.
    costs    = df["System Cost (₹)"].to_numpy()
    savings0 = df["Financial Savings (₹)"].to_numpy()

    paybacks = []
    for cost, s0 in zip(costs, savings0):
        if s0 < MIN_SAVINGS_THRESHOLD or cost <= 0:
            paybacks.append(float("nan"))
            continue
        cumulative, year = 0.0, 0
        while cumulative < cost and year < 50:
            year      += 1
            cumulative += s0 * ((1 - PANEL_DEGRADATION_RATE) ** (year - 1))
        paybacks.append(float(year) if cumulative >= cost else float("nan"))

    df["Payback Period (Years)"] = paybacks
    return df


def clean_input(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Normalise column names, drop bad rows, and deduplicate building names.
    Returns the cleaned DataFrame plus counts of rows dropped for NaN and negative area.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.title()   # accepts 'area', 'AREA', 'Area', etc.

    df["Area"]  = pd.to_numeric(df["Area"], errors="coerce")
    nan_dropped = int(df["Area"].isna().sum())
    df          = df.dropna(subset=["Area"])

    neg_mask    = df["Area"] <= 0
    neg_dropped = int(neg_mask.sum())
    df          = df[~neg_mask].reset_index(drop=True)

    # Rename duplicates to "Block A (2)", "Block A (3)" so users spot them easily
    if df["Building"].duplicated().any():
        counts = df.groupby("Building").cumcount()
        df["Building"] = df.apply(
            lambda r: f"{r['Building']} ({int(counts[r.name]) + 1})" if counts[r.name] > 0 else r["Building"],
            axis=1,
        )

    return df, nan_dropped, neg_dropped


# ── External API calls — no Streamlit, safe to unit-test independently ────────

# One shared session so we reuse HTTP connections across all API calls
_session = requests.Session()
_session.headers.update({"User-Agent": NOMINATIM_USER_AGENT})


def _request_with_retry(url: str, params: dict, timeout: int = 10) -> requests.Response:
    """GET with simple retry on server-side or rate-limit errors."""
    import time
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = _session.get(url, params=params, timeout=timeout)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(f"Retryable status {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_BASE_DELAY_S * (attempt + 1))
    raise last_exc


def geocode_city(location_query: str) -> tuple[float | None, float | None, str | None]:
    """
    Turn a free-text location into (lat, lon, short_label) via OSM Nominatim.
    Returns (None, None, None) if the lookup fails.
    """
    try:
        resp = _request_with_retry(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_query.strip(), "format": "json", "limit": 1},
            timeout=8,
        )
        data = resp.json()
        if data:
            short_label = data[0]["display_name"].split(",")[0].strip()
            return float(data[0]["lat"]), float(data[0]["lon"]), short_label
    except Exception:
        pass
    return None, None, None


def fetch_annual_irradiance(lat: float, lon: float) -> float:
    """
    Pull annual mean GHI (kWh/m²/day) from NASA POWER.
    Falls back to DEFAULT_IRRADIANCE if the request fails or returns nothing useful.
    """
    try:
        resp = _request_with_retry(
            "https://power.larc.nasa.gov/api/temporal/climatology/point",
            params={
                "parameters": "ALLSKY_SFC_SW_DWN",
                "community":  "RE",
                "longitude":  lon,
                "latitude":   lat,
                "format":     "JSON",
            },
            timeout=8,
        )
        annual_val = resp.json()["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].get("ANN")
        if annual_val and float(annual_val) > 0:
            return float(annual_val)
    except Exception:
        pass
    return DEFAULT_IRRADIANCE


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Solar Insights Dashboard",
    page_icon="☀️",
    layout="wide",
)

st.markdown("<style>footer { visibility: hidden; }</style>", unsafe_allow_html=True)


# Cache geocoding and irradiance so repeat renders don't hit the APIs again
@st.cache_data(show_spinner=False, ttl=86_400)
def _geocode(location_query: str) -> tuple:
    return geocode_city(location_query)


@st.cache_data(show_spinner=False, ttl=86_400)
def _irradiance(lat: float, lon: float) -> float:
    return fetch_annual_irradiance(lat, lon)


with st.sidebar:
    st.title("⚙️ Configuration")

    with st.form("settings_form"):
        st.header("📍 Project Location")
        location_query = st.text_input(
            "City / Location",
            "Ranchi, India",
            help='Type any location, e.g. "Tokyo, Japan" or "Berlin" or "Mumbai"',
        ).strip()

        st.header("💰 Cost Settings")
        elec_rate = st.number_input(
            "Electricity Tariff (₹/kWh)", value=7.50, step=0.10, format="%.2f"
        )

        st.header("🔧 System Parameters")
        efficiency   = st.slider("PV Panel Efficiency (%)",           10, 25,  18) / 100
        loss_factor  = st.slider("System Derate / Loss Factor (%)",    5, 40,  15) / 100
        usage_factor = st.slider("Rooftop Utilisation (%)",           10, 100, 70) / 100

        st.form_submit_button("▶ Update Analysis", use_container_width=True)

    st.header("📂 Upload Rooftop Data")
    uploaded_file = st.file_uploader("Upload Rooftop Dataset (CSV)", type=["csv"])

    if st.button("🔄 Reset Data", use_container_width=True):
        for key in ["uploaded_file"]:
            st.session_state.pop(key, None)
        st.rerun()


st.title("Solar Energy Potential Dashboard")

lat, lon, city_label = _geocode(location_query)

if lat is None:
    st.warning(
        f"⚠️ Could not geocode **{location_query}**. "
        f"Using default irradiance of **{DEFAULT_IRRADIANCE} kWh/m²/day**."
    )
    irradiance = DEFAULT_IRRADIANCE
else:
    irradiance = _irradiance(lat, lon)
    st.success(f"📍 **{city_label}** — Annual Average GHI: **{irradiance:.2f} kWh/m²/day**")

df_raw: pd.DataFrame | None = None

# Prefer data synced from the tracer map; a manual upload overrides it
if "manual_df" in st.session_state:
    df_raw = st.session_state["manual_df"].copy()

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.session_state.pop("manual_df", None)

if df_raw is None:
    st.info(
        "**Getting started:** Upload a CSV via the sidebar. "
        "It must contain a **Building** column and an **Area** column (m²)."
    )
    st.download_button(
        "📄 Download Example Template",
        data=pd.DataFrame({
            "Building": ["Block A", "Block B", "Block C"],
            "Area":     [1000, 2500, 750],
        }).to_csv(index=False),
        file_name="template.csv",
        mime="text/csv",
    )
    st.stop()

df_clean, nan_dropped, neg_dropped = clean_input(df_raw)

if "Building" not in df_clean.columns or "Area" not in df_clean.columns:
    st.error("❌ CSV must contain **Building** and **Area** columns (any capitalisation).")
    st.stop()
if nan_dropped:
    st.warning(f"⚠️ {nan_dropped} row(s) skipped — missing or non-numeric Area.")
if neg_dropped:
    st.warning(f"⚠️ {neg_dropped} row(s) skipped — zero or negative Area values.")
if df_clean.empty:
    st.error("No valid rows remain after cleaning. Please check your dataset.")
    st.stop()

df_result = calculate_solar_metrics(
    df_clean, irradiance, efficiency, loss_factor, usage_factor, elec_rate
)

total_area     = df_result["Area"].sum()
total_capacity = df_result["System Size (kW)"].sum()
total_gen      = df_result["Estimated Annual Energy (kWh)"].sum()
total_co2      = df_result["CO2 Offset (tCO2e)"].sum()
total_savings  = df_result["Financial Savings (₹)"].sum()
total_cost     = df_result["System Cost (₹)"].sum()

portfolio_payback = (
    total_cost / total_savings
    if total_savings >= MIN_SAVINGS_THRESHOLD
    else float("nan")
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Area",              f"{total_area:,.0f} m²")
c2.metric("System Capacity",         f"{total_capacity:,.0f} kWp")
c3.metric("Estimated Annual Energy", f"{total_gen:,.0f} kWh")
c4.metric("Carbon Emission Reduction", f"{total_co2:,.1f} tCO₂e")
c5.metric("Estimated Annual Savings", format_inr(total_savings))
c6.metric("Payback Period", f"{portfolio_payback:.1f} yrs" if not pd.isna(portfolio_payback) else "N/A")

st.subheader("📊 Energy & Financial Projections")
tab1, tab2, tab3 = st.tabs(["Energy Generation", "Financial Impact", "Payback Period"])

# Build once and reuse across all three tabs
chart_base = df_result.set_index("Building")

with tab1:
    st.bar_chart(chart_base["Estimated Annual Energy (kWh)"], color="#1c3d5a")
with tab2:
    st.bar_chart(chart_base["Financial Savings (₹)"], color="#f0a500")
with tab3:
    payback_chart = chart_base["Payback Period (Years)"].dropna()
    if payback_chart.empty:
        st.info("Payback data unavailable — net savings are below the minimum threshold.")
    else:
        st.bar_chart(payback_chart, color="#2e8b57")

st.subheader("📋 System Breakdown")
st.dataframe(
    df_result[[
        "Building", "Area", "System Size (kW)",
        "Estimated Annual Energy (kWh)", "CO2 Offset (tCO2e)",
        "Financial Savings (₹)", "System Cost (₹)", "Payback Period (Years)",
    ]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Area":                          st.column_config.NumberColumn("Area (m²)",        format="%.0f"),
        "System Size (kW)":              st.column_config.NumberColumn("Capacity (kWp)",   format="%.1f"),
        "Estimated Annual Energy (kWh)": st.column_config.NumberColumn("Generation (kWh)", format="%,.0f"),
        "CO2 Offset (tCO2e)":            st.column_config.NumberColumn("CO₂ (tCO₂e)",      format="%.2f"),
        "Financial Savings (₹)":         st.column_config.NumberColumn("Net Savings (₹)",  format="₹%,.0f"),
        "System Cost (₹)":               st.column_config.NumberColumn("Install Cost (₹)", format="₹%,.0f"),
        "Payback Period (Years)":        st.column_config.NumberColumn("Payback (yrs)",     format="%.1f"),
    },
)

st.download_button(
    label="📥 Download Full Analysis (CSV)",
    data=df_result.to_csv(index=False).encode("utf-8"),
    file_name="solar_analysis.csv",
    mime="text/csv",
    use_container_width=True,
)