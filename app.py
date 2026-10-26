from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Commerce Pulse | Sales Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#172433"
MUTED = "#708090"
TEAL = "#0E9F91"
BLUE = "#4C78D0"
PALETTE = ["#0E9F91", "#4C78D0", "#F2A541", "#8F72C9", "#E26D78", "#4E9B72", "#74A9CF"]

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root {{ --ink:{INK}; --muted:{MUTED}; --teal:{TEAL}; }}
      html, body, [class*="css"] {{ font-family:'DM Sans', sans-serif; color:var(--ink); }}
      .stApp {{ background:linear-gradient(180deg,#f5f8fb 0%,#f7f9fb 44%,#f3f7f8 100%); }}
      [data-testid="stHeader"] {{ background:rgba(245,248,251,.85); }}
      .block-container {{ padding-top:1.7rem; padding-bottom:3rem; max-width:1440px; }}
      h1,h2,h3 {{ font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:#172433; }}
      h1 {{ font-weight:800; }}
      [data-testid="stMetric"] {{ background:#fff; border:1px solid #e7edf1; border-radius:16px; padding:18px 20px; box-shadow:0 5px 18px rgba(23,36,51,.035); }}
      [data-testid="stMetricLabel"] {{ color:#718091; font-weight:600; }}
      [data-testid="stMetricValue"] {{ font-family:'Manrope',sans-serif; color:#172433; font-weight:800; }}
      div[data-testid="stTabs"] button {{ font-weight:650; }}
      div[data-testid="stTabs"] button[aria-selected="true"] {{ color:#0E9F91; }}
      .hero {{ background:linear-gradient(110deg,#102b3d,#154f5a 61%,#148a7d); color:#fff; padding:28px 32px; border-radius:22px; margin-bottom:22px; box-shadow:0 12px 35px rgba(19,64,76,.16); }}
      .hero h1 {{ color:#fff; margin:0; font-size:2.15rem; }}
      .hero p {{ color:#c5dfdf; margin:.5rem 0 0; font-size:1rem; }}
      .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#94d9cf; margin-bottom:9px; }}
      .section-note {{ color:#718091; font-size:.9rem; margin-top:-.55rem; margin-bottom:1.1rem; }}
      .insight {{ border-left:3px solid #0E9F91; border-radius:8px; background:#fff; padding:14px 16px; box-shadow:0 4px 16px rgba(23,36,51,.04); min-height:90px; }}
      .insight b {{ display:block; color:#172433; margin-bottom:4px; }}
      .insight span {{ color:#687988; font-size:.9rem; }}
      .source-pill {{ display:inline-block; background:#e5f5f2; color:#087d72; padding:4px 10px; border-radius:99px; font-size:.75rem; font-weight:700; }}
      [data-testid="stSidebar"] {{ background:#f0f5f6; border-right:1px solid #e3ebed; }}
      .small-muted {{ color:#7b8996; font-size:.8rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def make_demo_data(rows: int = 1100) -> pd.DataFrame:
    """Deterministic demo data; clearly labeled so it is never mistaken for the class CSV."""
    rng = np.random.default_rng(17)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    date = rng.choice(dates, rows)
    categories = np.array(["Electronics", "Home & Kitchen", "Fashion", "Beauty", "Sports", "Books", "Toys"])
    regions = np.array(["North", "South", "East", "West", "Central"])
    category = rng.choice(categories, rows, p=[.17,.18,.2,.14,.12,.1,.09])
    price = pd.Series(category).map({"Electronics":165,"Home & Kitchen":72,"Fashion":48,"Beauty":33,"Sports":61,"Books":24,"Toys":39}).to_numpy()
    units = rng.integers(1, 9, rows)
    seasonal = 1 + .18 * np.sin((pd.DatetimeIndex(date).dayofyear - 35) * 2 * np.pi / 365)
    revenue = np.maximum(5, price * units * seasonal * rng.lognormal(0, .3, rows))
    ad_spend = revenue * rng.uniform(.035,.17,rows)
    margin = rng.normal(.24,.07,rows) - ad_spend / revenue * .3
    profit = revenue * margin
    return pd.DataFrame({
        "Date": pd.to_datetime(date), "Year": pd.DatetimeIndex(date).year,
        "Month": pd.DatetimeIndex(date).strftime("%Y-%m"), "Category": category,
        "Region": rng.choice(regions, rows), "Revenue": revenue.round(2),
        "Profit": profit.round(2), "Ad_Spend": ad_spend.round(2), "Units_Sold": units,
    })


def find_local_csv() -> Path | None:
    candidates = [Path("ecommerce_sales_data.csv"), Path("data/ecommerce_sales_data.csv")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(payload))


def key_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def match_column(df: pd.DataFrame, options: list[str]) -> str | None:
    mapping = {key_name(col): col for col in df.columns}
    for option in options:
        if key_name(option) in mapping:
            return mapping[key_name(option)]
    return None


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = raw.copy()
    aliases = {
        "Revenue": ["Revenue", "Sales", "Total Sales", "Order Value", "Amount"],
        "Profit": ["Profit", "Net Profit", "Gross Profit"],
        "Units_Sold": ["Units_Sold", "Units Sold", "Quantity", "Qty"],
        "Ad_Spend": ["Ad_Spend", "Ad Spend", "Marketing Spend", "Advertising Cost"],
        "Category": ["Category", "Product Category", "Product"],
        "Region": ["Region", "Market", "State", "Country", "Location"],
    }
    resolved = {standard: match_column(df, choices) for standard, choices in aliases.items()}
    for standard, original in resolved.items():
        if original and original != standard:
            df[standard] = df[original]
    for column in ["Revenue", "Profit", "Units_Sold", "Ad_Spend"]:
        if column in df:
            df[column] = pd.to_numeric(df[column].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")

    date_col = match_column(df, ["Date", "Order Date", "Transaction Date", "Datetime", "Timestamp"])
    parsed_date = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Series(pd.NaT, index=df.index)
    if parsed_date.notna().sum() == 0:
        year_col = match_column(df, ["Year"])
        month_col = match_column(df, ["Month"])
        if year_col and month_col:
            month_values = df[month_col]
            # Month may be a full date in the source notebook or a month name/number.
            month_dates = pd.to_datetime(month_values, errors="coerce")
            if month_dates.notna().sum() == 0:
                month_num = pd.to_numeric(month_values, errors="coerce")
                if month_num.notna().sum() == 0:
                    month_num = month_values.astype(str).str.strip().str[:3].str.title().map({m:i for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1)})
                year_num = pd.to_numeric(df[year_col], errors="coerce")
                parsed_date = pd.to_datetime(dict(year=year_num, month=month_num, day=1), errors="coerce")
            else:
                parsed_date = month_dates
                # The notebook date parsing uses a timestamp format for Year and Month;
                # a year-only column may still provide the year if the month has no year.
                years = pd.to_numeric(df[year_col], errors="coerce")
                year_only = years.between(1900, 2200)
                missing_year = parsed_date.notna() & parsed_date.dt.year.eq(1970) & year_only
                parsed_date.loc[missing_year] = pd.to_datetime(
                    {"year": years[missing_year], "month": parsed_date.loc[missing_year].dt.month, "day": 1}, errors="coerce"
                )
    df["_Date"] = parsed_date
    if "Category" not in df:
        df["Category"] = "All products"
    else:
        df["Category"] = df["Category"].fillna("Unspecified").astype(str)
    if "Region" not in df:
        df["Region"] = "Unspecified"
    else:
        df["Region"] = df["Region"].fillna("Unspecified").astype(str)
    return df, resolved


def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    sign = "−" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}${value/1_000_000:,.2f}M"
    if value >= 10_000:
        return f"{sign}${value/1_000:,.1f}K"
    return f"{sign}${value:,.0f}"


def style_chart(fig, height: int = 360):
    fig.update_layout(
        height=height, margin=dict(l=10,r=10,t=28,b=10),
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#172433", font_color="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf1f4", zeroline=False, linecolor="#e5ebef")
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f4", zeroline=False, linecolor="#e5ebef")
    return fig


def card(title: str, body: str):
    st.markdown(f'<div class="insight"><b>{title}</b><span>{body}</span></div>', unsafe_allow_html=True)


st.sidebar.markdown("## ◈ Commerce Pulse")
st.sidebar.caption("E-commerce performance explorer")
uploaded = st.sidebar.file_uploader("Load your sales CSV", type=["csv"], help="Upload ecommerce_sales_data.csv to replace the demo data.")
local_path = find_local_csv()
if uploaded is not None:
    raw_df = read_csv_bytes(uploaded.getvalue())
    source_label = f"Uploaded · {uploaded.name}"
    is_demo = False
elif local_path:
    raw_df = pd.read_csv(local_path)
    source_label = f"Local file · {local_path.name}"
    is_demo = False
else:
    raw_df = make_demo_data()
    source_label = "Illustrative demo data"
    is_demo = True

df, columns_found = prepare_data(raw_df)

st.sidebar.markdown("---")
st.sidebar.markdown("### Slice the data")
categories = sorted(df["Category"].dropna().unique().tolist())
regions = sorted(df["Region"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
filtered = df[df["Category"].isin(selected_categories) & df["Region"].isin(selected_regions)].copy()
dates = df["_Date"].dropna()
if not dates.empty:
    min_date, max_date = dates.min().date(), dates.max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered["_Date"].isna() | filtered["_Date"].between(start, end)]
st.sidebar.markdown("---")
st.sidebar.markdown(f'<span class="source-pill">{source_label}</span>', unsafe_allow_html=True)
if is_demo:
    st.sidebar.info("The notebook’s CSV is not in this folder yet. Charts currently use clearly labeled sample data. Upload your CSV here to see its actual results.")

st.markdown(
    f'<div class="hero"><div class="eyebrow">Sales intelligence · interactive report</div><h1>Commerce Pulse</h1><p>Explore revenue, profit, product mix, regional performance, and marketing efficiency.</p></div>',
    unsafe_allow_html=True,
)
if is_demo:
    st.info("Preview mode: this is illustrative sample data, not the class dataset. Upload `ecommerce_sales_data.csv` in the sidebar to replace it with the real figures.", icon="ℹ️")
if filtered.empty:
    st.warning("No rows match these filters. Select at least one category and region, or widen the date range.")
    st.stop()

revenue_total = filtered["Revenue"].sum(min_count=1) if "Revenue" in filtered else np.nan
profit_total = filtered["Profit"].sum(min_count=1) if "Profit" in filtered else np.nan
units_total = filtered["Units_Sold"].sum(min_count=1) if "Units_Sold" in filtered else np.nan
rows_count = len(filtered)
margin_pct = (profit_total / revenue_total * 100) if pd.notna(profit_total) and pd.notna(revenue_total) and revenue_total else np.nan
spend_total = filtered["Ad_Spend"].sum(min_count=1) if "Ad_Spend" in filtered else np.nan
roas = revenue_total / spend_total if pd.notna(spend_total) and spend_total > 0 else np.nan

st.markdown(f'<span class="source-pill">{rows_count:,} records in current view</span>', unsafe_allow_html=True)
kpis = st.columns(5)
kpis[0].metric("Revenue", money(revenue_total))
kpis[1].metric("Profit", money(profit_total), f"{margin_pct:.1f}% margin" if pd.notna(margin_pct) else None)
kpis[2].metric("Units sold", f"{units_total:,.0f}" if pd.notna(units_total) else "—")
kpis[3].metric("Average order row", money(revenue_total / rows_count) if pd.notna(revenue_total) and rows_count else "—")
kpis[4].metric("Revenue / ad spend", f"{roas:.2f}×" if pd.notna(roas) else "—")

tabs = st.tabs(["Overview", "Products & regions", "Marketing", "Data explorer"])

with tabs[0]:
    st.subheader("Performance at a glance")
    st.markdown('<div class="section-note">The selected filters apply to every chart and table in this report.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65,1])
    with left:
        if "Revenue" in filtered and filtered["_Date"].notna().any():
            trend_measures = {"Revenue": ("Revenue", "sum")}
            if "Profit" in filtered:
                trend_measures["Profit"] = ("Profit", "sum")
            if "Units_Sold" in filtered:
                trend_measures["Units"] = ("Units_Sold", "sum")
            trend = filtered.dropna(subset=["_Date"]).set_index("_Date").resample("MS").agg(**trend_measures).reset_index()
            measure = st.selectbox("Trend metric", [x for x in ["Revenue","Profit","Units"] if x in trend], key="trend_metric")
            fig = px.area(trend, x="_Date", y=measure, color_discrete_sequence=[TEAL], markers=True)
            fig.update_traces(line_width=2.5, fillcolor="rgba(14,159,145,.12)", hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>")
            fig.update_layout(title=f"Monthly {measure.lower()} trend", xaxis_title=None, yaxis_title=measure)
            st.plotly_chart(style_chart(fig, 390), use_container_width=True)
        elif "Revenue" in filtered:
            st.caption("No valid date column was found, so a time trend is unavailable.")
            metric = st.selectbox("Compare", [c for c in ["Revenue","Profit","Units_Sold"] if c in filtered], key="trend_fallback")
            chart = filtered.groupby("Category", as_index=False)[metric].sum().sort_values(metric, ascending=False)
            st.plotly_chart(style_chart(px.bar(chart, x="Category",y=metric,color="Category",color_discrete_sequence=PALETTE,title=f"{metric} by category"),390),use_container_width=True)
        else:
            st.warning("Revenue column not found. Check the uploaded file headers.")
    with right:
        if "Revenue" in filtered:
            mix = filtered.groupby("Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
            fig = px.pie(mix, values="Revenue", names="Category", hole=.66, color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>%{value:$,.0f}<extra></extra>")
            fig.update_layout(title="Revenue mix", showlegend=True, legend=dict(orientation="v",x=1.02,y=.5))
            st.plotly_chart(style_chart(fig,390),use_container_width=True)
    st.markdown("#### Quick reads")
    insights = []
    if "Revenue" in filtered and not filtered.empty:
        cat_rev = filtered.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
        if len(cat_rev):
            share = cat_rev.iloc[0] / cat_rev.sum() * 100 if cat_rev.sum() else 0
            insights.append(("Leading category", f"{cat_rev.index[0]} contributes {share:.1f}% of filtered revenue."))
        if filtered["_Date"].notna().any():
            monthly = filtered.dropna(subset=["_Date"]).assign(_Month=lambda x:x["_Date"].dt.to_period("M")).groupby("_Month")["Revenue"].sum()
            if len(monthly) >= 2:
                delta = (monthly.iloc[-1]/monthly.iloc[-2]-1)*100 if monthly.iloc[-2] else np.nan
                insights.append(("Latest month", f"Revenue changed {delta:+.1f}% vs. the preceding month." if pd.notna(delta) else "Previous month revenue was zero."))
    if pd.notna(margin_pct):
        insights.append(("Profitability", f"Overall profit margin for this selection is {margin_pct:.1f}%."))
    if pd.notna(roas):
        insights.append(("Ad return", f"Each $1 of ad spend corresponds to ${roas:.2f} in revenue."))
    if not insights:
        insights.append(("Ready to explore", "Choose filters or upload a dataset with revenue, profit, or date fields for more insights."))
    for start_i in range(0, len(insights), 4):
        cols = st.columns(min(4,len(insights)-start_i))
        for col, (title, body) in zip(cols, insights[start_i:start_i+4]):
            with col: card(title,body)

with tabs[1]:
    st.subheader("Product and regional performance")
    c1,c2 = st.columns(2)
    with c1:
        metric_options = [x for x in ["Revenue","Profit","Units_Sold"] if x in filtered]
        if metric_options:
            product_metric = st.selectbox("Rank categories by", metric_options, key="product_metric")
            products = filtered.groupby("Category",as_index=False)[product_metric].sum().sort_values(product_metric,ascending=True)
            fig=px.bar(products.tail(15),x=product_metric,y="Category",orientation="h",color=product_metric,color_continuous_scale=["#c9eeea",TEAL],title=f"Category ranking · {product_metric}")
            fig.update_layout(coloraxis_showscale=False,xaxis_title=product_metric,yaxis_title=None)
            st.plotly_chart(style_chart(fig,430),use_container_width=True)
    with c2:
        metric_region = st.selectbox("Regional metric", metric_options, key="region_metric") if metric_options else None
        if metric_region:
            region_data=filtered.groupby("Region",as_index=False)[metric_region].sum().sort_values(metric_region,ascending=False)
            fig=px.bar(region_data,x="Region",y=metric_region,color="Region",color_discrete_sequence=PALETTE,title=f"Regional comparison · {metric_region}")
            fig.update_layout(showlegend=False,xaxis_title=None)
            st.plotly_chart(style_chart(fig,430),use_container_width=True)
    if "Profit" in filtered and "Revenue" in filtered:
        st.markdown("#### Revenue and profit by category")
        comp=filtered.groupby("Category",as_index=False)[["Revenue","Profit"]].sum().melt(id_vars="Category",var_name="Measure",value_name="Amount")
        fig=px.bar(comp,x="Category",y="Amount",color="Measure",barmode="group",color_discrete_map={"Revenue":BLUE,"Profit":TEAL})
        fig.update_layout(xaxis_title=None,yaxis_title="Amount")
        st.plotly_chart(style_chart(fig,360),use_container_width=True)

with tabs[2]:
    st.subheader("Marketing efficiency")
    if "Ad_Spend" not in filtered or "Revenue" not in filtered:
        st.info("Marketing view needs both an ad spend field and a revenue field. Expected headers include `Ad_Spend` and `Revenue`.")
    else:
        c1,c2=st.columns([1.5,1])
        with c1:
            if "Category" in filtered:
                points=filtered.groupby("Category",as_index=False).agg(Revenue=("Revenue","sum"),Ad_Spend=("Ad_Spend","sum"),Profit=("Profit","sum") if "Profit" in filtered else ("Revenue","sum"))
            else: points=filtered
            fig=px.scatter(points,x="Ad_Spend",y="Revenue",size="Profit" if "Profit" in filtered else None,color="Category" if "Category" in points else None,hover_name="Category" if "Category" in points else None,color_discrete_sequence=PALETTE,title="Revenue vs. advertising spend")
            fig.update_layout(xaxis_title="Ad spend",yaxis_title="Revenue")
            st.plotly_chart(style_chart(fig,410),use_container_width=True)
        with c2:
            spend_by=filtered.groupby("Category",as_index=False).agg(Ad_Spend=("Ad_Spend","sum"),Revenue=("Revenue","sum"))
            spend_by["ROAS"]=spend_by["Revenue"]/spend_by["Ad_Spend"].replace(0,np.nan)
            fig=px.bar(spend_by.sort_values("ROAS",ascending=True),x="ROAS",y="Category",orientation="h",color="ROAS",color_continuous_scale=["#f4c36d",TEAL],title="Revenue / ad spend by category")
            fig.update_layout(coloraxis_showscale=False,xaxis_title="Revenue per $1 ad spend",yaxis_title=None)
            st.plotly_chart(style_chart(fig,410),use_container_width=True)
        st.caption("ROAS is an aggregate ratio (revenue ÷ ad spend), not a causal estimate of advertising impact.")

with tabs[3]:
    st.subheader("Explore the underlying rows")
    st.markdown('<div class="section-note">Filter the view in the sidebar, sort columns, and download the rows currently in scope.</div>',unsafe_allow_html=True)
    q=st.text_input("Search visible values",placeholder="Type a product, category, region, or other value…")
    view=filtered.drop(columns=["_Date"],errors="ignore").copy()
    if q:
        mask=view.astype(str).apply(lambda col:col.str.contains(q,case=False,na=False)).any(axis=1)
        view=view[mask]
    st.caption(f"Showing {len(view):,} of {len(filtered):,} matching rows")
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)
    st.download_button("Download filtered CSV",data=view.to_csv(index=False).encode("utf-8"),file_name="commerce_pulse_filtered.csv",mime="text/csv")
    st.markdown("#### Data quality snapshot")
    quality=pd.DataFrame({"Column":raw_df.columns,"Missing values":raw_df.isna().sum().values,"Missing %":(raw_df.isna().mean().values*100).round(1),"Distinct values":raw_df.nunique(dropna=True).values})
    st.dataframe(quality,use_container_width=True,hide_index=True)
    if columns_found["Revenue"] is None:
        st.warning("No revenue field was detected. Add a column named Revenue (or Sales / Amount) to enable sales KPIs.")

st.markdown("<div class='small-muted' style='text-align:center;margin-top:32px'>Commerce Pulse · Built from the class notebook’s e-commerce analysis themes</div>",unsafe_allow_html=True)
