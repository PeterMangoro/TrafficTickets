import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dateutil import parser

st.set_page_config(page_title="FleetSafe Violations Dashboard", layout="wide")

# ------------- Utility helpers -------------

def read_csv_if_exists(path: str):
	if os.path.exists(path):
		try:
			return pd.read_csv(path)
		except Exception:
			try:
				return pd.read_csv(path, encoding="latin-1")
			except Exception:
				return None
	return None

@st.cache_data(show_spinner=False)
def load_data():
	# Try to load full datasets first
	df_2023 = read_csv_if_exists("Traffic_Tickets_Issued_Window_2023.csv")
	df_2022 = read_csv_if_exists("Traffic_Tickets_Issued_Window_2022.csv")
	weather = read_csv_if_exists("cleaned_weather_dataset.csv")
	
	# If full dataset not available, fall back to sample data
	if df_2023 is None:
		df_2023 = read_csv_if_exists("sample_traffic_tickets_2023.csv")
		if df_2023 is not None:
			st.info("🚀 **Demo Mode**: Using sample 2023 dataset for demonstration. Upload full datasets for complete analysis.")
	
	if df_2022 is None:
		df_2022 = read_csv_if_exists("sample_traffic_tickets_2022.csv")
		if df_2022 is not None:
			st.info("🔍 **Demo Mode**: Using sample 2022 dataset for year-over-year comparison.")
	
	if weather is None:
		weather = read_csv_if_exists("sample_weather_dataset.csv")
		if weather is not None:
			st.info("🌤️ **Demo Mode**: Using sample weather data.")
	
	return df_2022, df_2023, weather


def coerce_datetime(series: pd.Series) -> pd.Series:
	def _parse(x):
		try:
			return parser.parse(str(x))
		except Exception:
			return pd.NaT
	return series.apply(_parse)


def to_month_period(dt_series: pd.Series) -> pd.Series:
	return pd.to_datetime(dt_series).dt.to_period("M").dt.to_timestamp()


def bin_age(age_series: pd.Series) -> pd.Series:
	ages = pd.to_numeric(age_series, errors="coerce")
	bins = [0, 17, 25, 35, 45, 55, 65, 200]
	labels = ["<=17", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
	return pd.cut(ages, bins=bins, labels=labels, right=True, include_lowest=True)


def map_required_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
	out = df.copy()
	for std_col, src in mapping.items():
		if src and src in out.columns:
			out[std_col] = out[src]
		else:
			out[std_col] = np.nan
	return out

# ------------- Load data -------------

df_2022, df_2023, weather = load_data()

st.title("FleetSafe Traffic Violations Dashboard")

if df_2023 is None or df_2023.empty:
	st.error("Traffic_Tickets_Issued_Window_2023.csv not found or is empty. Place it in the project root.")
	st.stop()

st.sidebar.header("Column Mapping")
all_cols = list(df_2023.columns)

# Defaults from known schema if present
_default_violation = "Violation Description" if "Violation Description" in all_cols else None
_default_datetime = "Date" if "Date" in all_cols else None
_default_age = "Age at Violation" if "Age at Violation" in all_cols else ("Age" if "Age" in all_cols else None)
_default_gender = "Gender" if "Gender" in all_cols else None

col_violation = st.sidebar.selectbox("Violation type column", options=[None] + all_cols, index=([None] + all_cols).index(_default_violation) if _default_violation else 0)
col_datetime = st.sidebar.selectbox("Date/Time column (optional)", options=[None] + all_cols, index=([None] + all_cols).index(_default_datetime) if _default_datetime else 0)
col_age = st.sidebar.selectbox("Driver age column", options=[None] + all_cols, index=([None] + all_cols).index(_default_age) if _default_age else 0)
col_gender = st.sidebar.selectbox("Driver gender column", options=[None] + all_cols, index=([None] + all_cols).index(_default_gender) if _default_gender else 0)

# Optional explicit time component mapping if datetime not available
st.sidebar.subheader("If no Date/Time column, map these (optional)")
col_year = st.sidebar.selectbox("Year column", options=[None] + all_cols, index=([None] + all_cols).index("Violation Year") if "Violation Year" in all_cols else 0)
col_month = st.sidebar.selectbox("Month column", options=[None] + all_cols, index=([None] + all_cols).index("Violation Month") if "Violation Month" in all_cols else 0)
col_weekday = st.sidebar.selectbox("Weekday column", options=[None] + all_cols, index=([None] + all_cols).index("Violation Day of Week") if "Violation Day of Week" in all_cols else 0)

# Weather mapping (optional)
weather_cols = list(weather.columns) if weather is not None else []
col_weather_condition = None
col_weather_datetime = None
if weather is not None and len(weather_cols) > 0:
	st.sidebar.subheader("Weather Mapping (optional)")
	# defaults for cleaned_weather_dataset.csv
	_default_wx_cond = "conditions" if "conditions" in weather_cols else ("condition" if "condition" in weather_cols else None)
	_default_wx_dt = "datetime" if "datetime" in weather_cols else ("date" if "date" in weather_cols else None)
	col_weather_condition = st.sidebar.selectbox("Weather condition column", options=[None] + weather_cols, index=([None] + weather_cols).index(_default_wx_cond) if _default_wx_cond else 0)
	col_weather_datetime = st.sidebar.selectbox("Weather Date/Time column", options=[None] + weather_cols, index=([None] + weather_cols).index(_default_wx_dt) if _default_wx_dt else 0)
else:
	st.sidebar.info("cleaned_weather_dataset.csv not found; weather visuals will be limited to tickets-only if needed.")

mapping = {
	"violation": col_violation,
	"datetime": col_datetime,
	"age": col_age,
	"gender": col_gender,
}

base = map_required_columns(df_2023, mapping)

# Parse datetime (if provided)
if pd.notna(base["datetime"]).any():
	base["datetime"] = coerce_datetime(base["datetime"])
	base["year"] = pd.to_datetime(base["datetime"]).dt.year
	base["month"] = to_month_period(base["datetime"])
	base["weekday"] = pd.to_datetime(base["datetime"]).dt.day_name()
else:
	# Fallback to provided split columns
	if col_year and col_year in df_2023.columns:
		base["year"] = pd.to_numeric(df_2023[col_year], errors="coerce")
	if col_month and col_month in df_2023.columns:
		base["month_raw"] = df_2023[col_month]
		# Try to normalize to ordered month labels
		month_map = {
			"jan": 1, "january": 1,
			"feb": 2, "february": 2,
			"mar": 3, "march": 3,
			"apr": 4, "april": 4,
			"may": 5,
			"jun": 6, "june": 6,
			"jul": 7, "july": 7,
			"aug": 8, "august": 8,
			"sep": 9, "september": 9,
			"oct": 10, "october": 10,
			"nov": 11, "november": 11,
			"dec": 12, "december": 12,
		}
		def _to_mon_num(x):
			try:
				n = int(x)
				if 1 <= n <= 12:
					return n
			except Exception:
				pass
			s = str(x).strip().lower()
			return month_map.get(s, np.nan)
		base["month_num"] = base["month_raw"].apply(_to_mon_num)
		# Create ordered month label for plotting
		month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
		base["month_label"] = pd.Categorical(base["month_num"].map(lambda n: month_labels[int(n)-1] if pd.notna(n) else np.nan), categories=month_labels, ordered=True)
	if col_weekday and col_weekday in df_2023.columns:
		# Map numeric weekdays to day names
		weekday_map = {
			1: "Monday", "1": "Monday", "mon": "Monday", "monday": "Monday",
			2: "Tuesday", "2": "Tuesday", "tue": "Tuesday", "tuesday": "Tuesday",
			3: "Wednesday", "3": "Wednesday", "wed": "Wednesday", "wednesday": "Wednesday",
			4: "Thursday", "4": "Thursday", "thu": "Thursday", "thursday": "Thursday",
			5: "Friday", "5": "Friday", "fri": "Friday", "friday": "Friday",
			6: "Saturday", "6": "Saturday", "sat": "Saturday", "saturday": "Saturday",
			7: "Sunday", "7": "Sunday", "sun": "Sunday", "sunday": "Sunday",
		}
		def _map_weekday(x):
			s = str(x).strip().lower()
			return weekday_map.get(s, weekday_map.get(int(s) if s.isdigit() else s, np.nan))
		base["weekday"] = df_2023[col_weekday].apply(_map_weekday)

# Add age groups and normalize strings
if pd.notna(base["age"]).any():
	base["age_group"] = bin_age(base["age"]) 
else:
	base["age_group"] = np.nan

if pd.notna(base["gender"]).any():
	base["gender"] = base["gender"].astype(str).str.title()

if pd.notna(base["violation"]).any():
	base["violation"] = base["violation"].astype(str).str.strip()

# Merge weather if available
tickets = base.copy()
if weather is not None and col_weather_condition and col_weather_datetime and col_weather_condition in weather.columns and col_weather_datetime in weather.columns:
	wx = weather[[col_weather_datetime, col_weather_condition]].copy()
	wx[col_weather_datetime] = coerce_datetime(wx[col_weather_datetime])
	wx["wx_day"] = pd.to_datetime(wx[col_weather_datetime]).dt.date

	# Create day column from tickets data (handle both datetime and split columns)
	if "datetime" in tickets.columns and pd.notna(tickets["datetime"]).any():
		tickets["day"] = pd.to_datetime(tickets["datetime"]).dt.date
	elif col_year and col_month and col_year in df_2023.columns and col_month in df_2023.columns:
		# Create a synthetic date from year/month columns
		year_col = df_2023[col_year].astype(str)
		month_col = df_2023[col_month].astype(str)
		# Create first of month dates for grouping
		synthetic_dates = pd.to_datetime(year_col + "-" + month_col + "-01", errors="coerce")
		tickets["day"] = synthetic_dates.dt.date
	else:
		tickets["day"] = pd.to_datetime("2023-01-01").date()  # fallback

	# Ensure both day columns are the same type before merge
	wx_days_clean = wx[["wx_day", col_weather_condition]].dropna().drop_duplicates("wx_day")
	if not wx_days_clean.empty and "day" in tickets.columns:
		tickets = tickets.merge(wx_days_clean, left_on="day", right_on="wx_day", how="left")
		tickets.rename(columns={col_weather_condition: "weather_condition"}, inplace=True)
	else:
		tickets["weather_condition"] = np.nan
else:
	tickets["weather_condition"] = np.nan

# Layout
with st.expander("Data preview", expanded=False):
	st.dataframe(tickets.head(50))

# -------------------- Visual 1: 2022 vs 2023 line --------------------
st.subheader("Violations: 2022 vs 2023")

def normalize_to_month_names(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
	"""Convert various month formats to consistent month names"""
	month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
	
	# Try different month column formats
	month_data = None
	
	# Try datetime column first
	if mapping.get("datetime") and mapping["datetime"] in df.columns:
		try:
			datetime_col = coerce_datetime(df[mapping["datetime"]])
			month_data = pd.to_datetime(datetime_col).dt.month.apply(lambda x: month_labels[x-1] if pd.notna(x) else np.nan)
		except Exception:
			pass
	
	# Try Violation Month column
	if month_data is None and "Violation Month" in df.columns:
		def _to_mon_name(x):
			s = str(x).strip().lower()
			month_map = {
				"jan":0,"january":0,"1":0,
				"feb":1,"february":1,"2":1,
				"mar":2,"march":2,"3":2,
				"apr":3,"april":3,"4":3,
				"may":4,"5":4,
				"jun":5,"june":5,"6":5,
				"jul":6,"july":6,"7":6,
				"aug":7,"august":7,"8":7,
				"sep":8,"september":8,"9":8,
				"oct":9,"october":9,"10":9,
				"nov":10,"november":10,"11":10,
				"dec":11,"december":11,"12":11
			}
			try:
				return month_labels[month_map[s]] if s in month_map else None
			except Exception:
				return None
		
		month_data = df["Violation Month"].apply(_to_mon_name)
	
	return month_data

# Process 2023 data
counts_2023 = None
if pd.notna(tickets.get("violation", pd.Series([pd.NA]))).any():
	month_names_2023 = normalize_to_month_names(tickets, mapping)
	if month_names_2023 is not None:
		counts_2023 = pd.DataFrame({"month": month_names_2023}).groupby("month").size().reset_index(name="count")
		counts_2023["year"] = "2023"

# Process 2022 data
counts_2022 = pd.DataFrame(columns=["month", "count", "year"])
if df_2022 is not None and not df_2022.empty:
	month_names_2022 = normalize_to_month_names(df_2022, mapping)
	if month_names_2022 is not None:
		counts_2022 = pd.DataFrame({"month": month_names_2022}).groupby("month").size().reset_index(name="count")
		counts_2022["year"] = "2022"

# Combine and plot
if counts_2023 is not None and not counts_2023.empty:
	line_df = pd.concat([counts_2022, counts_2023], ignore_index=True)
	if not line_df.empty:
		# Ensure proper month ordering
		month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
		line_df["month"] = pd.Categorical(line_df["month"], categories=month_order, ordered=True)
		line_df = line_df.sort_values("month")
		
		fig = px.line(line_df, x="month", y="count", color="year", markers=True, 
		             title="Monthly Violations Comparison: 2022 vs 2023",
		             color_discrete_map={"2022": "#FF6B6B", "2023": "#4ECDC4"})
		fig.update_layout(
			xaxis_title="Month", 
			yaxis_title="Number of Violations",
			font=dict(color="#ffffff"),  # White text for dark theme
			plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
			paper_bgcolor="rgba(0,0,0,0)",  # Transparent overall background
			legend=dict(
				bgcolor="rgba(27,31,42,0.9)",  # Dark background matching sidebar
				bordercolor="rgba(255,255,255,0.2)",  # Subtle border
				font=dict(color="#ffffff")  # White text in legend
			),
			xaxis=dict(color="#ffffff"),  # White axis text
			yaxis=dict(color="#ffffff")   # White axis text
		)
		st.plotly_chart(fig, use_container_width=True)
	else:
		st.info("No time series data available.")
else:
	st.info("Map violation and date/month columns to enable comparison.")

# -------------------- Visual 2: Top 5 violation types by month --------------------
st.subheader("Top 5 Violations by Month (Seasonality)")
violation_col = tickets.get("violation")
month_col = None
if "month" in tickets.columns:
	month_col = "month"
elif "month_label" in tickets.columns:
	month_col = "month_label"

if violation_col is not None and pd.notna(violation_col).any() and month_col and pd.notna(tickets[month_col]).any():
	top5 = tickets["violation"].value_counts().head(5).index.tolist()
	t5 = tickets[tickets["violation"].isin(top5)]
	monthly = t5.groupby([month_col, "violation"]).size().reset_index(name="count").rename(columns={month_col:"x"})
	fig2 = px.line(monthly, x="x", y="count", color="violation", markers=True,
	              title="Top 5 Violation Types: Seasonal Trends")
	fig2.update_layout(
		font=dict(color="#ffffff"),
		plot_bgcolor="rgba(0,0,0,0)",
		paper_bgcolor="rgba(0,0,0,0)",
		legend=dict(
			bgcolor="rgba(27,31,42,0.9)",
			bordercolor="rgba(255,255,255,0.2)",
			font=dict(color="#ffffff")
		),
		xaxis=dict(color="#ffffff"),
		yaxis=dict(color="#ffffff")
	)
	st.plotly_chart(fig2, use_container_width=True)
else:
	st.info("Need mapped violation and a month field for this visual.")

# -------------------- Visual 3: Weather vs violations --------------------
st.subheader("Weather Condition vs Violations")
if pd.notna(tickets.get("weather_condition")).any():
	wx_counts = tickets.groupby("weather_condition").size().reset_index(name="count").sort_values("count", ascending=False)
	fig3 = px.bar(wx_counts, x="weather_condition", y="count",
	             title="Weather Conditions vs Number of Violations")
	fig3.update_layout(
		font=dict(color="#ffffff"),
		plot_bgcolor="rgba(0,0,0,0)",
		paper_bgcolor="rgba(0,0,0,0)",
		xaxis=dict(color="#ffffff"),
		yaxis=dict(color="#ffffff")
	)
	st.plotly_chart(fig3, use_container_width=True)
else:
	st.info("Weather data not available or not mapped.")

# -------------------- Visual 4: Violations by Weekday with average guidance --------------------
st.subheader("Violations by Weekday (vs Average)")
weekday_col = tickets.get("weekday")
if weekday_col is not None and pd.notna(weekday_col).any():
	weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	wd = tickets.groupby("weekday").size().reset_index(name="count")
	wd["weekday"] = pd.Categorical(wd["weekday"], categories=weekday_order, ordered=True)
	wd = wd.sort_values("weekday")
	avg_count = wd["count"].mean() if not wd.empty else 0
	fig4 = px.bar(wd, x="weekday", y="count",
	             title="Violations by Day of Week (vs Average)")
	fig4.add_hline(y=avg_count, line_dash="dash", line_color="#FFA500", 
	              annotation_text=f"Average: {avg_count:.0f}",
	              annotation=dict(font=dict(color="#ffffff")))
	fig4.update_layout(
		font=dict(color="#ffffff"),
		plot_bgcolor="rgba(0,0,0,0)",
		paper_bgcolor="rgba(0,0,0,0)",
		xaxis=dict(color="#ffffff"),
		yaxis=dict(color="#ffffff")
	)
	st.plotly_chart(fig4, use_container_width=True)
else:
	st.info("Map a valid weekday field to compute weekday counts.")

# -------------------- Visual 5: Heatmap age group vs violation --------------------
st.subheader("Heatmap: Age Group vs Violation Type")
if pd.notna(tickets.get("age_group")).any() and pd.notna(tickets.get("violation")).any():
	hm = tickets.groupby(["age_group", "violation"]).size().reset_index(name="count")
	pivot = hm.pivot(index="age_group", columns="violation", values="count").fillna(0)
	fig5 = px.imshow(pivot, aspect="auto", color_continuous_scale="Plasma",
	                title="Heatmap: Age Group vs Violation Type Patterns")
	fig5.update_layout(
		font=dict(color="#ffffff"),
		plot_bgcolor="rgba(0,0,0,0)",
		paper_bgcolor="rgba(0,0,0,0)",
		xaxis=dict(color="#ffffff"),
		yaxis=dict(color="#ffffff")
	)
	st.plotly_chart(fig5, use_container_width=True)
else:
	st.info("Need age and violation mappings for heatmap.")

# -------------------- Visual 6: Stacked bar by age group and gender --------------------
st.subheader("Stacked Bar: Violations by Age Group and Gender")
if pd.notna(tickets.get("age_group")).any() and pd.notna(tickets.get("gender")).any():
	ag = tickets.groupby(["age_group", "gender"]).size().reset_index(name="count")
	fig6 = px.bar(ag, x="age_group", y="count", color="gender", barmode="stack",
	             title="Violations by Age Group (Stacked by Gender)")
	fig6.update_layout(
		xaxis_title="Age Group", 
		yaxis_title="Violations",
		font=dict(color="#ffffff"),
		plot_bgcolor="rgba(0,0,0,0)",
		paper_bgcolor="rgba(0,0,0,0)",
		legend=dict(
			bgcolor="rgba(27,31,42,0.9)",
			bordercolor="rgba(255,255,255,0.2)",
			font=dict(color="#ffffff")
		),
		xaxis=dict(color="#ffffff"),
		yaxis=dict(color="#ffffff")
	)
	st.plotly_chart(fig6, use_container_width=True)
	with st.expander("Insight & Action"):
		st.write("If the highest bar appears at 26-35, prioritize targeted, gender-aware training for this group.")
else:
	st.info("Need age and gender mappings for stacked bar.")

st.caption("Tip: Use the sidebar to map columns if your dataset uses different names.")
