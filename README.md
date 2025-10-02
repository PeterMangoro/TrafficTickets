### FleetSafe Traffic Violations Analytics

**Goal**: Reduce road traffic violations and improve road safety through targeted training and awareness campaigns using data-driven insights.

### Project Structure
- `app.py`: Streamlit dashboard with the requested visuals
- `notebooks/eda.ipynb`: Exploratory data analysis notebook
- `Traffic_Tickets_Issued_Window_2023.csv`: Tickets dataset (provided)
- `cleaned_weather_dataset.csv`: Weather dataset (optional; merged by date/time if present)
- `intro.txt`: Project intro copy (optional)
- `.streamlit/config.toml`: Dark theme config
- `BLOG.md`: Project write-up

### Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Dashboard
```bash
streamlit run app.py
```

### Data Expectations and Column Mapping
The app includes a sidebar to map your columns. Expected logical fields:
- Date/time of violation
- Violation type (offense description/code)
- Driver age
- Driver gender
- Weather condition (from joined weather data)

If your column names differ, use the sidebar to map them. The app will persist choices in the session.

### Visuals in the Dashboard
- Line chart: Compare violation counts 2022 vs 2023
- Line chart: Top 5 violations by month (seasonality)
- Bar chart: Weather condition vs number of violations
- Bar chart: Which days have the highest violations (and weekday > average guidance)
- Heatmap: Age group vs violation type
- Stacked bar: Violations by age group stacked by gender

### Notebook (EDA)
The notebook explores:
- Data loading and sanity checks
- Missing values and distributions
- Temporal patterns (monthly, weekday)
- Violation types and age/gender breakdowns
- Optional merge with weather by date/time (day-level join fallback)

### Notes
- If a 2022 dataset exists (e.g., `Traffic_Tickets_Issued_Window_2022.csv`), the app will detect it automatically; otherwise it will display a note.
- If `cleaned_weather_dataset.csv` exists, the app and notebook will include weather analyses; otherwise a note is shown.

### License
MIT
