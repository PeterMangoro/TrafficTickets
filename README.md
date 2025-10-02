# FleetSafe Traffic Violations Analytics

**Goal**: Reduce road traffic violations and improve road safety through targeted training and awareness campaigns using data-driven insights.

## 📊 Data Access & Deployment

### Demo Data Included
The repository includes sample datasets for testing and demonstration:
- `sample_traffic_tickets_2023.csv`: 500 sample records
- `sample_weather_dataset.csv`: 50 sample weather records

### Production Data
Full datasets are **not included** in this repository due to size constraints (~280MB files).
- `Traffic_Tickets_Issued_Window_2023.csv`: Full 2023 dataset
- `cleaned_weather_dataset.csv`: Full weather dataset

### Deployment Options

#### **Option 1: Streamlit Cloud (Free Hosting)**
1. **Fork this repository**
2. **Upload your data files** to your fork:
   - Add `Traffic_Tickets_Issued_Window_2023.csv` to repository root
   - Add `cleaned_weather_dataset.csv` to repository root
3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repo
   - Select `app.py` as main file

#### **Option 2: Cloud Storage Integration (Recommended)**
For privacy or when datasets are too large:

```python
# Example: Load from AWS S3
import streamlit as st
import pandas as pd
import boto3

def load_data_from_s3():
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='your-bucket', Key='data/Traffic_Tickets_2023.csv')
    return pd.read_csv(obj['Body'])

# In app.py, replace the local file loading with:
# df_2023 = load_data_from_s3() if st.secrets.get("aws_enabled") else pd.read_csv("local_file.csv")
```

#### **Option 3: Dataset Links**
```python
# Example: Load from public URLs
DATA_URLS = {
    "traffic_2023": "https://your-storage.com/traffic_tickets_2023.csv",
    "weather": "https://your-storage.com/weather_data.csv"
}

def load_from_url(url):
    return pd.read_csv(url)
```

### Project Structure
- `app.py`: Streamlit dashboard with the requested visuals
- `notebooks/eda.ipynb`: Exploratory data analysis notebook
- `sample_traffic_tickets_2023.csv`: **Sample data for demos** (included in repo)
- `sample_weather_dataset.csv`: **Sample weather data for demos** (included in repo)
- `create_sample_data.py`: Script to generate sample datasets
- `.streamlit/config.toml`: Dark theme config
- `FleetSafe_Blog_Post.md`: Comprehensive project documentation

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
