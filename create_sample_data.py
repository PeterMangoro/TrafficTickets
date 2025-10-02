#!/usr/bin/env python3
"""
Create sample data for demo/testing purposes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_data():
    """Create a small sample dataset for demo purposes"""
    
    # Sample violation types
    violations = [
        "Speeding - 1-10 mph over limit",
        "Speeding - 11-20 mph over limit", 
        "Red light violation",
        "Stop sign violation",
        "Unsafe lane change",
        "Following too closely",
        "Equipment violation",
        "Mobile phone violation",
        "Seat belt violation"
    ]
    
    # Weather conditions
    weather_conditions = [
        "Clear", "Partly Cloudy", "Cloudy", "Light Rain", 
        "Heavy Rain", "Snow", "Fog"
    ]
    
    # Days of week
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Generate dates for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Generate sample data (500 rows)
    sample_size = 500
    
    data = {
        'Date': random.choices(dates, k=sample_size),
        'Violation Year': np.random.choice([2023], sample_size),
        'Violation Month': [random.randint(1, 12) for _ in range(sample_size)],
        'Violation Day of Week': np.random.choice(range(1, 8), sample_size),  # 1=Monday, 7=Sunday
        'Age at Violation': np.random.choice(range(18, 66), sample_size),
        'Gender': np.random.choice(['Male', 'Female'], sample_size),
        'Violation Description': np.random.choice(violations, sample_size),
        'Violation Charged Code': ['TC' + str(random.randint(1000, 9999)) for _ in range(sample_size)]
    }
    
    df = pd.DataFrame(data)
    
    # Create sample weather data
    wx_data = []
    for date in random.sample(dates, 50):  # 50 different dates
        wx_data.append({
            'datetime': date.strftime('%Y-%m-%d %H:%M:%S'),
            'conditions': random.choice(weather_conditions)
        })
    
    wx_df = pd.DataFrame(wx_data)
    
    # Save sample data
    df.to_csv('sample_traffic_tickets_2023.csv', index=False)
    wx_df.to_csv('sample_weather_dataset.csv', index=False)
    
    print(f"Created sample data:")
    print(f"- Traffic tickets: {len(df)} rows")
    print(f"- Weather data: {len(wx_df)} rows")
    print("\nSample datasets saved:")
    print("- sample_traffic_tickets_2023.csv")
    print("- sample_weather_dataset.csv")

if __name__ == "__main__":
    create_sample_data()
