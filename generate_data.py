import pandas as pd
import numpy as np

np.random.seed(42)

# -----------------------------------
# PARAMETERS
# -----------------------------------

num_atms = 60
num_days = 180
start_date = pd.to_datetime("2025-01-01")

dates = pd.date_range(start_date, periods=num_days)

data = []

for atm_id in range(1, num_atms + 1):
    
    # Assign location type
    location_type = np.random.choice([1, 2, 3])  # 1=Urban, 2=Semi-Urban, 3=Rural
    
    # Base demand by location
    if location_type == 1:
        base_demand = np.random.randint(18000, 25000)
    elif location_type == 2:
        base_demand = np.random.randint(12000, 18000)
    else:
        base_demand = np.random.randint(6000, 12000)
    
    competitor = np.random.choice([0, 1])
    previous_cash = base_demand * 1.5

    for date in dates:
        
        day_of_week = date.dayofweek
        time_of_day = np.random.choice([1, 2, 3, 4])
        
        holiday_flag = 1 if np.random.rand() < 0.08 else 0
        special_event_flag = 1 if np.random.rand() < 0.05 else 0
        
        weather = np.random.choice([1, 2, 3])  # 1=Normal, 2=Rain, 3=Storm
        
        # Weekly boost (Fri/Sat higher)
        weekly_factor = 1.2 if day_of_week in [4, 5] else 1.0
        
        holiday_factor = 1.4 if holiday_flag == 1 else 1.0
        event_factor = 1.5 if special_event_flag == 1 else 1.0
        
        withdrawals = base_demand * weekly_factor * holiday_factor * event_factor
        withdrawals += np.random.normal(0, 1500)
        
        # Inject rare anomaly spikes
        if np.random.rand() < 0.02:
            withdrawals *= 2.5
        
        withdrawals = max(0, withdrawals)
        
        deposits = withdrawals * np.random.uniform(0.6, 0.9)
        next_day_demand = withdrawals * np.random.uniform(0.9, 1.1)
        
        data.append([
            atm_id,
            date,
            day_of_week,
            time_of_day,
            round(withdrawals, 2),
            round(deposits, 2),
            round(previous_cash, 2),
            location_type,
            holiday_flag,
            special_event_flag,
            weather,
            competitor,
            round(next_day_demand, 2)
        ])
        
        previous_cash = withdrawals * 1.3

# -----------------------------------
# CREATE DATAFRAME
# -----------------------------------

columns = [
    "ATM_ID",
    "Date",
    "Day_of_Week",
    "Time_of_Day",
    "Total_Withdrawals",
    "Total_Deposits",
    "Previous_Day_Cash_Level",
    "Location_Type",
    "Holiday_Flag",
    "Special_Event_Flag",
    "Weather_Condition",
    "Nearby_Competitor_ATMs",
    "Cash_Demand_Next_Day"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("cleaned_atm_data.csv", index=False)

print("✅ Dataset generated successfully!")
print("Shape:", df.shape)