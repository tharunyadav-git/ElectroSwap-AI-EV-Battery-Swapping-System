import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
DATASET_PATH = 'Battery_RUL.csv'
MODEL_PATH = 'xgboost_rul_model.pkl'
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

# Set Dark Theme for Plots (Cyberpunk Style)
plt.style.use('dark_background')
sns.set_context("talk")
colors = ["#06b6d4", "#8b5cf6", "#ec4899", "#10b981", "#3b82f6"] # Cyan, Purple, Pink, Emerald, Blue
sns.set_palette(sns.color_palette(colors))

FEATURES = [
    'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 
    'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 
    'Time at 4.15V (s)', 'Time constant current (s)', 
    'Charging time (s)'
]

# Weights from Paper (Fig 3)
WEIGHTS = {
    'Discharge Time (s)': 0.01,
    'Time at 4.15V (s)': 0.18,
    'Time constant current (s)': 0.04,
    'Decrement 3.6-3.4V (s)': 0.01,
    'Max. Voltage Dischar. (V)': 0.78,
    'Min. Voltage Charg. (V)': -0.76,
    'Charging time (s)': 0.02
}

def train_and_visualize():
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset not found!")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # --- 1. TRAIN MODEL ---
    print("Training XGBoost Model...")
    X = df[FEATURES]
    y = df['RUL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        learning_rate=0.30, max_depth=10, n_estimators=500,
        objective='reg:squarederror', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    
    y_pred = model.predict(X_test)
    print("Model trained and saved.")

    # --- 2. GENERATE PLOTS ---
    print("Generating Static Graphs...")
    
    def save_plot(filename):
        path = os.path.join(STATIC_DIR, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved {filename}")

    # CHART 1: RUL Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['RUL'], bins=50, kde=True, color=colors[0], edgecolor='black')
    plt.title('Distribution of Remaining Useful Life (RUL)', color='white', pad=20)
    plt.xlabel('RUL (Cycles)', color='gray')
    save_plot('chart1_dist.png')

    # CHART 2: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df[FEATURES + ['RUL']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='icefire', square=True)
    plt.title('Feature Correlation Heatmap', color='white', pad=20)
    save_plot('chart2_heatmap.png')

    # CHART 3: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color=colors[1], edgecolor='none', s=50)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'w--', lw=2)
    plt.title('Model Performance: Actual vs Predicted', color='white', pad=20)
    plt.xlabel('Actual RUL', color='gray')
    plt.ylabel('Predicted RUL', color='gray')
    save_plot('chart3_pred_vs_actual.png')

    # CHART 4: Degradation Curve
    if 'Cycle_Index' in df.columns:
        plt.figure(figsize=(10, 6))
        sample = df.sample(min(2000, len(df)))
        sc = plt.scatter(sample['Cycle_Index'], sample['RUL'], c=sample['Discharge Time (s)'], 
                    cmap='viridis', alpha=0.8, s=30)
        plt.colorbar(sc, label='Discharge Time (s)')
        plt.title('Battery Degradation: Cycle vs RUL', color='white', pad=20)
        plt.xlabel('Cycle Index', color='gray')
        plt.ylabel('RUL', color='gray')
        save_plot('chart4_degradation.png')

    # CHART 5: Feature Impact
    plt.figure(figsize=(10, 6))
    sample = df.sample(min(2000, len(df)))
    sc = plt.scatter(sample['Discharge Time (s)'], sample['RUL'], c=sample['Max. Voltage Dischar. (V)'], 
                cmap='plasma', alpha=0.8, s=30)
    plt.colorbar(sc, label='Max Voltage (V)')
    plt.title('Discharge Time vs RUL', color='white', pad=20)
    plt.xlabel('Discharge Time (s)', color='gray')
    plt.ylabel('RUL', color='gray')
    save_plot('chart5_feature_impact.png')

    # --- NEW CHART 6: PRICING STRATEGY ANALYSIS ---
    # Simulate a user bringing in batteries at different health levels
    # compared to a fixed "Healthy Station Battery"
    print("Generating Pricing Analysis...")
    
    # 1. Define "Station Battery" (Healthy, Low Cycle Count)
    station_batt = df.sort_values('RUL', ascending=False).iloc[0]
    station_rul = station_batt['RUL']
    
    # 2. Select a range of "Vehicle Batteries" (High to Low RUL)
    # We sample 100 points across the degradation curve
    vehicle_batts = df.sort_values('RUL', ascending=False).iloc[::len(df)//100]
    
    prices = []
    ruls = []
    
    BASE_COST = 20.0
    RUL_PRICE = 0.1
    
    for _, veh in vehicle_batts.iterrows():
        # RUL Cost
        veh_rul = veh['RUL']
        rul_diff = station_rul - veh_rul
        rul_cost = rul_diff * RUL_PRICE
        
        # Perf Cost
        perf_cost = 0
        for feat, w in WEIGHTS.items():
            if feat in veh:
                diff = station_batt[feat] - veh[feat]
                perf_cost += diff * w
        
        total = BASE_COST + rul_cost + perf_cost
        prices.append(total)
        ruls.append(veh_rul)

    plt.figure(figsize=(10, 6))
    
    # Plot Fair Price Curve
    plt.plot(ruls, prices, color='#10b981', linewidth=3, label='Proposed Fair Price')
    
    # Plot Fixed Price Baseline (e.g. Average of Fair Prices or Arbitrary $50)
    avg_price = np.mean(prices)
    plt.axhline(y=avg_price, color='#ef4444', linestyle='--', linewidth=2, label=f'Fixed Market Price (${avg_price:.0f})')
    
    plt.gca().invert_xaxis() # High RUL (New) -> Low RUL (Old) on X-axis
    plt.title('Fair Pricing Strategy vs Fixed Pricing', color='white', pad=20)
    plt.xlabel('Vehicle Battery RUL (Health)', color='gray')
    plt.ylabel('Swap Cost ($)', color='gray')
    plt.legend()
    plt.grid(color='#333', linestyle='--', linewidth=0.5)
    
    # Annotations
    plt.text(ruls[0], prices[0], ' New Battery\n(Low Cost)', color='white', ha='right')
    plt.text(ruls[-1], prices[-1], ' Dead Battery\n(High Cost)', color='white', ha='left')

    save_plot('chart6_pricing_strategy.png')

    print("All tasks complete. Run 'python app.py' now.")

if __name__ == "__main__":
    train_and_visualize()
