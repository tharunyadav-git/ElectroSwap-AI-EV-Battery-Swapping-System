import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'battery-swap-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- EXTENSIONS ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- CONSTANTS & PATHS ---
DATASET_PATH = 'Battery_RUL.csv'
MODEL_PATH = 'xgboost_rul_model.pkl'

# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

# --- GLOBAL DATA & FEATURES ---
FEATURES = [
    'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 
    'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 
    'Time at 4.15V (s)', 'Time constant current (s)', 
    'Charging time (s)'
]

try:
    global_df = pd.read_csv(DATASET_PATH)
except:
    global_df = pd.DataFrame()

# --- MODEL MANAGEMENT ---
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists(DATASET_PATH):
        print("Model not found. Training new model...")
        df = pd.read_csv(DATASET_PATH)
        X = df[FEATURES]
        y = df['RUL']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(
            learning_rate=0.30, max_depth=10, n_estimators=500,
            objective='reg:squarederror', n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        return model
    else:
        return None

model = get_model()

# --- PRICING LOGIC ---
def calculate_pricing(veh_data, stat_data):
    if model is None: return 0, 0, 0, 0, 0
    
    # Predict
    veh_df = pd.DataFrame([veh_data], columns=FEATURES)
    stat_df = pd.DataFrame([stat_data], columns=FEATURES)
    
    rul_veh = model.predict(veh_df)[0]
    rul_stat = model.predict(stat_df)[0]
    
    # Constants
    BASE_COST = 20.0
    RUL_UNIT_PRICE = 0.1
    
    # RUL Difference Cost
    rul_diff = rul_stat - rul_veh
    rul_cost = rul_diff * RUL_UNIT_PRICE
    
    # Performance Weights (from Paper Fig 3)
    weights = {
        'Discharge Time (s)': 0.01,
        'Time at 4.15V (s)': 0.18,
        'Time constant current (s)': 0.04,
        'Decrement 3.6-3.4V (s)': 0.01,
        'Max. Voltage Dischar. (V)': 0.78,
        'Min. Voltage Charg. (V)': -0.76,
        'Charging time (s)': 0.02
    }
    
    perf_cost = 0
    
    for feat in FEATURES:
        diff = stat_data[feat] - veh_data[feat]
        cost = diff * weights.get(feat, 0)
        perf_cost += cost
            
    total = BASE_COST + rul_cost + perf_cost
    return round(rul_veh, 2), round(rul_stat, 2), round(rul_cost, 2), round(perf_cost, 2), round(total, 2)

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.is_admin:
                return redirect(url_for('admin_panel'))
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
        else:
            new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    result = None
    defaults = {
        'discharge_time': 2500, 'dec_36_34': 800, 
        'max_v': 3.9, 'min_v': 3.4, 'time_415': 5000, 
        'const_curr': 6000, 'charge_time': 10000
    }

    if request.method == 'POST':
        veh_data = {
            'Discharge Time (s)': float(request.form['discharge_time']),
            'Decrement 3.6-3.4V (s)': float(request.form['dec_36_34']),
            'Max. Voltage Dischar. (V)': float(request.form['max_v']),
            'Min. Voltage Charg. (V)': float(request.form['min_v']),
            'Time at 4.15V (s)': float(request.form['time_415']),
            'Time constant current (s)': float(request.form['const_curr']),
            'Charging time (s)': float(request.form['charge_time'])
        }
        
        stat_data = {
            'Discharge Time (s)': 2600,
            'Decrement 3.6-3.4V (s)': 1200,
            'Max. Voltage Dischar. (V)': 4.2,
            'Min. Voltage Charg. (V)': 3.2,
            'Time at 4.15V (s)': 5500,
            'Time constant current (s)': 6800,
            'Charging time (s)': 10500
        }
        
        rul_veh, rul_stat, rul_cost, perf_cost, total_price = calculate_pricing(veh_data, stat_data)
        
        result = {
            'rul_veh': rul_veh, 'rul_stat': rul_stat,
            'rul_cost': rul_cost, 'perf_cost': perf_cost,
            'total': total_price
        }

    return render_template('dashboard.html', defaults=defaults, result=result)

@app.route('/graphs')
@login_required
def graphs():
    if global_df.empty:
        flash("Dataset not loaded. Graphs unavailable.", "error")
        return redirect(url_for('dashboard'))

    # Subsample for performance
    sample_df = global_df.sample(min(2000, len(global_df)))

    # --- CHART 1: RUL Distribution (Histogram) ---
    fig1 = px.histogram(global_df, x='RUL', nbins=50, title='RUL Distribution',
                        color_discrete_sequence=['#06b6d4'])
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # --- CHART 2: Correlation Heatmap (Figure 3 in Paper) ---
    # Select numeric columns relevant to the paper
    corr_cols = FEATURES + ['RUL']
    corr_matrix = global_df[corr_cols].corr()
    fig2 = px.imshow(corr_matrix, 
                     text_auto=True, 
                     aspect="auto",
                     color_continuous_scale='RdBu_r',
                     title='Feature Correlation Matrix (Paper Fig. 3)')
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # --- CHART 3: Actual vs Predicted RUL (Model Performance) ---
    # Generate predictions for the sample
    if model:
        X_sample = sample_df[FEATURES]
        y_true = sample_df['RUL']
        y_pred = model.predict(X_sample)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', 
                                  marker=dict(color='#8b5cf6', opacity=0.6),
                                  name='Predictions'))
        # Add a perfect prediction line
        fig3.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], 
                                  y=[y_true.min(), y_true.max()],
                                  mode='lines', line=dict(color='white', dash='dash'),
                                  name='Perfect Fit'))
        fig3.update_layout(title='Actual vs Predicted RUL', xaxis_title='Actual RUL', yaxis_title='Predicted RUL')
        graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph3JSON = None

    # --- CHART 4: Degradation Curve (RUL vs Cycle Index) ---
    if 'Cycle_Index' in global_df.columns:
        fig4 = px.scatter(sample_df, x='Cycle_Index', y='RUL', 
                          title='Battery Degradation Curve',
                          color='Discharge Time (s)',
                          color_continuous_scale='Viridis')
        graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph4JSON = None

    # --- CHART 5: Feature Analysis (Discharge Time vs RUL) ---
    fig5 = px.scatter(sample_df, x='Discharge Time (s)', y='RUL', 
                      title='Impact of Discharge Time on RUL',
                      color='Max. Voltage Dischar. (V)',
                      color_continuous_scale='Plasma')
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('graphs.html', 
                           graph1JSON=graph1JSON, 
                           graph2JSON=graph2JSON, 
                           graph3JSON=graph3JSON,
                           graph4JSON=graph4JSON,
                           graph5JSON=graph5JSON)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash("Access Denied. Admins only.", "error")
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'delete':
            user_id = request.form.get('user_id')
            User.query.filter_by(id=user_id).delete()
            db.session.commit()
            flash('User deleted.', 'success')
        elif action == 'create':
            uname = request.form.get('username')
            pwd = request.form.get('password')
            is_admin = request.form.get('is_admin') == 'on'
            if User.query.filter_by(username=uname).first():
                flash('User already exists', 'error')
            else:
                new_u = User(username=uname, password=generate_password_hash(pwd, method='pbkdf2:sha256'), is_admin=is_admin)
                db.session.add(new_u)
                db.session.commit()
                flash('User created successfully', 'success')

    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# --- DB INIT ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', password=generate_password_hash('admin', method='pbkdf2:sha256'), is_admin=True)
            db.session.add(admin)
        if not User.query.filter_by(username='user').first():
            user = User(username='user', password=generate_password_hash('user', method='pbkdf2:sha256'), is_admin=False)
            db.session.add(user)
        db.session.commit()
        print("Database initialized.")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) and not os.path.exists(DATASET_PATH):
        print("WARNING: Neither model file nor dataset csv found.")
    init_db()
    app.run(debug=True, port=5000)
