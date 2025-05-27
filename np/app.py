from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from datetime import datetime, timedelta, date
import pytz

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Required for session and flash messages

# SQLite database setup
DATABASE = 'users.db'
SESSION_TIMEOUT_MINUTES = 30  # Session timeout duration
IST = pytz.timezone('Asia/Kolkata')

# Sample flight codes for autocomplete
FLIGHT_CODES = ['AI-101', '6E-233', 'SG-872', 'UK-901', 'VJ-456', 'AA-123', 'DL-789', 'BA-456', 'EK-567', 'QR-890']

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # Create users table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 email TEXT,
                 is_admin INTEGER DEFAULT 0
                 )''')
    # Create sessions table to track logged-in users
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                 user_id INTEGER PRIMARY KEY,
                 username TEXT NOT NULL,
                 login_time TEXT NOT NULL,
                 FOREIGN KEY (user_id) REFERENCES users(id)
                 )''')
    # Create session_history table to track login and logout times
    c.execute('''CREATE TABLE IF NOT EXISTS session_history (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 username TEXT NOT NULL,
                 login_time TEXT NOT NULL,
                 logout_time TEXT,
                 FOREIGN KEY (user_id) REFERENCES users(id)
                 )''')
    # Create predictions table to store prediction results
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 flight_name TEXT NOT NULL,
                 prediction_time TEXT NOT NULL,
                 is_delayed INTEGER NOT NULL,
                 carrier_ct REAL NOT NULL,
                 nas_ct REAL NOT NULL,
                 security_ct REAL NOT NULL,
                 FOREIGN KEY (user_id) REFERENCES users(id)
                 )''')
    # Check if email column exists and add it if missing
    c.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in c.fetchall()]
    if 'email' not in columns:
        print("Adding 'email' column to users table...")
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")
        print("Email column added successfully.")
    # Check if is_admin column exists and add it if missing
    if 'is_admin' not in columns:
        print("Adding 'is_admin' column to users table...")
        c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        print("is_admin column added successfully.")
    # Ensure at least one admin user exists (username: admin, password: admin123)
    c.execute("SELECT id FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                 ('admin', generate_password_hash('admin123'), 1))
    conn.commit()
    conn.close()

# Initialize or update database schema
try:
    init_db()
    # Verify the schema
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in c.fetchall()]
    required_columns = {'email', 'is_admin'}
    if not all(col in columns for col in required_columns):
        # If adding columns failed, create a new table and migrate data
        print("Failed to add required columns. Migrating to new table structure...")
        c.execute('''CREATE TABLE users_new (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE NOT NULL,
                     email TEXT,
                     password TEXT NOT NULL,
                     is_admin INTEGER DEFAULT 0
                     )''')
        # Migrate existing data
        c.execute("INSERT INTO users_new (id, username, password, email) SELECT id, username, password, email FROM users")
        c.execute("DROP TABLE users")
        c.execute("ALTER TABLE users_new RENAME TO users")
        # Recreate admin user
        c.execute("SELECT id FROM users WHERE username = 'admin'")
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', generate_password_hash('admin123'), 1))
        conn.commit()
        print("Migration to new table structure completed.")
    print("Database schema verified successfully:", columns)
    conn.close()

except sqlite3.Error as e:
    print(f"Database initialization failed: {str(e)}")
    print("Please ensure no other process is using 'users.db' and restart the app.")
    print("If the issue persists, manually delete 'users.db' and restart the app.")
    raise SystemExit("Terminating due to database initialization failure.")

# Email configuration
EMAIL_ADDRESS = 'Sinannp07@gmail.com'
EMAIL_PASSWORD = 'ytfemikymzbfhwxy'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

def send_otp_email(to_email, otp):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = 'Flight Delay Prediction - OTP Verification'
    body = f'Your OTP for email verification is: {otp}\nThis OTP is valid for 10 minutes.'
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        return False

# Load or train the model
MODEL_PATH = "flight_delay_model.pkl"
EXPECTED_FEATURES = ['carrier_ct', 'nas_ct', 'security_ct']

def train_model():
    # Train a dummy model with the expected features
    df = pd.DataFrame({
        'carrier_ct': [0.1, 0.2, 0.15, 0.3],
        'nas_ct': [0.05, 0.1, 0.07, 0.12],
        'security_ct': [0.01, 0.02, 0.015, 0.03],
        'delay_status': [0, 1, 0, 1]
    })
    X = df[EXPECTED_FEATURES]
    y = df['delay_status']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    return model

# Load the model, retrain if necessary
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        # Check if the model expects the correct features
        try:
            # Create a dummy input to test the model's feature expectations
            dummy_features = pd.DataFrame([[0.1, 0.1, 0.1]], columns=EXPECTED_FEATURES)
            model.predict(dummy_features)
        except ValueError as e:
            print(f"Model feature mismatch: {str(e)}. Retraining model...")
            model = train_model()
    else:
        print("Model file not found. Training new model...")
        model = train_model()
except Exception as e:
    print(f"Failed to load model: {str(e)}. Training new model...")
    model = train_model()

# Clean up expired sessions
def cleanup_expired_sessions(conn, c):
    try:
        # Fetch all sessions and filter in Python
        c.execute("SELECT user_id, login_time FROM sessions")
        sessions = c.fetchall()
        current_time = datetime.utcnow()
        expired_ids = []
        for user_id, login_time in sessions:
            try:
                login_dt = datetime.fromisoformat(login_time)
                if (current_time - login_dt) > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    expired_ids.append(user_id)
                    # Update session_history with logout time for expired sessions
                    logout_time = current_time.isoformat()
                    c.execute("UPDATE session_history SET logout_time = ? WHERE user_id = ? AND logout_time IS NULL",
                             (logout_time, user_id))
            except ValueError as e:
                print(f"Invalid login_time format for user_id {user_id}: {login_time}. Error: {str(e)}")
                expired_ids.append(user_id)  # Remove invalid entries
        if expired_ids:
            c.executemany("DELETE FROM sessions WHERE user_id = ?", [(id,) for id in expired_ids])
            conn.commit()
            print(f"Cleaned up {len(expired_ids)} expired sessions: {expired_ids}")
    except sqlite3.Error as e:
        print(f"Failed to clean up expired sessions: {str(e)}")

# Login required decorator
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        # Check if session is still valid
        try:
            with sqlite3.connect(DATABASE) as conn:
                c = conn.cursor()
                c.execute("SELECT login_time FROM sessions WHERE user_id = ?", (session['user_id'],))
                session_record = c.fetchone()
                if not session_record:
                    print(f"No session found for user_id={session['user_id']}")
                    session.pop('user_id', None)
                    session.pop('is_admin', None)
                    flash('Session expired. Please log in again.', 'error')
                    return redirect(url_for('login'))
                login_time = session_record[0]
                try:
                    login_dt = datetime.fromisoformat(login_time)
                except ValueError as e:
                    print(f"Invalid login_time format for user_id={session['user_id']}: {login_time}. Error: {str(e)}")
                    c.execute("DELETE FROM sessions WHERE user_id = ?", (session['user_id'],))
                    conn.commit()
                    session.pop('user_id', None)
                    session.pop('is_admin', None)
                    flash('Session data corrupted. Please log in again.', 'error')
                    return redirect(url_for('login'))
                current_time = datetime.utcnow()
                time_diff = current_time - login_dt
                print(f"Session check for user_id={session['user_id']}: login_time={login_time}, current_time={current_time.isoformat()}, time_diff={time_diff}")
                if time_diff > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    print(f"Session expired for user_id={session['user_id']}. Time difference: {time_diff}")
                    c.execute("UPDATE session_history SET logout_time = ? WHERE user_id = ? AND logout_time IS NULL",
                             (current_time.isoformat(), session['user_id']))
                    c.execute("DELETE FROM sessions WHERE user_id = ?", (session['user_id'],))
                    conn.commit()
                    session.pop('user_id', None)
                    session.pop('is_admin', None)
                    flash('Session expired. Please log in again.', 'error')
                    return redirect(url_for('login'))
        except sqlite3.Error as e:
            print(f"Database error in login_required: {str(e)}")
            flash('Database error occurred. Please try again.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# Admin required decorator
def admin_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        print(f"Admin check - Session: user_id={session.get('user_id')}, is_admin={session.get('is_admin')}")
        if not session.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validate inputs
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))

        # Validate email format
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_regex, email):
            flash('Invalid email format.', 'error')
            return redirect(url_for('register'))

        try:
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            # Check for existing username or email
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            if c.fetchone():
                conn.close()
                flash('Username already exists.', 'error')
                return redirect(url_for('register'))

            c.execute("SELECT id FROM users WHERE email = ? AND email IS NOT NULL", (email,))
            if c.fetchone():
                conn.close()
                flash('Email already exists.', 'error')
                return redirect(url_for('register'))

            # Generate 6-digit OTP
            otp = ''.join(random.choices(string.digits, k=6))
            session['pending_user'] = {
                'username': username,
                'email': email,
                'password': generate_password_hash(password),
                'otp': otp
            }

            # Send OTP email
            if send_otp_email(email, otp):
                conn.close()
                flash('An OTP has been sent to your email.', 'success')
                return redirect(url_for('verify_otp'))
            else:
                conn.close()
                flash('Failed to send OTP. Please try again.', 'error')
                return redirect(url_for('register'))

        except sqlite3.Error as e:
            conn.close()
            flash(f'Registration failed: {str(e)}', 'error')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'pending_user' not in session:
        flash('No pending registration found.', 'error')
        return redirect(url_for('register'))

    if request.method == 'POST':
        user_otp = request.form['otp'].strip()

        pending_user = session['pending_user']
        if user_otp == pending_user['otp']:
            try:
                conn = sqlite3.connect(DATABASE)
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                         (pending_user['username'], pending_user['email'], pending_user['password']))
                conn.commit()
                conn.close()
                
                session.pop('pending_user', None)
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.Error as e:
                conn.close()
                flash(f'Registration failed: {str(e)}', 'error')
                return redirect(url_for('verify_otp'))
        else:
            flash('Invalid OTP. Please try again.', 'error')
            return redirect(url_for('verify_otp'))
    
    return render_template('verify.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        try:
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute("SELECT id, password, is_admin FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            
            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['is_admin'] = bool(user[2])
                # Add user to sessions table
                login_time = datetime.utcnow().isoformat()
                c.execute("INSERT OR REPLACE INTO sessions (user_id, username, login_time) VALUES (?, ?, ?)",
                         (user[0], username, login_time))
                # Add to session_history
                c.execute("INSERT INTO session_history (user_id, username, login_time) VALUES (?, ?, ?)",
                         (user[0], username, login_time))
                conn.commit()
                print(f"Session created for user_id={user[0]}, username={username}, login_time={login_time}")
                conn.close()
                flash('Login successful!', 'success')
                # Redirect admin to admin page, others to index
                if session['is_admin']:
                    return redirect(url_for('admin'))
                return redirect(url_for('index'))
            else:
                conn.close()
                flash('Invalid username or password.', 'error')
                return redirect(url_for('login'))
        except sqlite3.Error as e:
            conn.close()
            flash(f'Login failed: {str(e)}', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    if user_id:
        try:
            with sqlite3.connect(DATABASE) as conn:
                c = conn.cursor()
                # Update session_history with logout time
                logout_time = datetime.utcnow().isoformat()
                c.execute("UPDATE session_history SET logout_time = ? WHERE user_id = ? AND logout_time IS NULL",
                         (logout_time, user_id))
                c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                conn.commit()
                print(f"Logged out user_id={user_id}, logout_time={logout_time}")
        except sqlite3.Error as e:
            print(f"Error during logout: {str(e)}")
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    stats = {'total_predictions': 0, 'delay_percentage': 0.0}
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM predictions")
            stats['total_predictions'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM predictions WHERE is_delayed = 1")
            delayed_count = c.fetchone()[0]
            if stats['total_predictions'] > 0:
                stats['delay_percentage'] = (delayed_count / stats['total_predictions']) * 100
    except sqlite3.Error as e:
        print(f"Error fetching prediction stats: {str(e)}")
        flash('Unable to load prediction statistics.', 'error')
    
    return render_template('index.html', stats=stats)

@app.route('/admin')
@admin_required
def admin():
    print(f"Accessing admin dashboard for user_id={session.get('user_id')}")
    logged_in_users = []
    logged_out_users = []
    current_time = datetime.utcnow()
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            # Verify schema
            c.execute("PRAGMA table_info(users)")
            user_columns = {info[1] for info in c.fetchall()}
            c.execute("PRAGMA table_info(sessions)")
            session_columns = {info[1] for info in c.fetchall()}
            c.execute("PRAGMA table_info(session_history)")
            history_columns = {info[1] for info in c.fetchall()}
            required_user_columns = {'id', 'username', 'password', 'email', 'is_admin'}
            required_session_columns = {'user_id', 'username', 'login_time'}
            required_history_columns = {'id', 'user_id', 'username', 'login_time', 'logout_time'}
            if not required_user_columns.issubset(user_columns):
                raise sqlite3.Error(f"Users table missing required columns. Expected: {required_user_columns}, Found: {user_columns}")
            if not required_session_columns.issubset(session_columns):
                raise sqlite3.Error(f"Sessions table missing required columns. Expected: {required_session_columns}, Found: {session_columns}")
            if not required_history_columns.issubset(history_columns):
                raise sqlite3.Error(f"Session_history table missing required columns. Expected: {required_history_columns}, Found: {history_columns}")

            # Clean up expired sessions and fetch active users in one transaction
            cleanup_expired_sessions(conn, c)

            # Fetch currently logged-in non-admin users
            c.execute("SELECT sessions.username, sessions.login_time, users.is_admin FROM sessions JOIN users ON sessions.user_id = users.id")
            sessions = c.fetchall()
            print(f"Raw session data: {sessions}")
            for username, login_time, is_admin in sessions:
                try:
                    login_dt = datetime.fromisoformat(login_time)
                    time_diff = current_time - login_dt
                    print(f"User: {username}, is_admin: {is_admin}, login_time: {login_time}, time_diff: {time_diff}, within_timeout: {time_diff <= timedelta(minutes=SESSION_TIMEOUT_MINUTES)}")
                    if time_diff <= timedelta(minutes=SESSION_TIMEOUT_MINUTES) and is_admin == 0:
                        # Convert login_time to IST
                        login_dt = login_dt.replace(tzinfo=pytz.UTC)
                        login_ist = login_dt.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S')
                        logged_in_users.append({'username': username, 'login_time': login_ist})
                except ValueError as e:
                    print(f"Invalid login_time format for username {username}: {login_time}. Error: {str(e)}")

            # Fetch logged-out non-admin users from session_history
            c.execute("SELECT session_history.username, session_history.login_time, session_history.logout_time, users.is_admin FROM session_history JOIN users ON session_history.user_id = users.id WHERE session_history.logout_time IS NOT NULL")
            past_sessions = c.fetchall()
            print(f"Raw session_history data: {past_sessions}")
            for username, login_time, logout_time, is_admin in past_sessions:
                if is_admin == 0:
                    try:
                        login_dt = datetime.fromisoformat(login_time)
                        logout_dt = datetime.fromisoformat(logout_time)
                        # Convert to IST
                        login_dt = login_dt.replace(tzinfo=pytz.UTC)
                        logout_dt = logout_dt.replace(tzinfo=pytz.UTC)
                        login_ist = login_dt.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S')
                        logout_ist = logout_dt.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S')
                        logged_out_users.append({'username': username, 'login_time': login_ist, 'logout_time': logout_ist})
                    except ValueError as e:
                        print(f"Invalid time format for username {username}: login_time={login_time}, logout_time={logout_time}. Error: {str(e)}")

            print(f"Logged-in users: {logged_in_users}")
            print(f"Logged-out users: {logged_out_users}")

    except sqlite3.Error as e:
        print(f"Database error in admin route: {str(e)}")
        flash(f'Failed to fetch user data: {str(e)}. Please try again or contact support.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Unexpected error in admin route: {str(e)}")
        flash(f'An unexpected error occurred: {str(e)}. Please try again or contact support.', 'error')
        return redirect(url_for('index'))

    current_time_ist = current_time.replace(tzinfo=pytz.UTC).astimezone(IST).strftime('%Y-%m-%d %H:%M:%S')
    return render_template('admin.html', logged_in_users=logged_in_users, logged_out_users=logged_out_users, current_time=current_time_ist, session_timeout_minutes=SESSION_TIMEOUT_MINUTES)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get form data
        flight_name = request.form['flight_name'].strip().upper()
        departure_datetime = request.form['departure_datetime']
        carrier_ct = float(request.form['carrier_ct'])
        nas_ct = float(request.form['nas_ct'])
        security_ct = float(request.form['security_ct'])

        # Parse departure_datetime (format: YYYY-MM-DD HH:mm)
        try:
            dep_dt = datetime.strptime(departure_datetime, '%Y-%m-%d %H:%M')
            year = dep_dt.year
            month = dep_dt.month
            day = dep_dt.day
            dep_hour = dep_dt.hour
        except ValueError:
            flash('Invalid date format. Please select a valid date and time.', 'error')
            return redirect(url_for('index'))

        # Validate inputs
        if not (2000 <= year <= 2025):
            flash('Year must be between 2000 and 2025.', 'error')
            return redirect(url_for('index'))
        if not (1 <= month <= 12):
            flash('Month must be between 1 and 12.', 'error')
            return redirect(url_for('index'))
        if not (1 <= day <= 31):
            flash('Day must be between 1 and 31.', 'error')
            return redirect(url_for('index'))
        # Validate the date
        try:
            date(year, month, day)
        except ValueError:
            flash('Invalid date. Please check the day of the month for the given year and month (e.g., February has 28 or 29 days).', 'error')
            return redirect(url_for('index'))
        if not (0 <= dep_hour <= 23):
            flash('Departure hour must be between 0 and 23.', 'error')
            return redirect(url_for('index'))
        if any(x < 0 for x in [carrier_ct, nas_ct, security_ct]):
            flash('Delay counts cannot be negative.', 'error')
            return redirect(url_for('index'))

        # Prepare features for prediction
        features = pd.DataFrame({
            'carrier_ct': [carrier_ct],
            'nas_ct': [nas_ct],
            'security_ct': [security_ct]
        }, columns=EXPECTED_FEATURES)

        # Make prediction
        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
        except Exception as e:
            flash(f'Prediction failed: {str(e)}. The model may be incompatible. Please contact the administrator.', 'error')
            return redirect(url_for('index'))

        # Store prediction in database
        try:
            with sqlite3.connect(DATABASE) as conn:
                c = conn.cursor()
                prediction_time = datetime.utcnow().isoformat()
                c.execute("INSERT INTO predictions (user_id, flight_name, prediction_time, is_delayed, carrier_ct, nas_ct, security_ct) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (session['user_id'], flight_name, prediction_time, prediction, carrier_ct, nas_ct, security_ct))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error storing prediction: {str(e)}")
            flash('Prediction was successful, but failed to save the result.', 'error')

        # Interpret result
        result = "Delayed (more than 10 minutes)" if prediction == 1 else "Not Delayed"
        confidence = f"{probability:.2%}" if prediction == 1 else f"{1 - probability:.2%}"

        return render_template('result.html',
                              flight_name=flight_name,
                              year=year,
                              month=month,
                              day=day,
                              dep_hour=dep_hour,
                              result=result,
                              confidence=confidence,
                              carrier_ct=carrier_ct,
                              nas_ct=nas_ct,
                              security_ct=security_ct)

    except ValueError:
        flash('Please enter valid numerical values for all fields (e.g., numbers for delay counts).', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}. Please try again or contact the administrator.', 'error')
        return redirect(url_for('index'))

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').upper()
    suggestions = [code for code in FLIGHT_CODES if query in code]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)