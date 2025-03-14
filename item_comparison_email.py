import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pyodbc
from fuzzywuzzy import fuzz
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# Load environment variables
load_dotenv()

def normalize_text(text):
    """Normalize text by removing special characters and converting to lowercase"""
    return ''.join(c.lower() for c in text if c.isalnum())

def get_db_connection():
    """Create database connection"""
    server = os.getenv('SERVER_')
    database = os.getenv('DATABASE_')
    username = os.getenv('USERNAME_')
    password = os.getenv('PASSWORD_')
    
    conn = pyodbc.connect(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};DATABASE={database};'
        f'UID={username};PWD={password}'
    )
    return conn

def get_db_items():
    """Fetch items from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT Description, LastUpdated FROM Item"
    cursor.execute(query)
    items = [(row.Description, row.LastUpdated) for row in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    return items
    
def send_email_report(email_address, email_password, recipient_email, report_file):
    """Send email with comparison report"""
    try:
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = recipient_email
        msg['Subject'] = f'Item Comparison Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        # Read the report to get a summary
        report_df = pd.read_csv(report_file)

        # Create email body
        body = f"Item Comparison Report Summary:\n\n"
        total_matches = len(report_df)
        
        if total_matches > 0:
            body += f"Total Matches Found: {total_matches}\n"
            body += f"Summary of Detected Items:\n"
            
            # Group items and their match count
            item_counts = report_df['Detected_Item'].value_counts()
            for item, count in item_counts.items():
                body += f"- {item}: {count} matches\n"

            body += f"\nFor detailed information, please find the attached report file."
        else:
            body += "No matches were detected within the specified time window."

        # Attach the CSV file
        with open(report_file, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(report_file))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(report_file)}"'
        
        msg.attach(MIMEText(body, 'plain'))
        msg.attach(part)

        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)

        print(f"Email report sent successfully at {datetime.now()}")

    except Exception as e:
        print(f"Error sending email: {e}")

def compare_detections():
    """Compare detected items with database entries"""
    try:
        # Read detections from CSV
        detections_df = pd.read_csv('detections.csv')
        
        # Convert timestamp strings to datetime objects
        detections_df['Timestamp'] = pd.to_datetime(detections_df['Timestamp'], format='%Y-%m-%d_%H-%M-%S', errors='coerce')

        # Get database items
        db_items = get_db_items()

        matches = []
        # Get current time for comparison
        current_time = datetime.now()

        # Process each detection
        for _, detection in detections_df.iterrows():
            detection_time = detection['Timestamp']
            
            # Skip if detection is older than 20 minutes
            if current_time - detection_time > timedelta(minutes=20):
                continue

            detected_object = detection['Object']
            normalized_detected = normalize_text(detected_object)

            # Convert confidence to percentage for reporting
            confidence_pct = detection['Confidence'] * 100

            best_match = None
            highest_ratio = 0

            # Compare with database items
            for db_desc, db_time in db_items:
                if db_desc:
                    normalized_db = normalize_text(db_desc)
                    ratio = fuzz.ratio(normalized_detected, normalized_db)

                    # Only consider matches where:
                    # 1. The fuzzy match ratio is > 80
                    # 2. The confidence is > 0.5 (50%)
                    if ratio > highest_ratio and ratio > 80 and detection['Confidence'] > 0.5:
                        highest_ratio = ratio
                        best_match = (db_desc, db_time)

            if best_match:
                detection_time = detection_time if pd.notna(detection_time) else best_match[1]
                matches.append({
                    'Detected_Item': detected_object,
                    'DB_Item': best_match[0],
                    'Detection_Time': detection_time,
                    'DB_LastUpdated': best_match[1],
                })

        # Create DataFrame and drop duplicates
        matches_df = pd.DataFrame(matches).drop_duplicates(subset=['Detected_Item'])
        matches_df['Detection_Time'].fillna(matches_df['DB_LastUpdated'], inplace=True)

        # Save matches to a report file
        if not matches_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'enhanced_comparison_report_{timestamp}.csv'
            matches_df.to_csv(report_filename, index=False)
            print(f"Enhanced comparison report generated: {report_filename}")
            print(f"Found {len(matches)} matches with confidence > 60%")

            # Send email with report
            email_address = os.getenv('EMAIL_ADDRESS')
            email_password = os.getenv('EMAIL_PASSWORD')
            recipient_email = os.getenv('RECIPIENT_EMAIL')

            if email_address and email_password and recipient_email:
                send_email_report(email_address, email_password, recipient_email, report_filename)
            else:
                print("Email credentials not fully configured. Skipping email.")
        else:
            print("No matches found in the current time window")

    except Exception as e:
        print(f"Error during comparison: {e}")

def run_comparison_scheduler():
    """Run the comparison at specified intervals"""
    interval_minutes = int(os.getenv('COMPARISON_INTERVAL_MINUTES', 5))
    
    print(f"Starting comparison scheduler (interval: {interval_minutes} minutes)")
    
    # Run initial comparison immediately
    compare_detections()
    
    schedule.every(interval_minutes).minutes.do(compare_detections)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run_comparison_scheduler()
