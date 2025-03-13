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
from datetime import datetime, timedelta
import pandas as pd


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

        # Read report to get summary
        report_df = pd.read_csv(report_file)
        
        # Create email body
        body = f"Item Comparison Report Summary:\n\n"
        body += f"Total Matches Found: {len(report_df)}\n"
        body += "Top 5 Matched Items:\n"
        top_items = report_df['Detected_Item'].value_counts().head(5)
        for item, count in top_items.items():
            body += f"- {item}: {count} matches\n"

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
    """Compare detected items with database entries and return matched results."""
    try:
        # Read detections from CSV
        detections_df = pd.read_csv('detections.csv')

        # Convert timestamp strings to datetime objects
        detections_df['Timestamp'] = pd.to_datetime(detections_df['Timestamp'], format='%Y-%m-%d_%H-%M-%S', errors='coerce')

        # Get database items
        db_items = get_db_items()

        # Define the reasonable time difference (e.g., 1 minute)
        time_threshold = timedelta(minutes=5)
        
        # Define the fuzzy matching threshold (e.g., 70%)
        fuzzy_threshold = 70

        matches = []

        # Process each detection
        for _, detection in detections_df.iterrows():
            detection_time = detection['Timestamp']
            detected_object = detection['Object']

            best_match = None
            best_score = 0

            # Compare with database items
            for db_desc, db_time in db_items:
                if db_desc:
                    # Calculate fuzzy similarity
                    similarity_score = fuzz.ratio(detected_object, db_desc)

                    # Check if the similarity score passes the threshold and time difference is reasonable
                    if pd.isna(detection_time) or pd.isna(db_time):
                        continue  # Skip comparison if either timestamp is NaT
                    time_difference = abs((detection_time - db_time))
                    if similarity_score >= fuzzy_threshold and time_difference <= time_threshold:
                        # Update best match if this score is higher
                        if similarity_score > best_score:
                            best_score = similarity_score
                            best_match = (db_desc, db_time)

            # If a match is found, add to the results
            if best_match:
                matches.append({
                    'Detected_Item': detected_object,
                    'Database_Item': best_match[0],
                    'Detection_Time': detection_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Database_Time': best_match[1].strftime('%Y-%m-%d %H:%M:%S'),
                    'Similarity_Score': best_score
                })

        # Save matches to a report file
        matches_df = pd.DataFrame(matches)
        if not matches_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'comparison_report_{timestamp}.csv'
            matches_df.to_csv(report_filename, index=False)
            print(f"Comparison report generated: {report_filename}")
            print(f"Found {len(matches)} matches with similarity >= {fuzzy_threshold}%")

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

        return matches

    except Exception as e:
        print(f"Error during comparison: {e}")
        return []

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
