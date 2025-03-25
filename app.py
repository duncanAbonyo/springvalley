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
        msg['Subject'] = f'Product Detection Report - {datetime.now().strftime("%Y-%m-%d")}'

        # Read the report to get a summary
        report_df = pd.read_csv(report_file)
        
        # Create HTML email body for better formatting
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h2>Product Detection Report</h2>
            <p>Detection period: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h3>Summary</h3>
                <p>Total Products Detected: {len(report_df)}</p>
            </div>
        """
        
        if not report_df.empty:
            # Add product detection table
            html_body += """
            <h3>Detected Products</h3>
            <table>
                <tr>
                    <th>Product</th>
                    <th>Detection Time</th>
                </tr>
            """
            
            for _, row in report_df.iterrows():
                detection_time = pd.to_datetime(row['Detection_Time']).strftime("%Y-%m-%d %H:%M:%S")
                html_body += f"""
                <tr>
                    <td>{row['Detected_Item']}</td>
                    <td>{detection_time}</td>
                </tr>
                """
            
            html_body += "</table>"
            html_body += "<p>Please find the detailed report attached.</p>"
        else:
            html_body += "<p>No products were detected during this period.</p>"
        
        html_body += """
            </body>
            </html>
        """
        
        # Create plain text version for email clients that don't support HTML
        plain_text = f"""
        Product Detection Report
        Detection period: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        Summary:
        Total Products Detected: {len(report_df)}
        
        Please see the attached CSV file for detailed information.
        """
        
        # Attach the HTML and plain text versions
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach the CSV file
        with open(report_file, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(report_file))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(report_file)}"'
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

            # Compare with database items - ONLY MATCH WITHIN THE SAME DAY
            for db_desc, db_time in db_items:
                if db_desc and detection_time and db_time:
                    normalized_db = normalize_text(db_desc)
                    ratio = fuzz.ratio(normalized_detected, normalized_db)
                    
                    # CRITICAL: Only consider matches from the same day
                    if detection_time.date() == db_time.date():
                        # Only consider matches where:
                        # 1. The fuzzy match ratio is > 80
                        # 2. The confidence is > 0.5 (50%)
                        if ratio > highest_ratio and ratio > 60:
                            highest_ratio = ratio
                            best_match = (db_desc, db_time)
            
            if best_match:
                # Use db_time if we have a match (since we're already ensuring same-day matches)
                # This helps in case the time recording had slight differences
                final_time = best_match[1]
                
                matches.append({
                    'Detected_Item': detected_object,
                    'DB_Item': best_match[0],
                    'Detection_Time': final_time,
                    'Match_Confidence': highest_ratio
                })

        # Create DataFrame and drop duplicates to avoid reporting the same product multiple times
        if matches:
            matches_df = pd.DataFrame(matches)
            matches_df = matches_df.drop_duplicates(subset=['Detected_Item'])
            
            # Format the detection time for better readability
            matches_df['Detection_Time'] = pd.to_datetime(matches_df['Detection_Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save only the product and detection time to the final report
            final_report = matches_df[['Detected_Item', 'Detection_Time']]
            
            # Generate the report file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'product_detection_report_{timestamp}.csv'
            final_report.to_csv(report_filename, index=False)
            print(f"Product detection report generated: {report_filename}")
            print(f"Found {len(matches)} same-day products with confidence > 80%")

            # Send email with report
            email_address = os.getenv('EMAIL_ADDRESS')
            email_password = os.getenv('EMAIL_PASSWORD')
            recipient_email = os.getenv('RECIPIENT_EMAIL')

            if email_address and email_password and recipient_email:
                send_email_report(email_address, email_password, recipient_email, report_filename)
            else:
                print("Email credentials not fully configured. Skipping email.")
        else:
            print("No same-day matches found in the current time window")

    except Exception as e:
        print(f"Error during comparison: {e}")

def run_comparison_scheduler():
    """Run the comparison at specified intervals"""
    interval_minutes = int(os.getenv('COMPARISON_INTERVAL_MINUTES', 5))
    
    print(f"Starting comparison scheduler (interval: {interval_minutes} minutes)")
    print(f"NOTE: Only matching items detected on the SAME DAY as database entries")
    
    # Run initial comparison immediately
    compare_detections()
    
    schedule.every(interval_minutes).minutes.do(compare_detections)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run_comparison_scheduler()
