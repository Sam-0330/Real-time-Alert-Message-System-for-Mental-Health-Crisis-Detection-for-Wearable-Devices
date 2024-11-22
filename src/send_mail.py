from flask import Flask, request
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

@app.route('/', methods=['POST'])
def send_vitals():
    # Fetch data from the incoming POST request
    state = request.form.get('state')
    emotion = request.form.get('emotion')
    heartbeat = request.form.get('heartbeat')
    message = request.form.get('message')
    blood_pressure = request.form.get('blood_pressure')

    # Compose the email content
    vitals = (
        f"Current State: {state}\n"
        f"Emotion Detected: {emotion}\n"
        f"Heartbeat: {heartbeat}\n"
        f"Blood Pressure: {blood_pressure}\n"
        f"Message: {message}"
    )

    # Send the email
    try:
        send_email(vitals)
        return "Vitals sent!", 200
    except Exception as e:
        return f"Failed to send vitals. Error: {str(e)}", 500

def send_email(vitals):
    # Compose the email
    msg = MIMEText(vitals)
    msg['Subject'] = 'User Vitals'
    msg['From'] = 'chennaimetrojava@gmail.com'
    msg['To'] = 'receiver@example.com'  # Replace with the recipient's email address

    # SMTP configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = 'chennaimetrojava@gmail.com'
    sender_password = 'dirnckiygrfsscnc'  # Ensure this is correct and valid

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Upgrade the connection to secure
        server.login(sender_email, sender_password)
        server.send_message(msg)

if __name__ == '__main__':
    app.run (debug=True, port=8000)
