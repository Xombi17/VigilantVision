import os
import cv2
import uuid
import sqlite3
import smtplib
import requests
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from src.config import DB_NAME, current_settings

# Global state for active alerts
alert_payload = None
alert_lock = threading.Lock()


def get_and_clear_alert():
    global alert_payload
    with alert_lock:
        temp = alert_payload
        alert_payload = None
        return temp


def trigger_alert(cam_id, cam_name, message, frame):
    global alert_payload
    try:
        print(f"ALERT: {message}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alerts/alert_{cam_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # Log to Database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        alert_id = str(uuid.uuid4())
        c.execute(
            "INSERT INTO alerts VALUES (?,?,?,?)",
            (alert_id, message, timestamp, filename),
        )
        conn.commit()
        conn.close()

        # Update payload for frontend websocket transmission
        with alert_lock:
            alert_payload = {
                "id": alert_id,
                "message": message,
                "timestamp": timestamp,
                "image_path": filename,
                "camera_id": cam_id,
            }

        # Fire-and-forget notification delivery thread
        threading.Thread(
            target=send_notifications, args=(message, filename), daemon=True
        ).start()

    except Exception as e:
        print(f"Alert Error: {e}")


def send_notifications(message, image_path):
    try:
        if current_settings.emailEnabled:
            sender_email = os.getenv("SENDER_EMAIL", current_settings.senderEmail)
            sender_password = os.getenv(
                "SMTP_PASSWORD", current_settings.senderPassword
            )
            if sender_email and sender_password:
                msg = MIMEMultipart()
                msg["From"] = sender_email
                msg["To"] = current_settings.receiverEmail
                msg["Subject"] = "VigilantVision AI - Security Alert"

                body = f"ALERT: {message}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                msg.attach(MIMEText(body, "plain"))

                try:
                    with open(image_path, "rb") as f:
                        img_data = f.read()
                        image = MIMEImage(img_data, name=os.path.basename(image_path))
                        msg.attach(image)
                except Exception as img_e:
                    print(f"Could not attach image: {img_e}")

                server = smtplib.SMTP(
                    current_settings.smtpServer, int(current_settings.smtpPort)
                )
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                server.quit()
                print("Email notification sent.")

        if current_settings.telegramEnabled:
            bot_token = os.getenv(
                "TELEGRAM_BOT_TOKEN", current_settings.telegramBotToken
            )
            chat_id = current_settings.telegramChatId
            if bot_token and chat_id:
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(image_path, "rb") as photo:
                    data = {
                        "chat_id": chat_id,
                        "caption": f"🚨 VIGILANTVISION ALERT 🚨\n\n{message}",
                    }
                    files = {"photo": photo}
                    resp = requests.post(url, data=data, files=files)
                if resp.status_code == 200:
                    print("Telegram notification sent.")
                else:
                    print(f"Telegram Error: {resp.text}")
    except Exception as e:
        print(f"Notification Error: {e}")
