import os
import json
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fastapi import APIRouter

from src.config import SETTINGS_FILE, SettingsModel, current_settings

router = APIRouter(tags=["Settings"])


@router.get("/settings")
async def get_settings():
    return current_settings


@router.post("/settings")
async def save_settings(settings: SettingsModel):
    global current_settings
    import src.config

    src.config.current_settings = settings
    current_settings = settings

    from dotenv import set_key

    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, "a").close()

    if settings.senderPassword and settings.senderPassword != "********":
        set_key(env_path, "SMTP_PASSWORD", settings.senderPassword)
    if settings.telegramBotToken and settings.telegramBotToken != "********":
        set_key(env_path, "TELEGRAM_BOT_TOKEN", settings.telegramBotToken)

    safe_settings = settings.dict()
    safe_settings["senderPassword"] = ""
    safe_settings["telegramBotToken"] = ""

    with open(SETTINGS_FILE, "w") as f:
        json.dump(safe_settings, f, indent=4)

    load_dotenv(override=True)
    return {"status": "success", "message": "Settings saved"}


@router.post("/roi")
async def save_roi(data: dict):
    global current_settings
    import src.config

    if "points" in data:
        points = data["points"]
        current_settings.roiPoints = points
        src.config.current_settings.roiPoints = points
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current_settings.dict(), f, indent=4)
        print(f"ROI Updated: {points}")
        return {"status": "success"}
    return {"status": "error"}


@router.get("/roi")
async def get_roi():
    return {"points": current_settings.roiPoints}


@router.post("/settings/test")
async def test_settings(settings: SettingsModel):
    if settings.emailEnabled:
        try:
            msg = MIMEMultipart()
            msg["From"] = settings.senderEmail
            msg["To"] = settings.receiverEmail
            msg["Subject"] = "VigilantVision - Test Email"
            msg.attach(
                MIMEText(
                    "This is a test email from your VigilantVision Security System.",
                    "plain",
                )
            )
            server = smtplib.SMTP(settings.smtpServer, int(settings.smtpPort))
            server.starttls()
            server.login(settings.senderEmail, settings.senderPassword)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            return {"status": "error", "message": f"Email Test Failed: {str(e)}"}

    if settings.telegramEnabled:
        try:
            url = f"https://api.telegram.org/bot{settings.telegramBotToken}/sendMessage"
            data = {
                "chat_id": settings.telegramChatId,
                "text": "VigilantVision - Test Message",
            }
            resp = requests.post(url, data=data)
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Telegram Test Failed: {resp.text}",
                }
        except Exception as e:
            return {"status": "error", "message": f"Telegram Test Failed: {str(e)}"}

    return {"status": "success", "message": "All enabled tests sent successfully!"}
