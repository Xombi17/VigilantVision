import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# File paths and constants
SETTINGS_FILE = "settings.json"
DB_NAME = "vigilant_vision.db"
LOITERING_THRESHOLD = 30.0  # seconds in ROI before loitering alert fires
ALERT_COOLDOWN = 30.0       # minimum seconds between consecutive alerts per camera

# Create alert folder
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# Feature availability flags
try:
    import face_recognition

    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# Settings Model
class SettingsModel(BaseModel):
    emailEnabled: bool = False
    smtpServer: str = "smtp.gmail.com"
    smtpPort: str = "587"
    senderEmail: str = ""
    senderPassword: str = ""
    receiverEmail: str = ""
    telegramEnabled: bool = False
    telegramBotToken: str = ""
    telegramChatId: str = ""
    roiPoints: list[list[int]] = []
    showHeatmap: bool = False


# Load settings
try:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings_data = json.load(f)
            current_settings = SettingsModel(**settings_data)
            roi_points = current_settings.roiPoints
    else:
        current_settings = SettingsModel()
        roi_points = []
except Exception as e:
    current_settings = SettingsModel()
    roi_points = []
