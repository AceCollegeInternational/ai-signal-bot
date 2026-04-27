import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def test_telegram():
    if not TOKEN or not CHAT_ID:
        print("Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")
        return

    print(f"Testing Telegram Bot with Token: {TOKEN[:10]}... and Chat ID: {CHAT_ID}")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "Hello from the Trading Bot test script! 🚀",
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("Successfully sent message!")
        else:
            print(f"Failed to send message. Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    test_telegram()
