import os
import dotenv
import datetime
import json
import time

dotenv.load_dotenv(".env")

TWILIO_NUMBER = "whatsapp:+18774467072"
TWILIO_SMS_NUMBER = "+18774467072"
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_MESSAGING_SERVICE_SID = "MGdadfd2c85ec7e22833d012852d8fa58a"

from twilio.rest import Client

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
curr_time = datetime.datetime.now(datetime.timezone.utc)

schedule_time = datetime.datetime(year=2025, month=7, day=21, hour=11) + datetime.timedelta(hours=48)
import json

all_msgs = json.load(open("message_logs.json"))
for k in all_msgs:
    message = twilio_client.messages.create(
            content_sid="HX2e855ed5ae297f2dbdd35a294b275079",
            to=k.replace("whatsapp:", ""),
            from_=TWILIO_SMS_NUMBER,
            messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
            send_at=schedule_time,
            schedule_type="fixed",
        )
    time.sleep(1)