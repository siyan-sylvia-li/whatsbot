import io
import os

import openai
import pydub
import requests
import soundfile as sf
import speech_recognition as sr
from flask import Flask, jsonify, request
import json
import copy
from scheduler import schedule_job, cancel_job
from argparse import ArgumentParser
import dotenv
import datetime

dotenv.load_dotenv(".env")

app = Flask(__name__)


# OpenAi API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# print(os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from empathy_framework import EmpatheticResponder

import dspy
openai_lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), max_tokens=4000)
    
dspy.configure(lm=openai_lm)

empathy_responder = EmpatheticResponder()

client = openai.OpenAI()

__p_location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Access token for your WhatsApp business account app
whatsapp_token = os.environ.get("WHATSAPP_TOKEN")

# Verify Token defined when configuring the webhook
verify_token = os.environ.get("VERIFY_TOKEN")

# Message log dictionary to enable conversation over multiple messages
if not os.path.exists("message_logs.json"):
    json.dump({}, open("message_logs.json", "w+"))
message_log_dict = json.load(open("message_logs.json"))

# Session logs
if not os.path.exists("session_logs.json"):
    json.dump({}, open("session_logs.json", "w+"))
session_log_dict = json.load(open("session_logs.json"))

# User JSON body template
if not os.path.exists("user_templates.json"):
    json.dump({}, open("user_templates.json", "w+"))
user_template_dict = json.load(open("user_templates.json"))

# Stress relief dict
if not os.path.exists("stress_relief_logs.json"):
    json.dump({}, open("stress_relief_logs.json", "w+"))
stress_relief_dict = json.load(open("stress_relief_logs.json"))

user_job_dict = {}


# language for speech to text recoginition
# TODO: detect this automatically based on the user's language
LANGUGAGE = "en-US"

INITIAL_PROMPT = open("initial_prompt.txt").read()
SUMMARY_PROMPT = open("summarization_prompt.txt").read()
PING_PROMPT = open("ping_prompt.txt").read()
MAINTENANCE_PROMPT = open("maintenance_prompt.txt").read()


# get the media url from the media id
def get_media_url(media_id):
    headers = {
        "Authorization": f"Bearer {whatsapp_token}",
    }
    url = f"https://graph.facebook.com/v16.0/{media_id}/"
    response = requests.get(url, headers=headers)
    print(f"media id response: {response.json()}")
    return response.json()["url"]


# download the media file from the media url
def download_media_file(media_url):
    headers = {
        "Authorization": f"Bearer {whatsapp_token}",
    }
    response = requests.get(media_url, headers=headers)
    print(f"first 10 digits of the media file: {response.content[:10]}")
    return response.content


# convert ogg audio bytes to audio data which speechrecognition library can process
def convert_audio_bytes(audio_bytes):
    ogg_audio = pydub.AudioSegment.from_ogg(io.BytesIO(audio_bytes))
    ogg_audio = ogg_audio.set_sample_width(4)
    wav_bytes = ogg_audio.export(format="wav").read()
    audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="int32")
    sample_width = audio_data.dtype.itemsize
    print(f"audio sample_rate:{sample_rate}, sample_width:{sample_width}")
    audio = sr.AudioData(audio_data, sample_rate, sample_width)
    return audio


# run speech recognition on the audio data
def recognize_audio(audio_bytes):
    recognizer = sr.Recognizer()
    audio_text = recognizer.recognize_google(audio_bytes, language=LANGUGAGE)
    return audio_text


# handle audio messages
def handle_audio_message(audio_id):
    audio_url = get_media_url(audio_id)
    audio_bytes = download_media_file(audio_url)
    audio_data = convert_audio_bytes(audio_bytes)
    audio_text = recognize_audio(audio_data)
    message = (
        "Please summarize the following message in its original language "
        f"as a list of bullet-points: {audio_text}"
    )
    return message


# send the response as a WhatsApp message back to the user
def send_whatsapp_message(body, message):
    value = body["entry"][0]["changes"][0]["value"]
    phone_number_id = value["metadata"]["phone_number_id"]
    from_number = value["messages"][0]["from"]
    headers = {
        "Authorization": f"Bearer {whatsapp_token}",
        "Content-Type": "application/json",
    }
    url = "https://graph.facebook.com/v15.0/" + phone_number_id + "/messages"
    data = {
        "messaging_product": "whatsapp",
        "to": from_number,
        "type": "text",
        "text": {"body": message},
    }
    user_template_dict.update({from_number: body})
    json.dump(user_template_dict, open("user_templates.json", "w+"))
    response = requests.post(url, json=data, headers=headers)
    print(f"whatsapp message response: {response.json()}")
    response.raise_for_status()


# create a message log for each phone number and return the current message log
def update_message_log(message, phone_number, role):
    if phone_number not in message_log_dict:
        initial_log = {
            "role": "system",
            "content": INITIAL_PROMPT,
        }
        message_log_dict[phone_number] = {"current_session": [initial_log]}
    if phone_number not in session_log_dict:
        session_log_dict[phone_number] = {
            "current_session": 1,
            "session_summaries": []
        }
        json.dump(session_log_dict, open("session_logs.json", "w+"))
    if phone_number not in user_job_dict:
        user_job_dict.update({
            phone_number: -1
        })
        json.dump(user_job_dict, open("user_job_dict.json", "w+"))
    message_log = {"role": role, "content": message}
    message_log_dict[phone_number]["current_session"].append(message_log)
    json.dump(message_log_dict, open("message_logs.json", "w+"))
    return message_log_dict[phone_number]["current_session"]


# remove last message from log if OpenAI request fails
def remove_last_message_from_log(phone_number):
    message_log_dict[phone_number]["current_session"].pop()


# make request to OpenAI
def make_openai_request(message, from_number):
    try:
        message_log = update_message_log(message, from_number, "user")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message_log,
            temperature=1.0,
        )
        response_message = response.choices[0].message.content
        print(f"openai response: {response_message}")
        update_message_log(response_message, from_number, "assistant")
    except Exception as e:
        print(f"openai error: {e}")
        response_message = "Sorry, the OpenAI API is currently overloaded or offline. Please try again later."
        remove_last_message_from_log(from_number)
    return response_message


def make_empathetic_response(message, from_number):
    try:
        message_log = update_message_log(message, from_number, "user")
        convo_history = copy.copy(message_log)
        response_message = empathy_responder.respond_empathetically(user_input=message, convo_history=convo_history)[0]
        print(f"empathetic response: {response_message}")
        update_message_log(response_message, from_number, "assistant")
    except Exception as e:
        print(f"openai error: {e}")
        response_message = "Sorry, the OpenAI API is currently overloaded or offline. Please try again later."
        remove_last_message_from_log(from_number)
    return response_message

def make_stress_relief_response(message, from_number):
    try:
        message_log = update_message_log(message, from_number, "user")
        convo_history = copy.copy(message_log)
        response_message = "STRESS_RELIEF_PLACEHOLDER"
        print(f"stress relief response: {response_message}")
        update_message_log(response_message, from_number, "assistant")
    except Exception as e:
        print(f"openai error: {e}")
        response_message = "Sorry, the OpenAI API is currently overloaded or offline. Please try again later."
        remove_last_message_from_log(from_number)
    return response_message

# Handle the specific case of pinging user
def create_ping(from_number):
    ping_msg = PING_PROMPT.replace("NUM_SESSIONS", str(session_log_dict[from_number]["current_session"]))
    if session_log_dict[from_number]["current_session"] > 1:
        ping_msg = ping_msg.replace("SESSION_SINGULAR_PLURAL", "sessions")
    else:
        ping_msg = ping_msg.replace("SESSION_SINGULAR_PLURAL", "session")
    ping_msg = ping_msg.replace("SESSION_SUMMARY", session_log_dict[from_number]["session_summaries"][-1])
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": ping_msg}],
            temperature=1.0,
        )
        response_message = response.choices[0].message.content
        print(f"openai response: {response_message}")
        update_message_log(MAINTENANCE_PROMPT.replace("SESSION_SUMMARY", session_log_dict[from_number]["session_summaries"][-1]), from_number, "system")
        update_message_log(response_message, from_number, "assistant")
        session_log_dict[from_number]["current_session"] += 1
        json.dump(session_log_dict, open("session_logs.json", "w+"))
    except Exception as e:
        print(f"openai error: {e}")
        response_message = "Sorry, the OpenAI API is currently overloaded or offline. Please try again later."
        remove_last_message_from_log(from_number)
    # Turn the stress relief state to false
    if from_number in stress_relief_dict:
        stress_relief_dict[from_number] = False
        json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
    # Create another ping in 15 minutes if the user has not responded
    curr_time = datetime.datetime.now()
    if args.short:
        user_job_dict.update({from_number: (curr_time + datetime.timedelta(minutes=15)).timestamp()})
    else:
        if user_job_dict[from_number] == -1:
            user_job_dict.update({from_number: (curr_time + datetime.timedelta(hours=48)).timestamp()})
    json.dump(user_job_dict, open("user_job_dict.json", "w+"))
    return response_message


# handle WhatsApp messages of different type
def handle_whatsapp_message(body):
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    if message["type"] == "text":
        # TODO: Have a specific case for CRON triggers
        message_body = message["text"]["body"]
        if message_body == "PING USER":
            # When pinging, incorporate previous user session
            response = create_ping(message["from"])
            send_whatsapp_message(body, response)
            return
    elif message["type"] == "audio":
        audio_id = message["audio"]["id"]
        message_body = handle_audio_message(audio_id)
    if stress_relief_dict.get(message["from"], False):
        response = make_stress_relief_response(message_body, message["from"])
    elif "EMPATHY" in message_body or args.empathy or session_log_dict[message["from"]]["current_session"] > 1:
        message_body = message_body.replace("EMPATHY", "")
        response = make_empathetic_response(message_body, message["from"])
    else:
        response = make_openai_request(message_body, message["from"])
    if "FINISHED" in response and message["from"] in stress_relief_dict:
        # Go into stress relief workflow
        stress_relief_dict[message["from"]] = True
        json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
        
    send_whatsapp_message(body, response)
    # Set up scheduling
    # notifier_script_path = os.path.join(__p_location__, "notifier.py")
    # msg_from = message["from"]
    curr_time = datetime.datetime.now()
    if args.short:
        # if user_job_dict[message["from"]] != (-1, -1):
        #     cancel_job(user_job_dict[message["from"]][0])
        #     cancel_job(user_job_dict[message["from"]][1])
        # new_job_num_ping = schedule_job(f"python3 {notifier_script_path} --ping --user_number={msg_from}", "15 minutes")
        # new_job_num_sum = schedule_job(f"python3 {notifier_script_path} --summarize --user_number={msg_from}", "10 minutes")
        user_job_dict.update({message["from"]: (curr_time + datetime.timedelta(minutes=15)).timestamp()})
    else:
        if user_job_dict[message["from"]] == -1:
            # new_job_num_ping = schedule_job(f"python3 {notifier_script_path} --ping --user_number={msg_from}", "48 hours")
            # new_job_num_sum = schedule_job(f"python3 {notifier_script_path} --summarize --user_number={msg_from}", "47 hours")
            user_job_dict.update({message["from"]: (curr_time + datetime.timedelta(hours=48)).timestamp()})
    json.dump(user_job_dict, open("user_job_dict.json", "w+"))



# handle incoming webhook messages
def handle_message(request):
    # Parse Request body in json format
    body = request.get_json()
    print(f"request body: {body}")

    try:
        # info on WhatsApp text message payload:
        # https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#text-messages
        if body.get("object"):
            if (
                body.get("entry")
                and body["entry"][0].get("changes")
                and body["entry"][0]["changes"][0].get("value")
                and body["entry"][0]["changes"][0]["value"].get("messages")
                and body["entry"][0]["changes"][0]["value"]["messages"][0]
            ):
                handle_whatsapp_message(body)
            return jsonify({"status": "ok"}), 200
        else:
            # if the request is not a WhatsApp API event, return an error
            return (
                jsonify({"status": "error", "message": "Not a WhatsApp API event"}),
                404,
            )
    # catch all other errors and return an internal server error
    except Exception as e:
        print(f"unknown error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Required webhook verifictaion for WhatsApp
# info on verification request payload:
# https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests
def verify(request):
    # Parse params from the webhook verification request
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == "subscribe" and token == verify_token:
            # Respond with 200 OK and challenge token from the request
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            # Responds with '403 Forbidden' if verify tokens do not match
            print("VERIFICATION_FAILED")
            return jsonify({"status": "error", "message": "Verification failed"}), 403
    else:
        # Responds with '400 Bad Request' if verify tokens do not match
        print("MISSING_PARAMETER")
        return jsonify({"status": "error", "message": "Missing parameters"}), 400


def format_conversation(msgs):
    msg_str = ""
    for m in msgs:
        if "role" not in m or m["role"] == "system":
            continue
        if m["role"] == "assistant":
            msg_str += "Counselor: " + m["content"] + "\n"
        else:
            msg_str += "User: " + m["content"] + "\n"
    return msg_str


# Sets homepage endpoint and welcome message
@app.route("/", methods=["GET"])
def home():
    return "WhatsApp OpenAI Webhook is listening!"


# Accepts POST and GET requests at /webhook endpoint
@app.route("/webhook", methods=["POST", "GET"])
def webhook():
    if request.method == "GET":
        return verify(request)
    elif request.method == "POST":
        return handle_message(request)


# Route to reset message log
@app.route("/reset", methods=["GET"])
def reset():
    global message_log_dict
    message_log_dict = {}
    json.dump({}, open("message_logs.json", "w+"))
    return "Message log resetted!"


@app.route("/summarize", methods=["GET"])
def summarize_session():
    phone_number = request.args.get("phone_number")
    print("Obtained phone number", phone_number)
    current_session_messages = []
    
    if len(message_log_dict[phone_number]["current_session"]):
        current_session_messages = copy.copy(message_log_dict[phone_number]["current_session"])
        formatted_convo = format_conversation(current_session_messages)
        response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": SUMMARY_PROMPT + "\n\n" + formatted_convo}],
                temperature=1.0,
            )
        response_message = response.choices[0].message.content
        print(response_message)
        # Find whether stress is a signficant barrier
        if session_log_dict[phone_number]["current_session"] == 1:
            stress_judge = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Does the user think stress is a significant barrier for them in terms of increasing physical activity? Answer with yes or no." + "\n\n" + formatted_convo}],
                temperature=0,
            )
            if stress_judge.choices[0].message.content.lower().startswith("yes"):
                stress_relief_dict.update({
                    phone_number: False
                })
        if phone_number in session_log_dict:
            session_log_dict[phone_number]["session_summaries"].append(response_message)
        session_num = f"session_" + str(session_log_dict[phone_number]["current_session"])
        message_log_dict[phone_number].update({
            session_num: current_session_messages
        })
        message_log_dict[phone_number]["current_session"] = []
        json.dump(session_log_dict, open("session_logs.json", "w+"))
        json.dump(message_log_dict, open("message_logs.json", "w+"))
    user_job_dict.update({phone_number: (-1, -1)})
    return f"Session summarized for {phone_number}!"
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--empathy", action="store_true")
    args = parser.parse_args()
    
    app.run(debug=True, use_reloader=True)
