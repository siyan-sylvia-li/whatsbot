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
from argparse import ArgumentParser
import dotenv
import datetime
import uuid
import time

import random
random.seed(42)

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

from stress_relief import StressReliefModule

# in-memory storage imports and configurations
import faiss
import numpy as np
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_FILE = "chat_index.faiss"
METADATA_FILE = "chat_metadata.json"
dim = 1536
faiss_index = None

# in-memory storage file creations
if os.path.exists(INDEX_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
else: 
    faiss_index = faiss.IndexFlat(dim) 
    

if not os.path.exists(METADATA_FILE): 
    json.dump({}, open(METADATA_FILE, "w+"))
chat_metadata = json.load(open(METADATA_FILE))


bot = StressReliefModule(
        profile_path="user_profiles.json",
        openai_api_key=os.environ["OPENAI_API_KEY"],  # Pass the API key directly
        favorite_threshold=7.0  # Use a custom threshold
    )

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

all_scheduled_messages = []


# EXP ID dict
if not os.path.exists("exp_id_map.json"):
    json.dump({}, open("exp_id_map.json", "w+"))
exp_id_map = json.load(open("exp_id_map.json"))

# language for speech to text recoginition
# TODO: detect this automatically based on the user's language
LANGUGAGE = "en-US"

INITIAL_PROMPT = open("initial_prompt.txt").read()
SUMMARY_PROMPT = open("summarization_prompt.txt").read()
PING_PROMPT = open("ping_prompt.txt").read()
MAINTENANCE_PROMPT = open("maintenance_prompt.txt").read()

TWILIO_NUMBER = "whatsapp:+18774467072"
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_MESSAGING_SERVICE_SID = "MGdadfd2c85ec7e22833d012852d8fa58a"

from twilio.rest import Client

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

def save_state(): 
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(METADATA_FILE, "w") as f: 
        json.dump(chat_metadata, f, indent=2)

def get_embedding(text: str): 
    return client.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding

def split_message_by_period(text, max_length=1600):
    """
        Twilio cannot send messages over 1600 characters. Instead of breaking it up into chunks mid sentence, this function finds the last period (full stop) in the sentence and adds that as a chunk. This makes the message more readable. This function is only called if the message is greater than 1600 characters. 
    """
    chunks = []
    start = 0
    
    while start < len(text): 
        # get the minimum between start indx + max_length and len(text) to determine the window
        end = min(start + max_length, len(text))
        
        # find the last period from start and end indexes
        period_index = text.rfind('.', start, end)
        
        # no period_index found in the search, so period_index will be the end
        if period_index == -1 or period_index <= start: 
            period_index = end
        else: 
            # include the period in the text
            period_index += 1
        
        # append and remove any whitespaces
        chunks.append(text[start:period_index].strip())
        # continue doing the same for the next characters
        start = period_index
    
    return chunks

# send the response as a WhatsApp message back to the user
def send_whatsapp_message(body, message):
    # if message length is greater than 1600, break it into readable chunks. More info is provided in the split_message_by_period function
    if len(message) > 1600: 
        chunks = split_message_by_period(message)
        for chunk in chunks: 
            twilio_msg_sid = twilio_client.messages.create(
                to=body["From"],
                from_=TWILIO_NUMBER,
                body=chunk
            )
            print(f"Sent chunk, sid: {twilio_msg_sid}")
    else: 
        twilio_msg_sid = twilio_client.messages.create(
            to=body["From"],
            from_=TWILIO_NUMBER,
            body=message
        )
        
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
def make_openai_request(message, from_number, non_empathetic=False):
    try:
        message_log = update_message_log(message, from_number, "user")
        if non_empathetic:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=message_log + [{"role": "system", "content": "You should be very professional and cold when responding. Do not be empathetic."}],
                temperature=1.0,
            )
        else:
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
    global stress_relief_dict
    stress_relief_dict = json.load(open("stress_relief_logs.json"))
    try:
        message_log = update_message_log(message, from_number, "user")
        if stress_relief_dict[from_number] == 1:
            print("ROUND 2 STRESS RELIEF CONVERSATION")
            response_message = bot.process_message(message, from_number)["text"]
            stress_relief_dict.update({from_number: 2})
            json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
        elif stress_relief_dict[from_number] == 2:
            print("ROUND 3 STRESS RELIEF CONVERSATION")
            response_message = bot.handle_feedback(message, from_number)["text"]
            convo_history = copy.copy(message_log)
            convo_history.append({"role": "assistant", "content": response_message})
            convo_history.append({"role": "system", "content": "Rewrite your last message to incorporate content earlier in the conversation and to say goodbye to the user."})
            response_message = openai_lm(messages=convo_history)[0]
            stress_relief_dict.update({from_number: False})
            json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
        print(f"stress relief response: {response_message}")
        print(stress_relief_dict[from_number])
        update_message_log(response_message, from_number, "assistant")
    except Exception as e:
        print(f"openai error: {e}")
        response_message = "Sorry, the OpenAI API is currently overloaded or offline. Please try again later."
        remove_last_message_from_log(from_number)
    return response_message

# Handle the specific case of pinging user
def create_ping(from_number):
    summarize_session(from_number)
    ping_msg = PING_PROMPT.replace("NUM_SESSIONS", str(session_log_dict[from_number]["current_session"]))
    if session_log_dict[from_number]["current_session"] > 1:
        ping_msg = ping_msg.replace("SESSION_SINGULAR_PLURAL", "sessions")
    else:
        ping_msg = ping_msg.replace("SESSION_SINGULAR_PLURAL", "session")
    ping_msg = ping_msg.replace("SESSION_SUMMARY", session_log_dict[from_number]["session_summaries"][-1])
    if from_number in exp_id_map:
        _, exp_condition = exp_id_map[from_number]
    else:
        exp_condition = 0
    try:
        if exp_condition == 0:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": ping_msg}, {"role": "system", "content": "You should be very professional and cold when responding. Do not be empathetic."}],
                temperature=0.2,
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": ping_msg}],
                temperature=0.8,
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
    user_job_dict.update({from_number: -1})
    json.dump(user_job_dict, open("user_job_dict.json", "w+"))
    return response_message

# handle WhatsApp messages of different type
def handle_whatsapp_message(body):
    if body["MessageType"] == "text":
        message_body = body["Body"]
        if "EXP_ID" in message_body:
            exp_id = message_body.replace("EXP_ID", "").strip()
            # exp_condition = random.choice([0, 1, 2])
            exp_condition = 1
            exp_id_map.update({
                body["From"]: (exp_id, exp_condition)
            })
            json.dump(exp_id_map, open("exp_id_map.json", "w+"))
            message_body = "Hi"
        else:
            _, exp_condition = exp_id_map[body["From"]]
    elif body["MessageType"] == "button":
        response = create_ping(body["From"])
        send_whatsapp_message(body, response)
        return
    print("PASS 1")
    if stress_relief_dict.get(body["From"], False):
        response = make_stress_relief_response(message_body, body["From"])
    elif body["From"] in session_log_dict and session_log_dict[body["From"]]["current_session"] > 1:
        if exp_condition == 0:
            response = make_openai_request(message_body, body["From"], non_empathetic=True)
        elif exp_condition == 1:
            response = make_openai_request(message_body, body["From"])
        elif exp_condition == 2:
            response = make_empathetic_response(message_body, body["From"])
        if "FINISHED" in response and body["From"] in stress_relief_dict:
            # Go into stress relief workflow
            stress_relief_dict[body["From"]] = True
            json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
            # TODO: Directly ask here if the user is stressed
            msg_logs = message_log_dict[body["From"]]["current_session"]
            convo_history = copy.copy(msg_logs)
            convo_history.pop(0)
            convo_history.append({"role": "system", "content": "Rewrite your last message so that you check with the user to determine whether they are stressed."})
            response = openai_lm(messages=convo_history)[0]
            message_log_dict[body["From"]]["current_session"][-1]["content"] = response
            stress_relief_dict.update({body["From"]: 1})
            json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
    else:
        if exp_condition == 0:
            response = make_openai_request(message_body, body["From"], non_empathetic=True)
        else:
            response = make_openai_request(message_body, body["From"])
        
        if "FINISHED" in response and body["From"] in stress_relief_dict:
            # Go into stress relief workflow
            stress_relief_dict[body["From"]] = True
            json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
            # TODO: Directly ask here if the user is stressed
            msg_logs = message_log_dict[body["From"]]["current_session"]
            convo_history = copy.copy(msg_logs)
            convo_history.pop(0)
            if exp_condition == 0:
                convo_history.append({"role": "system", "content": "Rewrite your last message so that you check with the user to determine whether they are stressed. Be professional and cold when responding. Do not be empathetic."})
            else:
                convo_history.append({"role": "system", "content": "Rewrite your last message so that you check with the user to determine whether they are stressed."})
            
            response = openai_lm(messages=convo_history)[0]
            message_log_dict[body["From"]]["current_session"][-1]["content"] = response
            stress_relief_dict.update({body["From"]: 1})
            json.dump(stress_relief_dict, open("stress_relief_logs.json", "w+"))
    print("PASS 2")
    
    if "FINISHED" in response:
        response = response.replace("FINISHED.", "").replace("FINISHED", "")
    
    # summarize session and store the summary of the convo in vectorDB
    # current_session_messages = copy.copy(message_log_dict[body["From"]]["current_session"])
    # formatted_convo = format_conversation(current_session_messages)
    # summary_response = openai.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[{"role": "user", "content": SUMMARY_PROMPT + "\n\n" + formatted_convo}],
    #         temperature=1.0,
    #     )
    
    curr_time = datetime.datetime.now(datetime.timezone.utc)
    """
    NOTE-UNCOMMENT LINE 405-418 WHEN RUNNING CHAT APP FOR THE FIRST TIME
    THIS STORES THE SUMMARY SO IT CAN QUERY LATER. 
    FILE CREATION AND INITIALIZATION TAKES PLACE AT THE TOP.
    """
    # response_message = summary_response.choices[0].message.content.strip()
    # embedding = get_embedding(response_message)
    # faiss_index.add(np.array([embedding]).astype("float32"))
    # chat_metadata[str(faiss_index.ntotal-1)] = {
    #     "session_id": str(uuid.uuid4()),
    #     "user_id": body["From"],
    #     "timestamp": curr_time.isoformat(),
    #     "summary": response_message
    # }
    # save_state()
    # print(">>> printing response_message", response_message)
    # print(">>> printing embedding", embedding)
    # print(">>> print chat_metadata", chat_metadata)
    
    start = time.time()
    query = "Is the user managing stress?"
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    D, I = faiss_index.search(query_embedding, k=3)
    for idx in I[0]: 
        try: 
            session = chat_metadata[str(idx)]
            print(">>> printing session")
            print(session)
        except Exception as e:
            print(e)
            continue 
    end = time.time()
    latency = end-start
    print(f">>> FAISS search took {latency:.4f} seconds")
    send_whatsapp_message(body, response)
    # Set up scheduling
    
    # datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    # if user_job_dict[body["From"]] == -1:
        # print(">>> user_job_dict === -1")
        # if args.short:
        #     schedule_time = curr_time + datetime.timedelta(minutes=12)
        # else:
        #     schedule_time = curr_time + datetime.timedelta(hours=48)
        # message = twilio_client.messages.create(
        #     content_sid="HX19c29fcf2aa11e5a484bf0542a65a572",
        #     to=body["From"],
        #     from_=TWILIO_NUMBER,
        #     messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
        #     send_at=schedule_time,
        #     schedule_type="fixed",
        # )
        # all_scheduled_messages.append(message)
        # print("SCHEDULED MESSAGE:", message)
        # user_job_dict.update({body["From"]: schedule_time.timestamp()})
    
    schedule_time = curr_time + datetime.timedelta(minutes=10)
    message = twilio_client.messages.create(
        content_sid="HX19c29fcf2aa11e5a484bf0542a65a572",
        to=body["From"],
        from_=TWILIO_NUMBER,
        messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
        send_at=schedule_time,
        schedule_type="fixed",
    )
    all_scheduled_messages.append(message)
    print("SCHEDULED MSG: ", message)
    json.dump(user_job_dict, open("user_job_dict.json", "w+"))


def cancel_all_scheduled_messages():
    if len(all_scheduled_messages) == 0:
        print("No messages to cancel")
    for m in all_scheduled_messages:
        message = twilio_client.messages(m.sid).update(
            status="canceled"
        )
        print("CANCELLED:", message)



# handle incoming webhook messages
def handle_message(request):
    # Parse Request body in json format
    body = request.form.to_dict()
    print(f"request body: {body}")

    try:
        # info on WhatsApp text message payload:
        # https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#text-messages
        if body.get("SmsStatus") == "received":
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
    print(request)
    try:
        if request.method == "GET":
            return verify(request)
        elif request.method == "POST":
            return handle_message(request)
    except Exception as e:
        print(e)
    return jsonify({"status": "ok"}, 200)


# Route to reset message log
@app.route("/reset", methods=["GET"])
def reset():
    global message_log_dict
    message_log_dict = {}
    json.dump({}, open("message_logs.json", "w+"))
    return "Message log resetted!"

def summarize_session(phone_number):
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
    import atexit

    atexit.register(cancel_all_scheduled_messages)

    parser = ArgumentParser()
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--empathy", action="store_true")
    args = parser.parse_args()
    
    app.run(port=5050, debug=True, use_reloader=True)
