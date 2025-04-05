import requests
from argparse import ArgumentParser
import json
import os
import sys
from datetime import datetime
import time

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

if __name__ == "__main__":
    while True:
        print("Current time", datetime.now().timestamp())
        # if not os.path.exists(os.path.join(__location__, "user_job_dict.json")):
        #     sys.exit(0)
        user_job_dict = json.load(open(os.path.join(__location__, "user_job_dict.json")))
        if os.path.exists(os.path.join(__location__, "done_ping_dict.json")):
            done_dict = json.load(open(os.path.join(__location__, "done_ping_dict.json"), "w+"))
        else:
            done_dict = {}

        # If it is within 5 minutes of the job

        for u in user_job_dict:
            # Summarize then ping
            curr_time = datetime.now().timestamp()
            if user_job_dict[u] != -1 and user_job_dict[u] <= curr_time and (u not in done_dict or done_dict[u] != user_job_dict[u]):
                print("Notifying:", u)
                resp = requests.get(f"http://127.0.0.1:5000/summarize?phone_number={u}")
                time.sleep(60)
                user_template_dict = json.load(open(os.path.join(__location__, "user_templates.json")))
                template = user_template_dict[u]
                template["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"] = "PING USER"
                template["entry"][0]["changes"][0]["value"]["messages"] = [template["entry"][0]["changes"][0]["value"]["messages"][0]]
                resp = requests.post(f"http://127.0.0.1:5000/webhook", json=template)
                done_dict[u] = user_job_dict[u]
                json.dump(done_dict, open(os.path.join(__location__, "done_ping_dict.json"), "w+"))
        time.sleep(600)

        # parser = ArgumentParser()
        # parser.add_argument("--summarize", action="store_true")
        # parser.add_argument("--ping", action="store_true")
        # parser.add_argument("--user_number", type=str)

        # args = parser.parse_args()
        
        # if args.summarize:
        #     requests.get(f"http://127.0.0.1:5000/summarize?phone_number={args.user_number}")
        # else:
        #     user_template_dict = json.load(open("user_templates.json"))
        #     template = user_template_dict[args.user_number]
        #     template["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"] = "PING USER"
        #     template["entry"][0]["changes"][0]["value"]["messages"] = [template["entry"][0]["changes"][0]["value"]["messages"][0]]
        #     requests.post(f"http://127.0.0.1:5000/webhook", json=template)
