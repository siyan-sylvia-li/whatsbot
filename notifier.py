import requests
from argparse import ArgumentParser
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--user_number", type=str)

    args = parser.parse_args()
    
    if args.summarize:
        requests.get(f"http://127.0.0.1:5000/summarize?phone_number={args.user_number}")
    else:
        user_template_dict = json.load(open("user_templates.json"))
        template = user_template_dict[args.user_number]
        template["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"] = "PING USER"
        template["entry"][0]["changes"][0]["value"]["messages"] = [template["entry"][0]["changes"][0]["value"]["messages"][0]]
        requests.post(f"http://127.0.0.1:5000/webhook", json=template)