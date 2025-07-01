import app_official
import json
import datetime

app_official.twilio_client = None

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--empathetic", action="store_true")
    args = parser.parse_args()

    BODY_TEMPLATE = {
        "From": "007",
        "MessageType": "text",
        "Body": "The name is Bond. James Bond."
    }

    
    exp_id_map = json.load(open("exp_id_map.json"))
    og_data = exp_id_map[BODY_TEMPLATE["From"]]
    if args.empathetic:
        og_data[1] = 1
    else:
        og_data[1] = 0
    og_data[2] = datetime.datetime.now().timestamp()
    exp_id_map[BODY_TEMPLATE["From"]] = og_data
    json.dump(exp_id_map, open("exp_id_map.json", "w+"))
    

    while True:
        user_input = input("USER >> ")
        body_t = BODY_TEMPLATE.copy()
        if user_input == "PING":
            body_t["MessageType"] = "button"
        else:
            body_t["Body"] = user_input
        twilio_client = None
        app_official.handle_whatsapp_message(body_t)