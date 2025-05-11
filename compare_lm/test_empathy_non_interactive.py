import sys
import os
import dspy
import dotenv
from argparse import ArgumentParser
import json

dotenv.load_dotenv("../.env")

base_lm = dspy.LM("openai/gpt-4o")
user_lm = dspy.LM("openai/gpt-4o-mini")

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

sys.path.insert(1, os.path.join(__location__, "../"))
from empathy_framework import EmpatheticResponder

PROFILES = json.load(open("user_profiles.json"))

MAINTENANCE_PROMPT = open(os.path.join(__location__, "../maintenance_prompt.txt")).read()

USER_PROMPT = open("user_sim_prompt.txt").read()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--empathy", action="store_true")
    args = parser.parse_args()

    all_dialogues = {}

    empathy_responder = EmpatheticResponder()
    for p in PROFILES:
        maintenance_curr = MAINTENANCE_PROMPT.replace("SESSION_SUMMARY", PROFILES[p]["summary"])
        new_dialogue = [{"role": "system", "content": maintenance_curr}, {"role": "system", "content": "Create a message to ping the user for your check-in. You check in with the user every other day. Make your message short and succinct, as in a text message."}]
        lm_response = base_lm(messages=new_dialogue)[0]
        print("ASSISTANT:", lm_response)

        user_act = PROFILES[p]["current_situation"]
        user_act[0] = "- " + user_act[0]
        user_act = "\n- ".join(user_act)

        user_curr = USER_PROMPT.replace("SESSION_SUMMARY", PROFILES[p]["summary"]).replace("USER_ACTIVITY", user_act)

        user_dialogue = [{"role": "system", "content": user_curr},
                         {"role": "user", "content": lm_response}]
        
        user_input = user_lm(messages=user_dialogue)[0]

        print("USER:", user_input)
        new_dialogue.pop(1)
        new_dialogue.append({"role": "assistant", "content": lm_response})
        new_dialogue.append({"role": "user", "content": user_input})
        user_dialogue.append({"role": "user", "content": lm_response})
        user_dialogue.append({"role": "assistant", "content": user_input})

        dspy.configure(lm=base_lm)

        try:
            while "FINISHED" not in lm_response:
                if args.empathy:
                    lm_response, lm_info = empathy_responder.respond_empathetically(user_input, new_dialogue, return_dict=True)
                    lm_response = lm_response[0]
                else:
                    lm_response = base_lm(messages=new_dialogue)[0]
                print("ASSISTANT:", lm_response)
                new_dialogue.append({"role": "assistant", "content": lm_response})
                if args.empathy:
                    new_dialogue[-1].update(lm_info)
                user_dialogue.append({"role": "user", "content": lm_response})
                user_input = user_lm(messages=user_dialogue)[0]
                print("USER:", user_input)
                new_dialogue.append({"role": "user", "content": user_input})
                user_dialogue.append({"role": "assistant", "content": user_input})
        except KeyboardInterrupt:
            if args.empathy:
                json.dump(new_dialogue, open("empathy_conversation.json", "w+"))
            else:
                json.dump(new_dialogue, open("conversation.json", "w+"))
            sys.exit(0)
        
        all_dialogues.update({p: new_dialogue})
    if args.empathy:
        json.dump(all_dialogues, open("empathy_conversation.json", "w+"))
    else:
        json.dump(all_dialogues, open("conversation.json", "w+"))