import sys
import os
import dspy
import dotenv
from argparse import ArgumentParser
import json

dotenv.load_dotenv("../.env")

base_lm = dspy.LM("openai/gpt-4o")

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

sys.path.insert(1, os.path.join(__location__, "../"))
from empathy_framework import EmpatheticResponder

MAINTENANCE_PROMPT = open(os.path.join(__location__, "../maintenance_prompt.txt")).read()

USER_SUMMARY = """- The user is feeling okay but is not as productive as hoped.  \n- The user unwinds by cooking, watching videos, and spending time with their boyfriend.  \n- The user enjoys reading and doing arts and crafts as sources of joy and recharge.  \n- The user lives in a city, which presents both opportunities and challenges for staying active.  \n- Time and lack of motivation are significant barriers to being more active for the user.  \n- Stress is also identified as a significant barrier to physical activity.  \n- The user is open to finding small opportunities throughout the day to add physical activity.  \n- The user is willing to try small changes like getting off public transport a stop early and taking short walks during breaks when working from home.  \n- The user believes being more active will lead to better health and more energy.  \n- On a scale of 0 to 10, the user rates the importance of being healthier and having more energy as an 8.  \n- The user plans to take walks in parts more as an action to become more active in the next 7 days.  \n- The user feels a confidence level of 7 in carrying out this plan.  \n- The counselor offers to follow up every two days for support, and the user agrees."""

MAINTENANCE_PROMPT = MAINTENANCE_PROMPT.replace("SESSION_SUMMARY", USER_SUMMARY)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--empathy", action="store_true")
    args = parser.parse_args()

    new_dialogue = [{"role": "system", "content": MAINTENANCE_PROMPT}, {"role": "system", "content": "Create a message to ping the user for your check-in. You check in with the user every other day. Make your message short and succinct, as in a text message."}]
    empathy_responder = EmpatheticResponder()

    ping_msg = base_lm(messages=new_dialogue)[0]
    print("ASSISTANT:", ping_msg)
    user_input = input("USER > ")
    new_dialogue.pop(1)
    new_dialogue.append({"role": "assistant", "content": ping_msg})
    new_dialogue.append({"role": "user", "content": user_input})

    dspy.configure(lm=base_lm)

    try:
        while True:
            if args.empathy:
                model_resp = empathy_responder.respond_empathetically(user_input, new_dialogue)[0]
            else:
                model_resp = base_lm(messages=new_dialogue)[0]
            print("ASSISTANT:", model_resp)
            new_dialogue.append({"role": "assistant", "content": model_resp})
            user_input = input("USER > ")
            new_dialogue.append({"role": "user", "content": user_input})
    except KeyboardInterrupt:
        if args.empathy:
            json.dump(new_dialogue, open("empathy_conversation.json", "w+"))
        else:
            json.dump(new_dialogue, open("conversation.json", "w+"))
        sys.exit(0)