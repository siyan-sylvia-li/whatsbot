import sys
import os
import dspy
import dotenv
from argparse import ArgumentParser
import json

dotenv.load_dotenv("../.env")

base_lm = dspy.LM("openai/gpt-4o")
gpt_35 = dspy.LM("openai/gpt-3.5-turbo")
gpt_41 = dspy.LM("openai/gpt-4.1")

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

sys.path.insert(1, os.path.join(__location__, "../"))
from empathy_framework import EmpatheticResponder

MESSAGES = json.load(open("empathy_comparison_texts.json"))

NE_SYSTEM_PROMPT = "Respond to the following user post. Be professional and succinct. Do not be empathetic."

SYSTEM_PROMPT = "Respond to the following user post."

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--empathy", action="store_true")
    args = parser.parse_args()

    all_messages = []
    all_emp_resp = []
    all_ne_resp = []
    all_35_resp = []
    all_35_ne_resp = []
    all_41_resp = []
    all_41_ne_resp = []

    empathy_responder = EmpatheticResponder()
    dspy.configure(lm=base_lm)
    for p in MESSAGES:
        user_input = p["message"]
        all_messages.append(user_input)
        emp_lm_response, lm_info = empathy_responder.respond_empathetically(user_input, [{"role": "user", "content": user_input}], return_dict=True)
        emp_lm_response = emp_lm_response[0]
        all_emp_resp.append(emp_lm_response)
        print(user_input)
        print(emp_lm_response)
        print("================")

        curr_ne_dialogue = [{"role": "system", "content": NE_SYSTEM_PROMPT}, {"role": "user", "content": user_input}]
        curr_dialogue = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_input}]

        # input(base_lm(messages=curr_ne_dialogue))
        # all_ne_resp.append(base_lm(messages=curr_ne_dialogue)[0])
        # all_35_resp.append(gpt_35(messages=curr_dialogue)[0])
        # all_35_ne_resp.append(gpt_35(messages=curr_ne_dialogue)[0])
        # all_41_resp.append(gpt_41(messages=curr_dialogue)[0])
        # all_41_ne_resp.append(gpt_41(messages=curr_ne_dialogue)[0])

    # make the pandas dataframes
    import pandas
    emp_df = pandas.DataFrame([])
    emp_df["seeker_post"] = all_messages
    emp_df["response_post"] = all_emp_resp
    emp_df.to_csv("empathetic_resps.csv")

    # emp_df = pandas.DataFrame([])
    # emp_df["seeker_post"] = all_messages
    # emp_df["response_post"] = all_ne_resp
    # emp_df.to_csv("4o_ne_resps.csv")
    
    # emp_df = pandas.DataFrame([])
    # emp_df["seeker_post"] = all_messages
    # emp_df["response_post"] = all_35_resp
    # emp_df.to_csv("35_resps.csv")

    # emp_df = pandas.DataFrame([])
    # emp_df["seeker_post"] = all_messages
    # emp_df["response_post"] = all_35_ne_resp
    # emp_df.to_csv("35_ne_resps.csv")

    # emp_df = pandas.DataFrame([])
    # emp_df["seeker_post"] = all_messages
    # emp_df["response_post"] = all_41_resp
    # emp_df.to_csv("41_resps.csv")

    # emp_df = pandas.DataFrame([])
    # emp_df["seeker_post"] = all_messages
    # emp_df["response_post"] = all_41_ne_resp
    # emp_df.to_csv("41_ne_resps.csv")