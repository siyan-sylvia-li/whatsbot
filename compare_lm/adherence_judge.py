import dspy
import dotenv
import json

dotenv.load_dotenv("../.env")

class DecideAdherence(dspy.Signature):
    """Determine whether a response to a conversation adheres to the strategies provided."""
    convo_history = dspy.InputField(desc="The conversation history")
    response = dspy.InputField(desc="The response to be judged.")
    strategies = dspy.InputField(desc="The empathetic strategies to adhere to.")
    output = dspy.OutputField(desc="Whether the response adheres to one or more of the strategies. Respond with yes or no.")

def format_convo(conv_hist):
    conv = ""
    for c in conv_hist:
        if c["role"] in ["user", "assistant"]:
            new_c = c["role"] + ": " + c["content"] + "\n"
            new_c = new_c[0].upper() + new_c[1:]
            conv = conv + new_c
    return conv


if __name__ == "__main__":
    openai_lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=openai_lm)

    all_convos = json.load(open("empathy_conversation.json"))

    all_scores = []

    adhere_judge = dspy.Predict(DecideAdherence)

    for p in all_convos:
        curr_convo = all_convos[p]
        for i, c in enumerate(curr_convo):
            if c["role"] == "assistant" and "all_empathetic_strategies" in c:
                conv = format_convo(curr_convo[:i])
                judge = adhere_judge(convo_history=conv,
                                     response=c["content"],
                                     strategies=c["all_empathetic_strategies"]).output
                # print(judge, c)
                judge = int(judge.lower().startswith("yes"))
                all_scores.append(judge)
    print(sum(all_scores) / len(all_scores))

    # 0.9272727272727272
    # Adherence to strategies



    