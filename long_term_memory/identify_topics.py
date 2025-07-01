import dspy
from typing import List
from .embed_bullets import retrieve_bullets

summary_lm = dspy.LM(model="openai/gpt-4o-mini")
# ----------------------
# Topic Identification Function
# ----------------------
def identify_topic(
    last_three: List[str]
) -> str:
    """
    Given the last three session summaries, ask the LLM to identify
    the most important topic to discuss next.
    Returns the topic as a string.
    """
    system_prompt = "You are an insightful coaching assistant."
    session_str = "\n\n".join(last_three)
    user_prompt = (
        "Here are the summaries of the most recent sessions you have had with a user:\n"
        f"{session_str}"
        "Based on these, identify the single most important topic you should discuss next. Be succinct."
    )
    response = summary_lm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )[0]
    return response

def identify_and_retrieve(last_three: List[str], user_prefix: str):
    query = identify_topic(last_three)
    res = retrieve_bullets(user_prefix=user_prefix, query=query)
    return query, res
