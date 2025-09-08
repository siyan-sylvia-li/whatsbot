import openai
import json
import os
import dotenv
import google.generativeai as genai
from scipy.stats import spearmanr

dotenv.load_dotenv(".env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# for model in genai.list_models():
#     print(f"Model Name: {model.name}")
#     print(f"  Description: {model.description}")
#     print(f"  Supported Generation Methods: {model.supported_generation_methods}")
#     print("-" * 30)

client = openai.OpenAI()

def load_prompt_template(model_name): 
  path=None
  if model_name == "gemini": 
    path="empathy-evaluation-prompt-gemini.txt"
  else: # gemini
    path="empathy-evaluation-prompt.txt"
  
  with open(path, "r", encoding="utf-8") as f: 
      return f.read()

# static items required for conversation evaluation
MODEL = "gpt-4.1"
INPUT_ROOT = os.path.join(os.getcwd(), "bot_chats")
OUTPUT_ROOT = os.path.join(os.getcwd(), "evaluated_bot_chats")
# change this to "gpt" or "gemini" to load LLM specific prompt
PROMPT_TEMPLATE = load_prompt_template("gemini")


def evaluate_exp_id_messages():
  input_path = os.path.join(os.path.dirname(__file__), "exp_id_messages_tbe.json")
  output_path = os.path.join(os.path.dirname(__file__), "exp_id_messages_tbe_evaluated.json")
  prompt_template = load_prompt_template("gemini")

  if not os.path.isfile(input_path):
    print(f"{input_path} is missing, skipping...")
    return

  with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  result = {}
  for user_name, conversations in data.items():
    if user_name in ["SMZG", "NAWV"]:
      continue
    result[user_name] = []
    for convo_obj in conversations:
      if not isinstance(convo_obj, dict) or "conversation" not in convo_obj:
        continue
      required_keys = [
        "I intend to follow, or continue to follow, the chatbot’s recommendation over the next 2 days",
        "If I do intend to follow the chatbot's recommendation, I am confident that I can follow the recommendation over the next 2 days",
        "I think the chatbot’s recommendation is useful for enhancing my physical activity"
      ]
      if not all(key in convo_obj for key in required_keys):
        continue
      convo_text = convo_obj["conversation"]
      prompt = prompt_template.format(conversation=convo_text)
      try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        generation_config = {
          "temperature": 0.0,
          "max_output_tokens": 5
        }
        response = model.generate_content(prompt, generation_config=generation_config)
        score_str = response.text.strip().split()[0] if response.text.strip() else None
        empathy_score = int(score_str) if score_str and score_str.isdigit() else None
        if empathy_score is not None:
          empathy_score = max(0, min(empathy_score, 5))
      except Exception as e:
        print(f"Error evaluating {user_name}: {e}")
        empathy_score = None

      # Copy all keys except 'original', add empathy score only
      new_obj = {}
      for k, v in convo_obj.items():
        if k != "original":
          new_obj[k] = v
      new_obj["empathy_score"] = empathy_score
      result[user_name].append(new_obj)

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
  print(f"Saved evaluated exp_id messages to {output_path}")

def format_conversation(convo_list): 
  lines = []
  for item in convo_list: 
    line = f"{item['role']}: {item['content']}"
    lines.append(line)
  
  return "\n".join(lines)

def get_empathy_score_gpt(convo_text): 
  try: 
    response = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "You are an empathy evaluator."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(conversation=convo_text)}
      ],
      temperature=0
    )
    score_str = response.choices[0].message.content.strip()
    score = int(score_str)
    return max(0, min(score, 5))  # Clamp to [0, 5]
  except Exception as e: 
    print(e)
    return None

def get_empathy_score_gemini(convo_text): 
  try: 
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    generation_config = {
      "temperature": 0.0,
      "max_output_tokens": 5
    }
    response = model.generate_content(PROMPT_TEMPLATE.format(conversation=convo_text), generation_config=generation_config)
    score_str = response.text.strip()
    score = int(score_str)
    return max(0, min(score, 5))
  except Exception as e: 
    print(e)
    return None

def process_round(round_path, round_name): 
  input_file = os.path.join(round_path, "message_logs.json")
  if not os.path.isfile(input_file): 
    print(f"{input_file} is missing, skipping...")
  
  with open(input_file, "r", encoding="utf-8") as f: 
    data = json.load(f)
  
  for phone, sessions, in data.items(): 
    # list takes a snapshot of the keys and doesn't affect the loop if more keys are added later
    for session_key, convo in list(sessions.items()):
      convo = sessions[session_key] 
      if isinstance(convo, list): 
        convo_text = format_conversation(convo)
        print(f"EVALUATING {round_name} | {phone} | {session_key}")
        # replace this with gpt or gemini depending on the evaluation LLM
        score = get_empathy_score_gemini(convo_text)
        if score is not None: 
          sessions[f"{session_key}_empathy_score"] = score
  
  data["round"] = round_name
  
  output_dir = os.path.join(OUTPUT_ROOT, round_name)
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, "message_logs.json")
  
  with open(output_path, "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=2, ensure_ascii=False)
  
  print(f"Save evaluated file to output_path {output_path}")

def compute_spearman_correlation(json_path):
    if not os.path.isfile(json_path):
        print(f"{json_path} is missing, skipping...")
        return {}
    
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    for user, convos in data.items():
        empathy_scores = []
        intent_scores = []
        selfefficacy_scores = []
        usefulness_scores = []

        for convo in convos:
            try:
                empathy_scores.append(convo["empathy_score"])
                intent_scores.append(convo["I intend to follow, or continue to follow, the chatbot’s recommendation over the next 2 days"])
                selfefficacy_scores.append(convo["If I do intend to follow the chatbot's recommendation, I am confident that I can follow the recommendation over the next 2 days"])
                usefulness_scores.append(convo["I think the chatbot’s recommendation is useful for enhancing my physical activity"])
            except KeyError:
                continue

        # Only calculate if there are at least 2 data points
        # it is guaranteed that len(empathy_scores) == len(intent_scores) == len(selfefficacy_scores) == len(usefulness_scores) since we checked required keys in the previous function
        if len(empathy_scores) > 1:
            results[user] = {
                "spearman_empathy_intent": spearmanr(empathy_scores, intent_scores).correlation,
                "spearman_empathy_selfefficacy": spearmanr(empathy_scores, selfefficacy_scores).correlation,
                "spearman_empathy_usefulness": spearmanr(empathy_scores, usefulness_scores).correlation,
                "n": len(empathy_scores)
            }
        else:
            results[user] = "Not enough data"

    # Save results to a new file in the same directory as the input json
    output_path = os.path.join(os.path.dirname(json_path), "spearman_correlation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Spearman correlation results saved to {output_path}")
    return results

# Example usage:
# results = spearman_empathy_intent_selfefficacy("exp_id_messages_tbe_evaluated.json")
# print(results)

def main(): 
  if not os.path.exists(INPUT_ROOT): 
    print(f"INPUT folder {INPUT_ROOT} not found")
    return
  
  for entry in os.listdir(INPUT_ROOT): 
    round_path = os.path.join(INPUT_ROOT, entry)
    if os.path.isdir(round_path) and entry.startswith("round_"): 
      process_round(round_path, entry)

if __name__ == "__main__": 
  # main()
  # evaluate_exp_id_messages()
  # NOTE-os.path.join(os.path.dirname(__file__), "exp_id_messages_tbe_evaluated.json") joins path of the "exp_id_messages_tbe_evaluated.json" file in the same directory as this script
  compute_spearman_correlation(os.path.join(os.path.dirname(__file__), "exp_id_messages_tbe_evaluated.json"))
