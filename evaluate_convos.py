import csv
import openai
import json
import os
import dotenv
from google import genai
from google.genai import types
from scipy.stats import spearmanr

dotenv.load_dotenv(".env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_client = genai.Client()

client = openai.OpenAI()

def load_prompt_template(prompt_type): 
  prompt_files = {
    "emotion": "empathy-evaluation-emotional-reaction.txt",
    "exploration": "empathy-evaluation-exploration.txt",
    "interpretation": "empathy-evaluation-interpretations.txt"
  }
  # default placeholder prompt if the type is not found is in the second argument
  filename = prompt_files.get(prompt_type, "empathy-evaluation-prompt.txt")
  with open(filename, "r", encoding="utf-8") as f: 
      return f.read()

# static items required for conversation evaluation
MODEL = "gpt-4.1"
INPUT_ROOT = os.path.join(os.getcwd(), "bot_chats")
OUTPUT_ROOT = os.path.join(os.getcwd(), "evaluated_bot_chats")
# change this to "gpt" or "gemini" to load LLM specific prompt
PROMPT_TEMPLATE = load_prompt_template("gemini")

def evaluate_epitome_csv(category, csv_path="sample_epitome.csv", output_json=None, with_description=False):
  """
  For each row in the CSV, constructs a conversation string, inserts it into the prompt,
  calls Gemini for a score, and outputs a JSON file with:
    - actual score (from CSV, for the given category)
    - conversation
    - gemini_rating (from Gemini)
    - match (True/False)
  """
  prompt_template = load_prompt_template(category)
  category_level_col = {
    "emotion": "level_emotional_reactions",
    "exploration": "level_explorations",
    "interpretation": "level_interpretations"
  }[category]

  # Set output filename if not provided, append description flag
  if output_json is None:
    desc_str = "description" if with_description else "no_description"
    output_json = f"epitome_gemini_eval_{category}_{desc_str}.json"

  results = []
  total_convos = 0
  mismatches = 0
  print(f"Starting evaluation for category: {category}, with_description: {with_description}")
  with open(csv_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader):
      if idx >= 10:
        print("Reached evaluation limit of 10 conversations.")
        break
      seeker_post = row["seeker_post"].strip()
      response_post = row["response_post"].strip()
      convo = f"User: {seeker_post}\nCounselor: {response_post}\n"
      print(f"\nEvaluating conversation {idx+1}:")
      print(convo)
      prompt = prompt_template.format(conversation=convo)
      try:
        response = gemini_client.models.generate_content(
          model="gemini-2.5-pro",
          contents=prompt,
          config=types.GenerateContentConfig(
            temperature=0.0
          )
        )
        score_str = response.text.strip().split()[0] if response.text and response.text.strip() else None
        gemini_rating = int(score_str) if score_str and score_str.isdigit() else None
        print(f"Gemini rating: {gemini_rating}")
      except Exception as e:
        print(f"Error evaluating: {e}")
        gemini_rating = None

      try:
        actual_score = int(row[category_level_col])
      except Exception:
        actual_score = None
      print(f"Actual score: {actual_score}")

      match = (gemini_rating == actual_score) if (gemini_rating is not None and actual_score is not None) else False
      print(f"Match: {match}")
      total_convos += 1
      if not match:
        mismatches += 1

      results.append({
        "conversation": convo,
        "actual_score": actual_score,
        "gemini_rating": gemini_rating,
        "match": match
      })

  # Add summary at the end of the JSON
  match_probability = ((total_convos - mismatches) / total_convos) if total_convos > 0 else 0.0
  output = {
    "results": results,
    "total": total_convos,
    "match_probability": match_probability
  }
  with open(output_json, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
  print(f"Saved Gemini evaluation results to {output_json}")
  print(f"Total conversations evaluated: {total_convos}")
  print(f"Number of mismatches: {mismatches}")
  print(f"Match probability: {match_probability:.3f}")

def evaluate_exp_id_messages(category):
  """
  Evaluate conversations using the specified category (emotion, exploration, interpretation).
  Loads the corresponding prompt and outputs to a category-specific JSON file.
  """
  input_path = os.path.join(os.path.dirname(__file__), "exp_id_messages_tbe.json")
  output_filename = f"exp_id_messages_tbe_evaluated_{category}.json"
  output_path = os.path.join(os.path.dirname(__file__), output_filename)
  prompt_template = load_prompt_template(category)

  if not os.path.isfile(input_path):
    print(f"{input_path} is missing, skipping...")
    return

  with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  result = {}
  user_processed = 0
  for user_name, conversations in data.items():
    if user_name in ["SMZG", "NAWV"]:
      continue
    if user_processed >= 4:
      break
    print(f"Processing user: {user_name}")
    result[user_name] = []
    convo_count = 0
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
      if convo_count >= 4:
        break
      convo_text = convo_obj["conversation"]
      prompt = prompt_template.format(conversation=convo_text)
      try:
        response = gemini_client.models.generate_content(
          model="gemini-2.5-pro",
          contents=prompt,
          config=types.GenerateContentConfig(
            temperature=0.0
          )
        )
        score_str = response.text.strip().split()[0] if response.text and response.text.strip() else None
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
      convo_count += 1
    user_processed += 1

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

def compute_spearman_correlation():
  """
  For each category (emotion, interpretation, exploration), compute Spearman correlation
  for users with more than 10 scores, and output a JSON file for each category.
  """
  categories = ["emotion", "interpretation", "exploration"]
  for category in categories:
    input_filename = f"exp_id_messages_tbe_evaluated_{category}_no_description.json"
    input_path = os.path.join(os.path.dirname(__file__), input_filename)
    if not os.path.isfile(input_path):
      print(f"{input_path} is missing, skipping...")
      continue
    with open(input_path, "r") as f:
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

      # Only calculate if there are more than 10 data points
      if len(empathy_scores) > 10:
        intent_corr = spearmanr(empathy_scores, intent_scores).correlation
        selfeff_corr = spearmanr(empathy_scores, selfefficacy_scores).correlation
        useful_corr = spearmanr(empathy_scores, usefulness_scores).correlation
        # Replace NaN with 'N/A'
        intent_corr = intent_corr if intent_corr == intent_corr else "N/A"
        selfeff_corr = selfeff_corr if selfeff_corr == selfeff_corr else "N/A"
        useful_corr = useful_corr if useful_corr == useful_corr else "N/A"
        results[user] = {
          "spearman_empathy_intent": intent_corr,
          "spearman_empathy_selfefficacy": selfeff_corr,
          "spearman_empathy_usefulness": useful_corr,
          "n": len(empathy_scores)
        }
      else:
        results[user] = "Not enough data"

    # Save results to a new file in the same directory as the input json
    output_filename = f"spearman_correlation_results_{category}.json"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Spearman correlation results saved to {output_path}")

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
  # evaluate_exp_id_messages("emotion")
  # print("--emotion done--")
  # evaluate_exp_id_messages("interpretation")
  # print("--interpretation done--")
  # evaluate_exp_id_messages("exploration")
  # print("--exploration done--")
  # compute_spearman_correlation()
  evaluate_epitome_csv("exploration", csv_path="sample_epitome.csv", output_json="epitome_gemini_eval_exploration_no_description.json", with_description=False)