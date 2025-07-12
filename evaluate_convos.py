import openai
import json
import os
import dotenv
import google.generativeai as genai

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

def main(): 
  if not os.path.exists(INPUT_ROOT): 
    print(f"INPUT folder {INPUT_ROOT} not found")
    return
  
  for entry in os.listdir(INPUT_ROOT): 
    round_path = os.path.join(INPUT_ROOT, entry)
    if os.path.isdir(round_path) and entry.startswith("round_"): 
      process_round(round_path, entry)

if __name__ == "__main__": 
  main()
  