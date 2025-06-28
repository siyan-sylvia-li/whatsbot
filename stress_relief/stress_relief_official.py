import json
import os
import dspy
from typing import List
import re
from .mab import UCBBandit
from .intervention_database import InterventionDatabase
import random

class StressReliefModule:
    def __init__(self,
                 user_profile_path: str):
        self.profile_path = user_profile_path
        if not os.path.exists(user_profile_path):
            json.dump({}, open(user_profile_path, "w+"))
        self.user_profile = json.load(open(user_profile_path))
        self.lm = dspy.LM(model="openai/gpt-4o")
        self.int_db = InterventionDatabase()
        self.epsilon = 0.3
    

    def update_user_profile(self):
        json.dump(self.user_profile, open(self.profile_path, "w+"))
    
    def generate_from_intervention(self, intervention, msg_history, non_empathetic=False):
        # Safely get intervention details
        intervention_name = intervention.get('name', 'stress relief technique')
        intervention_description = intervention.get('description', 'a helpful stress relief technique')
        intervention_link = intervention.get('link', '')
        
        if non_empathetic:
            prompt = f"""You will now recommend a specific stress relief practice for the user.
            
            Recommend this specific practice: {intervention_name} - {intervention_description}
            
            End your message with a link to the practice: {intervention_link}
            
            Keep your response short (2-3 sentences). Be professional, do not be empathetic, avoid lengthy explanations.
            End by asking: "On a scale of 1 to 10, how helpful was this suggestion? Please also share how it made you feel or any other thoughts you have about it in the same message."
            """
        else:
            prompt = f"""You will now recommend a specific stress relief practice for the user.
            
            Recommend this specific practice: {intervention_name} - {intervention_description}
            
            End your message with a link to the practice: {intervention_link}
            
            Keep your response short (2-3 sentences). Be empathetic but avoid lengthy explanations.
            End by asking: "On a scale of 1 to 10, how helpful was this suggestion? Please also share how it made you feel or any other thoughts you have about it inm the same message."
            """
        response = self.lm(messages=msg_history + [
                {
                    "role": "system",
                    "content": prompt
                }
            ])[0]
        return response
        
    
    def process_user_msg(self, user_rating: str, user_id: str, msg_history: List[dict], non_empathetic=False):
        # Regex to match digit 0-5 or word zero-one-two-three-four-five (case-insensitive)
        match = re.search(r'\b([0-5]|zero|one|two|three|four|five)\b', user_rating, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            word_to_num = {
                'zero': 0,
                'one': 1,
                'two': 2,
                'three': 3,
                'four': 4,
                'five': 5
            }
            if value.isdigit():
                rating = int(value)
            else:
                rating = word_to_num[value]
        if rating < 2:
            # This is too high of a stress rating
            response = self.lm(messages=msg_history + [
                {
                    "role": "system",
                    "content": "The user is not feeling extremely stressed. Indicate that you are glad to hear that the user's stress level seems generally manageable, and then say goodbye to the user."
                }
            ])[0]
            return False, response
        if user_id not in self.user_profile:
            self.user_profile.update({
                user_id: {
                    "mab_states": None,
                    "interaction_history": {
                        "interventions": [],
                        "scores": [],
                        "arms": []
                    }
                }
            })
        curr_mab = UCBBandit(self.user_profile[user_id]["mab_states"])
        curr_arm = curr_mab.select_arm()
        selected_category = self.int_db.intervention_categories[curr_arm]
        all_interventions = self.int_db.interventions[selected_category]
        if random.random() < self.epsilon or len(self.user_profile[user_id]["interaction_history"]["interventions"]) == 0:
            int_choices = [x for x in all_interventions if x not in self.user_profile[user_id]["interaction_history"]["interventions"]]
            chosen = random.choice(int_choices)
        else:
            # Find the index of the maximum score
            chosen_ind = self.user_profile[user_id]["interaction_history"]["scores"].index(
                max(self.user_profile[user_id]["interaction_history"]["scores"])
            )
            chosen = self.user_profile[user_id]["interaction_history"]["interventions"][chosen_ind]
        
        stress_rec = self.generate_from_intervention(chosen, msg_history, non_empathetic)

        self.user_profile[user_id]["mab_states"] = curr_mab.dump_states()
        self.user_profile[user_id]["interaction_history"]["interventions"].append(chosen)
        self.user_profile[user_id]["interaction_history"]["arms"].append(curr_arm)

        json.dump(self.user_profile, open(self.profile_path, "w+"))
        
        return 2, stress_rec


    def process_user_feedback(self, user_feedback_msg: str, user_id: str, msg_history: List[dict], non_empathetic=False):
        match = re.search(r'\b([0-9]|10)\b', user_feedback_msg)
        score = int(match.group(1)) if match else None

        if score is not None:
            curr_mab = UCBBandit(self.user_profile[user_id]["mab_states"])
            self.user_profile[user_id]["interaction_history"]["scores"].append(score)
            curr_mab.update(
                self.user_profile[user_id]["interaction_history"]["arms"][-1],
                self.user_profile[user_id]["interaction_history"]["scores"][-1]
            )
            self.user_profile[user_id]["mab_states"] = curr_mab.dump_states()
            json.dump(self.user_profile, open(self.profile_path, "w+"))

            resp = self.lm(messages=msg_history + [
                {
                    "role": "system",
                    "prompt": "Respond appropriately and say goodbye to the user, telling them that you will be checking in in two days."
                }
            ])[0]
            return 3, resp
        
        resp = "Can you please provide the specific rating of how helpful the practice was? Thank you!"
        return False, resp
        

        

