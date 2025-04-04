import dspy
from .eo_classifier import EOClassifierModule
from .sampling_appraisal import sample_appraisal
from .constants import CLINICAL_EMPATHY_DESCRIPTIONS, __location__
import os
from typing import List

empathy_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0)

class EmpatheticResponse(dspy.Signature):
    """You are an empathetic physical activity counselor talking with a user about increasing their levels of physical activity. When constructing an empathetic response to the user, you should reference the descriptions and definitions of specific empathetic response strategies given to you. Make your response short and like a text message."""
    user_input = dspy.InputField()
    conversation_history = dspy.InputField(desc="The past 10 turns of conversation, not including the current user utterance")
    empathy_techniques = dspy.InputField(desc="The specific empathetic response strategies that you would use to respond to the user")
    empathetic_response = dspy.OutputField()

class EmpatheticResponderDSPy(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.response_gen = dspy.Predict(EmpatheticResponse)
        self.eo_class = EOClassifierModule()
        self.eo_class.load(os.path.join(__location__, "eo_classifier_optimized_simba.json"))
        self.eo_descriptions = open(os.path.join(__location__, "eo_descriptions.txt")).read()

    def forward(self, user_input, convo_history):
        # Given user input, we will first break it down into individual statements
        # We will then perform classification on each individual statement
        # segmentation = self.sentence_segmenter(input_paragraph=user_input).output
        # segmentation = segmentation.split("\n")
        all_eos = []
        # for s in segmentation:
        #     # with dspy.context(lm=openai_4o):
        eos = self.eo_class(user_input=user_input, eo_descriptions=self.eo_descriptions).eo_classification
        all_eos.extend(eos)
        all_appraisals = sample_appraisal(all_eos, sampling_num=3)
        all_emp_techs = ""
        for a in all_appraisals:
            all_emp_techs = all_emp_techs + CLINICAL_EMPATHY_DESCRIPTIONS[a] + "\n\n"
        return self.response_gen(user_input=user_input, conversation_history=convo_history,
                                 empathy_techniques=all_emp_techs)
    
class EmpatheticResponder:
    def __init__(self):
        self.eo_class = EOClassifierModule()
        self.eo_class.load(os.path.join(__location__, "eo_classifier_optimized_simba.json"))
        
    def respond_empathetically(self, user_input, convo_history: List[dict]):
        all_eos = []
        eos = self.eo_class(user_input=user_input, eo_descriptions=self.eo_descriptions).eo_classification
        all_eos.extend(eos)
        all_appraisals = sample_appraisal(all_eos, sampling_num=3)
        all_emp_techs = ""
        for a in all_appraisals:
            all_emp_techs = all_emp_techs + CLINICAL_EMPATHY_DESCRIPTIONS[a] + "\n\n"
        convo_history.append({"role": "user", "content": user_input})
        convo_history.append({"role": "system", "content": "Respond to the user's message in an empathetic manner. Utilize the following empathetic response strategies when responding:\n\n" + all_emp_techs})
        response_text = empathy_lm(messages=convo_history)
        return response_text
        


