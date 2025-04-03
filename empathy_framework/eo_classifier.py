import dspy
from typing import Literal, List

class EOClassifier(dspy.Signature):
    """
    Classify the user utterance into different Empathy Opportunities (EO) or as General. Use the definitions and examples for each EO provided to you to make your classification. Provide the top three EO predictions.
    """

    user_input: str = dspy.InputField(desc="The user's utterance.")
    eo_descriptions: str = dspy.InputField(desc="Definitions and examples for each Empathy Opportunity type.")
    empathy_opportunity: List[Literal[
        "negative_feelings_explicit",
        "negative_feelings_implicit",
        "negative_judgment_explicit",
        "negative_judgment_implicit",
        "positive_self_judgment_explicit",
        "positive_self_judgment_implicit",
        "negative_appreciation_explicit",
        "negative_appreciation_implicit",
        "general"
    ]] = dspy.OutputField(desc="The top three Empathy Opportunity predictions")


class EOClassifierOne(dspy.Signature):
    """
    Classify a segment of a longer user utterance into different Empathy Opportunities (EO) or as General. Use the definitions and examples for each EO provided to you to make your classifications. Return the top three EO classifications.
    """

    user_segment: str = dspy.InputField(desc="The segment of the user's utterance.")
    user_utterance: str = dspy.InputField(desc="The full original user utterance.")
    eo_descriptions: str = dspy.InputField(desc="Definitions and examples for each Empathy Opportunity type.")
    empathy_opportunity: List[Literal[
        "negative_feelings_explicit",
        "negative_feelings_implicit",
        "negative_judgment_explicit",
        "negative_judgment_implicit",
        "positive_self_judgment_explicit",
        "positive_self_judgment_implicit",
        "negative_appreciation_explicit",
        "negative_appreciation_implicit",
        "general"
    ]] = dspy.OutputField(desc="The top three Empathy Opportunity prediction")


class EOClassifierModule(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.eo_class = dspy.ChainOfThought(EOClassifier)
        # self.eo_class.__signature__ = open("eo_prompt.txt").read()
    
    def forward(self, user_input, eo_descriptions):
        output = self.eo_class(user_input = user_input, eo_descriptions=eo_descriptions)
        return dspy.Prediction(eo_classification=output.empathy_opportunity)

class SingleEOClassifierModule(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.eo_class = dspy.ChainOfThought(EOClassifierOne)
        # self.eo_class.__signature__ = open("eo_prompt.txt").read()
    
    def forward(self, user_input, user_utt, eo_descriptions):
        output = self.eo_class(user_segment = user_input, user_utterance=user_utt, eo_descriptions=eo_descriptions)
        return dspy.Prediction(eo_classification=output.empathy_opportunity)




        
