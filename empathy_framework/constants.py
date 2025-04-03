CLINICAL_EMPATHY_DESCRIPTIONS = {}

import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

for l in open(os.path.join(__location__, "appraisal_description.txt")).readlines():
    spls = l.replace("\n", "").split("||")
    CLINICAL_EMPATHY_DESCRIPTIONS.update({spls[0]: spls[1]})


EO_SAMPLING_MAP = {
    "negative_feelings_explicit": {
        "population": ["neutral_support_explicit_appreciation", 
                       "sharing_feelings_views", 
                       "understand_feelings_views",
                       "elicit_indirect_confirmation",
                       "elicit_indirect_coaching",
                       "elicit_indirect_feeling_invitation",
                       "elicit_indirect_feeling_negative",
                       "elicit_direct_feeling"],
        "weights": [9, 3, 4, 1, 4, 1, 1, 1]
    },
    "negative_feelings_implicit": {
        "population": ["acceptance_explicit_implicit_judgment",
                       "neutral_support_explicit_appreciation",
                       "neutral_support_normalization",
                       "sharing_feelings_views",
                       "understand_feelings_views",
                       "elicit_indirect_coaching",
                       "elicit_indirect_feeling_invitation",
                       "elicit_direct_feeling"
                       ],
        "weights": [11, 8, 1, 2, 9, 14, 3, 1]
    },
    "negative_judgment_explicit": {
        "population": ["acceptance_explicit_implicit_judgment",
                       "neutral_support_explicit_appreciation",
                       "understand_feelings_views",
                       "elicit_indirect_coaching"
                       ],
        "weights": [2, 5, 3, 3]
    },
    "negative_judgment_implicit": {
        "population": ["acceptance_explicit_implicit_judgment",
                       "acceptance_repetition",
                       "neutral_support_explicit_appreciation",
                       "sharing_feelings_views",
                       "understand_feelings_views",
                       "elicit_indirect_coaching",
                       "elicit_indirect_judgment",
                       "elicit_direct_feeling",
                       "elicit_feelings_3rd"],
        "weights": [7, 1, 7, 2, 3, 15, 1, 1, 1]
    },
    "positive_self_judgment_explicit": {
        "population": ["acceptance_explicit_implicit_judgment",
                       "neutral_support_explicit_appreciation",
                       "sharing_feelings_views",
                       "elicit_direct_appreciation",
                       "elicit_indirect_coaching",
                       "elicit_direct_judgment",
                       "elicit_indirect_feeling_invitation"],
        "weights": [2, 1, 1, 1, 2, 1, 1]
    },
    "positive_self_judgment_implicit": {
        "population": ["acceptance_explicit_implicit_judgment",
                       "acceptance_repetition",
                       "neutral_support_explicit_appreciation",
                       "neutral_support_explicit_judgment",
                       "neutral_support_normalization",
                       "sharing_feelings_views",
                       "elicit_indirect_confirmation",
                       "elicit_indirect_coaching",
                       "elicit_direct_feeling"],
        "weights": [49, 1, 5, 1, 1, 1, 3, 7, 4]
    },
    "negative_appreciation_explicit": {
        "population": ["sharing_feelings_views",
                       "elicit_indirect_coaching",
                       "elicit_indirect_feeling_emotive",
                       "elicit_indirect_feeling_invitation"],
        "weights": [3, 9, 1, 1]
    },
    "negative_appreciation_implicit": {
        "population": ["acceptance_explicit_implicit_judgment",
                       "acceptance_repetition",
                       "neutral_support_explicit_appreciation",
                       "sharing_feelings_views",
                       "understand_feelings_views",
                       "elicit_indirect_coaching",
                       "elicit_indirect_feeling_invitation"],
        "weights": [2, 1, 10, 11, 10, 11, 3]
    },
    "general": {
        "population": ["acceptance_repetition",
                       "neutral_support_explicit_appreciation",
                       "neutral_support_explicit_judgment",
                       "sharing_feelings_views",
                       "elicit_indirect_coaching",
                       "elicit_indirect_feeling_invitation"],
        "weights": [7, 1, 1, 3, 1, 2]
    }
}

for k in EO_SAMPLING_MAP:
    sum_weights = sum(EO_SAMPLING_MAP[k]["weights"])
    EO_SAMPLING_MAP[k]["weights"] = [x / sum_weights for x in EO_SAMPLING_MAP[k]["weights"]]