from .constants import EO_SAMPLING_MAP
from typing import List
import random

def sample_appraisal(eos: List[str], sampling_num=1):
    total_mapping = {}
    for e in eos:
        eo_dict = EO_SAMPLING_MAP[e]
        for i in range(len(eo_dict["population"])):
            if eo_dict["population"][i] not in total_mapping:
                total_mapping[eo_dict["population"][i]] = eo_dict["weights"][i]
            else:
                total_mapping[eo_dict["population"][i]] += eo_dict["weights"][i]
    all_population = list(total_mapping.keys())
    all_weights = [total_mapping[k] for k in all_population]
    s_app = set(random.choices(population=all_population, weights=all_weights, k=sampling_num))
    return s_app