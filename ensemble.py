"""
Supports ensembling
"""
from preprocess import math_pre

def take_majority_vote(responses):
    responses_cleaned = [math_pre.get_answer(response) for response in responses]
    majority_output = max(set(responses_cleaned), key=responses_cleaned.count)
    return majority_output