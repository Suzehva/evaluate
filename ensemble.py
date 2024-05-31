import evaluation
from preprocess import natural_questions_pre


"""
Supports ensembling
"""
from preprocess import math_pre

def take_majority_vote_math(responses):
    responses_cleaned = [math_pre.get_answer(response) for response in responses]
    for response_c in responses_cleaned:
        print(response_c)
    majority_output = max(set(responses_cleaned), key=responses_cleaned.count)
    print("majority output: " + majority_output)
    return majority_output

def take_highest_f1_natural_questions(responses, answers):
    responses_cleaned = [natural_questions_pre.extract_answer(response) for response in responses]
    majority_output = max(set(responses_cleaned), key=responses_cleaned.count)
    #print("majority_output: " + majority_output)
    return majority_output

    """
    #TODO: use f1 score for every possible answer
    majority_output = ""
    f1 = 0
    for response in responses:
        for temp_ans in answers:
            f1_temp = evaluation.f1_score(response, temp_ans['text'][0])
            if f1_temp > f1:
                f1 = f1_temp
                majority_output = response
    return majority_output
    """




