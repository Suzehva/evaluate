import evaluation
from postprocessing import natural_questions_post
from postprocessing import math_post

def take_majority_vote_math(responses):
    responses_cleaned = [math_post.get_answer(response) for response in responses]
    majority_output = max(set(responses_cleaned), key=responses_cleaned.count)
    
    # we have to return the uncleaned version of our response since we will be cleaning later on
    majority_index = responses_cleaned.index(majority_output) 
    return responses[majority_index]

def take_highest_f1_natural_questions(responses, answers):
    responses_cleaned = [natural_questions_post.extract_answer(response) for response in responses]
    majority_output = max(set(responses_cleaned), key=responses_cleaned.count)
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




