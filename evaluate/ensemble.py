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
