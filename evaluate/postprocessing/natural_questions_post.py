""""
Post-processes responses returned by LLM for the natural questions dataset
"""

def extract_answer(answer):
    end_index = answer.rfind("]]")
    if end_index == -1:
        return answer #unable to find box; use full answer as alternative
    start_index = answer.rfind("[[", 0, end_index)
    if start_index == -1:
        return answer
    return answer[start_index + 2:end_index]

