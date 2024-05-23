# TODO: create class with different options for different models and their API calls

from openai import OpenAI
client = OpenAI()

FEW_SHOT_SIZE = 5

def call_few_shot_api(input, model, system_prompt, examples, dataset):
    system_prompt += "See some example below: \n"
    
    shot_size = FEW_SHOT_SIZE
    i = 0
    #for example in examples.select(range(shot_size)):
    for example in examples:
        if i >= shot_size:
            break
        i += 1
        if (dataset == "math"):
            problem = example['problem']
            solution = example['solution']
            system_prompt += f"Q: {problem}\nA: {solution}\n"
        elif (dataset == "natural_questions"):
            question = example['question']['text']
            answers = example['annotations']['short_answers'] # only use first answer given
            if (len(answers[0]['text']) == 0):
                shot_size += 1
                continue
            answer = answers[0]['text']
            system_prompt += f"Q: {question}\nA: {answer}\n"


    return call_default_api(input, model, system_prompt)

def call_COT_api(input, model, system_prompt):
    system_prompt += "Let's think step by step. Start broad, and step by step get more specific until you reach the answer."
    return call_default_api(input, model, system_prompt)

def call_default_api(input, model, system_prompt):
    if (model == "gpt-3.5-turbo"):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": system_prompt},
                {"role": "user", "content": input}
            ]
        )
    return completion
