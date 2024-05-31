from typing import Optional
from preprocess import math_pre
import config

def get_vanilla_prompt():
    prompt = "You answer questions and give short, concise answers. Give your shorest answer possible. Put your answer between double square brackets like this: [[ANSWER]].\n"
    return prompt

def get_fs_prompt(dataset):
    prompt = "You answer questions and give short, concise answers. Give your shortest answer possible. Put your answer between double square brackets like this: [[ANSWER]].\n"
    prompt += get_examples("FS", dataset, config.FEWSHOT_SIZE)
    return prompt

def get_cot_prompt():
    prompt = "You answer questions. Provide a step-by-step solution. Give your shortest final answer possible. Put your final answer between double square brackets like this: [[ANSWER]].\n"
    return prompt

def get_cot_fs_prompt(dataset):
    prompt = "You answer questions. Provide a step-by-step solution. Give your shortest final answer possible. Put your final answer between double square brackets like this: [[ANSWER]].\n"
    prompt += get_examples("COT", dataset, config.FEWSHOT_SIZE)
    return prompt


def get_tot_prompt():
    prompt = f"Imagine {main.NUM_EXPERTS} different experts are answering a question. All experts will write down 1 step of their thinking,then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. Once the experts have arrived at a final answer, put the answer between double square brackets like this: [[ANSWER]]. Give your shortest final answer possible\n"
    return prompt

def get_examples(prompt_type, dataset, fewshot_size):
    shot_size = fewshot_size
    system_prompt = ""
    i = 0
    for example in dataset:
        if i >= shot_size:
            break
        i += 1
        if (dataset == "math"):
            problem = example['problem']
            solution = example['solution']
            if prompt_type == "FS": # strip chain-of-thought reasoning in sample
                solution = math_pre.get_answer(solution)
            system_prompt += f"Q: {problem}\nA: {solution}\n"
        elif (dataset == "natural_questions"):
            question = example['question']['text']
            answers = example['annotations']['short_answers'] # only use first answer given
            if (len(answers[0]['text']) == 0):
                shot_size += 1
                continue
            answer = answers[0]['text']
            system_prompt += f"Q: {question}\nA: {answer}\n"

    return system_prompt


def get_prompt(prompt_type, dataset):
    """
    Take in prompt type and optional number of examples
    """
    if prompt_type == "COT":
        prompt = get_cot_prompt()
    elif prompt_type == "COT_FS":
        prompt = get_cot_fs_prompt(dataset)
    elif prompt_type == "FS":
        prompt = get_fs_prompt(dataset)
    elif prompt_type == "TOT":
        prompt = get_tot_prompt()
    elif prompt_type == "":
        prompt = get_vanilla_prompt()
    else:
        print("Invalid prompt type") # TODO: give error
        prompt = ""
    return prompt

