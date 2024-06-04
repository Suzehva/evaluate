from typing import Optional
from preprocess import math_pre
#import main
import config

def get_vanilla_prompt():
    prompt = "You answer questions about math problems. Enclose your final answer in LaTeX's \\boxed tag.\n"
    return prompt

def get_fs_prompt(dataset):
    prompt = "You answer questions about math problems. Enclose your final answer in LaTeX's \\boxed tag.\n"
    prompt += get_examples("FS", dataset, config.FEWSHOT_SIZE)
    return prompt

def get_cot_prompt():
    prompt = "You answer questions about math problems. Provide a step-by-step solution. Enclose your final answer in LaTeX's \\boxed tag.\n"
    return prompt

def get_cot_fs_prompt(dataset):
    prompt = "You answer questions about math problems. Provide a step-by-step solution. Enclose your final answer in LaTeX's \\boxed tag.\n"
    prompt += get_examples("COT", dataset, config.FEWSHOT_SIZE)
    return prompt


def get_tot_prompt():
    prompt = f"Imagine {config.NUM_EXPERTS} different experts are answering a question about a math problem. All experts will write down 1 step of their thinking,then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. Once the experts have arrived at a final answer, enclose the final answer in LaTeX's \\boxed tag.\n"
    return prompt

def get_examples(prompt_type, dataset, fewshot_size):
    prompt = ""
    for example in dataset.select(range(fewshot_size)):
        problem = example['problem']
        solution = example['solution']
        if prompt_type == "FS": # strip chain-of-thought reasoning in sample
            solution = math_pre.get_answer(solution)
        prompt += f"Q: {problem}\nA: {solution}\n"
    return prompt

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
    
