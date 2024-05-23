from typing import Optional
from preprocess import math_pre
import math_main

def get_fs_prompt(dataset):
    prompt = "You answer questions about math problems. Enclose your final answer in LaTeX's \\boxed tag.\n"
    prompt += get_examples("fs", dataset)
    return prompt

def get_cot_prompt(dataset):
    prompt = "You answer questions about math problems. Provide a step-by-step solution. Enclose your final answer in LaTeX's \\boxed tag.\n"
    prompt += get_examples("cot", dataset)
    return prompt

def get_tot_prompt():
    prompt = f"Imagine {math_main.NUM_EXPERTS} different experts are answering a question about a math problem. All experts will write down 1 step of their thinking,
    then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. 
    Once the experts have arrived at a final answer, enclose the final answer in LaTeX's \\boxed tag.\n"

def get_examples(prompt_type, dataset):
    prompt = ""
    for example in dataset.select(range(math_main.FEWSHOT_SIZE)):
        problem = example['problem']
        solution = example['solution']
        if prompt_type == "fs": # strip chain-of-thought reasoning in sample
            solution = math_pre.get_answer(solution)
        prompt += f"Q: {problem}\nA: {solution}\n"
    return prompt

def get_prompt(prompt_type, dataset):
    """
    Take in prompt type and optional number of examples
    """
    if prompt_type == "cot":
        prompt = get_cot_prompt(dataset)
    elif prompt_type == "fs":
        prompt = get_fs_prompt(dataset)
    elif prompt_type == "tot":
        prompt = get_tot_prompt(dataset)
    else:
        print("Invalid prompt type")
    return prompt
    