from typing import Optional
from openai import OpenAI
from datasets import load_dataset
from time import time
from sklearn.metrics import f1_score
import pandas as pd
from collections import Counter
from datetime import datetime
import sys
import argparse
# from preprocess/math FIX

DATASET_NAME = "hendrycks/competition_math"
SAMPLE_SIZE = 100
FEWSHOT_SIZE = 5

def get_fs_prompt(num_examples, dataset):
    prompt = "You answer questions about math problems. Enclose your final answer in LaTeX's \\boxed tag.\n"

def get_cot_prompt(num_examples, dataset):
    prompt = "You answer questions about math problems. Provide a step-by-step solution. Enclose your final answer in LaTeX's \\boxed tag.\n"
    examples = get_examples("cot", num_examples, dataset)

def get_tot_prompt(num_agents):
    prompt = f"Imagine {num_agents} different experts are answering a question about a math problem. All experts will write down 1 step of their thinking,
    then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. 
    Once the experts have arrived at a final answer, enclose the final answer in LaTeX's \\boxed tag.\n"


def get_examples(prompt_type, num_examples, dataset):
    prompt = ""
    for example in dataset.select(range(num_examples)):
        problem = example['problem']
        solution = example['solution']
        if prompt_type == "fs":
            solution = get_answer(solution)
        prompt += f"Q: {problem}\nA: {solution}\n"

def get_prompt(prompt_type, arg: 0, dataset: None):
    """
    Take in prompt type and optional number of examples
    """

    if prompt_type == "cot":
        prompt = get_cot_prompt(arg)
    elif prompt_type == "fs":
        prompt = get_fs_prompt(arg)
    elif prompt_type == "tot":
        prompt = get_tot_prompt(arg)
    else:
        print("Invalid prompt type")