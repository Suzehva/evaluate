from typing import Optional
from datasets import load_dataset
from time import time
from openai import OpenAI
import apis
from prompting import math_prompts
from preprocess import math_pre
import ensemble

FEWSHOT_SIZE = 5 # always set this to 0 if not using fewshot
NUM_EXPERTS = 3 # tree of thought
ENSEMBLE_SIZE = 4 # ensembling

def load_data():
    train_dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    test_dataset  = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)

    print("Loaded dataset")
    return train_dataset, test_dataset

def run_model(train, test, model, sample_size, prompting_t, use_ensemble = False):
    results = []
    total_tokens = 0
    total_time = 0
    total_correct = 0
    counter = 0

    prompt = math_prompts.get_prompt(prompting_t, train)
    if prompting_t == "COT_FS" or prompting_t == "FS": # filter out fs examples from train
        train = train.select(range(FEWSHOT_SIZE, len(train)))
    
    for sample in train.select(range(sample_size)):
        problem = sample['problem']

        if use_ensemble:
            start_time = time()
            responses = []
            num_tokens = 0
            for i in range(ENSEMBLE_SIZE):
                response = apis.call_default_api(problem, model, prompt)
                output_solution = response.choices[0].message.content
                num_tokens += response.usage.total_tokens
                responses.append(output_solution)
            latency = time() - start_time
            output_solution = ensemble.take_majority_vote(responses)
            correct = math_pre.check_answer(output_solution, sample['solution'])
        else:
            start_time = time() # should I do this call and latency call later within if statements?
            response = apis.call_default_api(problem, model, prompt)
            latency = time() - start_time
            output_solution = response.choices[0].message.content
            num_tokens = response.usage.total_tokens
            correct = math_pre.check_answer(output_solution, sample['solution'])

        results.append(
            {
                'problem': problem,
                'level': sample['level'],
                'type': sample['type'],
                'solution': sample['solution'],
                'model_output': output_solution,
                'correct':correct,
                'latency': latency,
                'num_tokens': num_tokens,
                'completion_tokens':response.usage.completion_tokens,
                'prompt_tokens':response.usage.prompt_tokens
            }
        )
        total_tokens += num_tokens
        total_time += latency
        total_correct += correct
        counter += 1
        print(f"processed {counter}/{sample_size}")

    return results, total_tokens, total_time, total_correct, prompt


