from datasets import load_dataset
from time import time
import apis
from prompting import math_prompts
from postprocessing import math_post
import ensemble
import config

def load_data():
    """
    loads dataset from huggingface
    """
    train_dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    test_dataset  = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    return train_dataset, test_dataset

def run_model(train, test):
    """
    main function used to generate LLM responses. Loops through dataset to generate responses
    """
    results = []
    total_tokens = 0
    total_time = 0
    total_correct = 0
    counter = 0

    prompt = math_prompts.get_prompt(config.PROMPTING_T, test)
    if config.PROMPTING_T == "COT_FS" or config.PROMPTING_T == "FS": # filter out fs examples from train
        test = test.select(range(config.FEWSHOT_SIZE, len(test)))
    
    for sample in test.select(range(config.SAMPLE_SIZE)):
        problem = sample['problem']

        if config.USE_ENSEMBLE:
            start_time = time()
            responses = []
            num_tokens = 0
            completion_tokens = 0
            prompt_tokens = 0
            for _ in range(config.ENSEMBLE_SIZE):
                response = apis.call_default_api(problem, config.MODEL, prompt, temperature=1)
                output_solution = response.choices[0].message.content
                num_tokens += response.usage.total_tokens
                completion_tokens += response.usage.completion_tokens
                prompt_tokens += response.usage.prompt_tokens
                responses.append(output_solution)
            latency = time() - start_time
            output_solution = ensemble.take_majority_vote_math(responses)
            correct = math_post.check_answer(output_solution, sample['solution'])
        else:
            start_time = time()
            response = apis.call_default_api(problem, config.MODEL, prompt)
            latency = time() - start_time
            output_solution = response.choices[0].message.content
            num_tokens = response.usage.total_tokens
            correct = math_post.check_answer(output_solution, sample['solution'])
            prompt_tokens = response.usage.prompt_tokens;
            completion_tokens = response.usage.completion_tokens

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
                'completion_tokens':completion_tokens,
                'prompt_tokens':prompt_tokens
            }
        )
        if config.USE_ENSEMBLE:
            results[-1]['ensemble_output'] = responses
        total_tokens += num_tokens
        total_time += latency
        total_correct += correct
        counter += 1

    return results, total_tokens, total_time, total_correct, prompt


