# TODO: create class to preprocess math data (select only last data between {}, latex format
# Return a pandas df with the following columns: question, original answer, original processed answer, level, genre

from datasets import load_dataset
from time import time
from openai import OpenAI
import apis

SYSTEM_INPUT = "You answer questions about math problems. Provide a concise answer. Enclose your final answer in LaTeX's \\boxed tag.\n"

def load_data():
    train_dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    test_dataset  = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    print("Loaded dataset")
    return train_dataset, test_dataset


def remove_boxed(string):
    """Source: https://github.com/hendrycks/math
    Extract the text within a \\boxed{...} environment.

    Example:
    >>> remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string):
    """Source: https://github.com/hendrycks/math
    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval


def get_answer(solution):
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    #if answer is None:
        #return None
    return answer


def check_answer(output, ground_truth):
    output_ans = get_answer(output)
    ground_truth_ans = get_answer(ground_truth)
    return output_ans == ground_truth_ans


def run_model(train, test, model, sample_size, prompting_t):
    results = []
    total_tokens = 0
    total_time = 0
    total_correct = 0
    counter = 0

    for sample in train.select(range(sample_size)):
        problem = sample['problem']

        start_time = time() # should I do this call and latency call later within if statements?
        if (prompting_t == ""):
            response = apis.call_default_api(problem, model, SYSTEM_INPUT)
        elif (prompting_t == "few_shot"):
            # we pass in test data for few-shot examples since we don't use that
            response = apis.call_few_shot_api(problem, model, SYSTEM_INPUT, test, "math");
        elif (prompting_t == "COT"):
            response = apis.call_COT_api(problem, model, SYSTEM_INPUT)
        else:
            print("prompting technique not supported")
            return
        latency = time() - start_time

        output_solution = response.choices[0].message.content
        num_tokens = response.usage.total_tokens
        correct = check_answer(output_solution, sample['solution'])

        results.append(
            {
                'problem': problem,
                'level': sample['level'],
                'type': sample['type'],
                'solution': sample['solution'],
                'model_output': output_solution,
                'correct':correct,
                'latency': latency,
                'num_tokens': num_tokens
            }
        )
        total_tokens += num_tokens
        total_time += latency
        total_correct += correct
        counter += 1
        print(f"processed {counter}/{sample_size}")

    return results, total_tokens, total_time, total_correct


