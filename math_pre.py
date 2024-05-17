# TODO: create class to preprocess math data (select only last data between {}, latex format
# Return a pandas df with the following columns: question, original answer, original processed answer, level, genre

from typing import Optional
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


def remove_boxed(string: str) -> Optional[str]:
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


def last_boxed_only_string(string: str) -> Optional[str]:
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


def _fix_fracs(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions.

    Examples:
    >>> _fix_fracs("\\frac1b")
    \frac{1}{b}
    >>> _fix_fracs("\\frac12")
    \frac{1}{2}
    >>> _fix_fracs("\\frac1{72}")
    \frac{1}{72}
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions formatted as a/b to \\frac{a}{b}.

    Example:
    >>> _fix_a_slash_b("2/3")
    \frac{2}{3}
    """
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Remove units (on the right).
    "\\text{ " only ever occurs (at least in the val set) when describing units.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat square roots.

    Example:
    >>> _fix_sqrt("\\sqrt3")
    \sqrt{3}
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Apply the reformatting helper functions above.
    """
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
    
def get_answer(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer

def is_equiv(str1: Optional[str], str2: Optional[str]) -> float:
    """Returns (as a float) whether two strings containing math are equivalent up to differences of formatting in
    - units
    - fractions
    - square roots
    - superfluous LaTeX.

    Source: https://github.com/hendrycks/math
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return float(ss1 == ss2)
    except Exception:
        return float(str1 == str2)   

def check_answer(output: str, ground_truth: str) -> int:
    output_ans = get_answer(output)
    ground_truth_ans = get_answer(ground_truth)
    return is_equiv(output_ans, ground_truth_ans)

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


