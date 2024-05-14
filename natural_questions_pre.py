from datasets import load_dataset
from time import time
from openai import OpenAI
import apis

# for f1 score
import unicodedata
import string
import re
from collections import Counter

SYSTEM_INPUT = "You answer questions. Provide a concise short answer, preferable a one word answer. \n"

def load_data():
    # streaming set to true because this is an enormous dataset
    train_dataset = load_dataset("natural_questions", split="train", trust_remote_code=True, streaming=True)
    validation_dataset  = load_dataset("natural_questions", split="validation", trust_remote_code=True, streaming=True)
    print("Loaded dataset")
    return train_dataset, validation_dataset

def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(output, ground_truth):
    # from Jungle
    prediction_tokens = normalize_text(output).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def run_model(train, validation, model, sample_size, prompting_t):
    results = []
    # TODO:closedbook vs other approaches (now always closedbook)

    total_tokens = 0
    total_time = 0
    average_f1 = 0
    counter = 0

    i = 0
    sample_size_changing = sample_size
    #for sample in train.select(range(sample_size)):
    for sample in train:
        if i >= sample_size_changing:
            break
        i += 1

        question = sample['question']['text']
        answers = sample['annotations']['short_answers']
        if (len(answers[0]['text']) == 0):
            # we only consider samples with short_answers marked by annotator
            sample_size_changing += 1
            continue

        #answer = sample['annotations']['short_answers'][0]['text']
        #print(answer)

        start_time = time() # should I do this call and latency call later within if statements?
        if (prompting_t == ""):
            response = apis.call_default_api(question, model, SYSTEM_INPUT)
        elif (prompting_t == "few_shot"):
            # we pass in test data for few-shot examples since we don't use that
            response = apis.call_few_shot_api(question, model, SYSTEM_INPUT, validation, "natural_questions");
        elif (prompting_t == "COT"):
            response = apis.call_COT_api(question, model, SYSTEM_INPUT)
        else:
            print("prompting technique not supported")
            return
        latency = time() - start_time

        output_answer = response.choices[0].message.content
        num_tokens = response.usage.total_tokens
        f1 = 0
        for answer in answers:
            # we take the f1 score of the output_answer combined with the answer that is the closest
            f1 = max(f1, f1_score(output_answer, answer['text'][0]))

        results.append(
            {
                'question': question,
                'answer': answer['text'],
                'model_output': output_answer,
                'f1':f1,
                'latency': latency,
                'num_tokens': num_tokens
            }
        )
        total_tokens += num_tokens
        total_time += latency
        average_f1 += f1
        counter += 1
        print(f"processed {counter}/{sample_size}")

    average_f1 = average_f1 / sample_size
    return results, total_tokens, total_time, average_f1
