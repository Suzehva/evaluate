from typing import Optional
from datasets import load_dataset
from time import time
from openai import OpenAI
import apis
from prompting import natural_questions_prompts
from preprocess import natural_questions_pre
import evaluation
import ensemble
#import main
import config

def load_data():
    train_dataset = load_dataset("natural_questions", split="train", trust_remote_code=True, streaming=True)
    test_dataset  = load_dataset("natural_questions", split="validation", trust_remote_code=True, streaming=True)

    print("Loaded dataset")
    return train_dataset, test_dataset


def run_model(train, test):
    #TODO: include closed book
    results = []
    total_tokens = 0
    total_time = 0
    total_correct = 0
    counter = 0
    average_f1 = 0
    fewshot_counter = 0

    prompt = natural_questions_prompts.get_prompt(config.PROMPTING_T, train)
    for sample in train:
        if counter >= config.SAMPLE_SIZE:
            break
        if (config.PROMPTING_T == "COT_FS" or config.PROMPTING_T == "FS") and fewshot_counter < config.FEWSHOT_SIZE:
            fewshot_counter+= 1
            continue
        counter += 1

        question = sample['question']['text']
        answers = sample['annotations']['short_answers']
        if (len(answers[0]['text']) == 0):
            # we only consider samples with short_answers marked by annotator
            counter -= 1 # this iteration doesnt count
            continue

        start_time = time()
        if config.USE_ENSEMBLE:
            responses = []
            num_tokens = 0
            completion_tokens = 0
            prompt_tokens = 0

            for i in range(config.ENSEMBLE_SIZE):
                response = apis.call_default_api(question, config.MODEL, prompt)

                num_tokens += response.usage.total_tokens
                completion_tokens += response.usage.completion_tokens
                prompt_tokens += response.usage.prompt_tokens

                response_str = response.choices[0].message.content
                #response_str = natural_questions_pre.extract_answer(response_str)
                responses.append(response_str)
            latency = time() - start_time
            output_answer = ensemble.take_highest_f1_natural_questions(responses, answers) 

        else:
            response = apis.call_default_api(question, config.MODEL, prompt)
            latency = time() - start_time
            output_answer = response.choices[0].message.content

            num_tokens = response.usage.total_tokens
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens

            output_answer = natural_questions_pre.extract_answer(output_answer)
        start_time = time() # should I do this call and latency call later within if statements?
        response = apis.call_default_api(question, config.MODEL, prompt)
        latency = time() - start_time
        output_answer = response.choices[0].message.content
        output_answer = natural_questions_pre.extract_answer(output_answer)
        num_tokens = response.usage.total_tokens

        f1 = 0
        answer = answers[0]['text'][0] # take 1st answer as default
        for temp_ans in answers:
            # we take the f1 score of the output_answer combined with the answer that is the closest
            f1_temp = evaluation.f1_score(output_answer, temp_ans['text'][0])
            if f1_temp > f1:
                f1 = f1_temp
                answer = temp_ans['text'][0]

        results.append(
            {
                'question': question,
                'answer': answer,
                'model_output': output_answer,
                'f1': f1,
                'latency': latency,
                'num_tokens': num_tokens,
                'completion_tokens': completion_tokens,
                'prompt_tokens': prompt_tokens
            }
        )
        total_tokens += num_tokens
        total_time += latency
        average_f1 += f1
        print(f"processed {counter}/{config.SAMPLE_SIZE}")

    average_f1 = average_f1 / config.SAMPLE_SIZE
    return results, total_tokens, total_time, average_f1, prompt

