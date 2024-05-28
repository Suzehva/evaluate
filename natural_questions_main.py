from typing import Optional
from datasets import load_dataset
from time import time
from openai import OpenAI
import apis
from prompting import natural_questions_prompts
# from preprocess import natural_questions_pre #(nothing to preprocess for now)
import evaluation

FEWSHOT_SIZE = 5 # amount of samples used for few_shot
NUM_EXPERTS = 3 # tree of thought

def load_data():
    train_dataset = load_dataset("natural_questions", split="train", trust_remote_code=True, streaming=True)
    test_dataset  = load_dataset("natural_questions", split="validation", trust_remote_code=True, streaming=True)

    print("Loaded dataset")
    return train_dataset, test_dataset

def run_model(train, test, model, sample_size, prompting_t):
    #TODO: include close book
    results = []
    total_tokens = 0
    total_time = 0
    total_correct = 0
    counter = 0
    average_f1 = 0

    fewshot_counter = 0

    prompt = natural_questions_prompts.get_prompt(prompting_t, train)
    #if prompting_t == "COT_FS" or prompting_t == "FS": # filter out fs examples from train
        #train = train.select(range(FEWSHOT_SIZE, len(train)))
    
    
    i = 0
    sample_size_changing = sample_size
    #for sample in train.select(range(sample_size)):
    for sample in train:
        if i >= sample_size_changing:
            break
        if (prompting_t == "COT_FS" or prompting_t == "FS") and fewshot_counter < FEWSHOT_SIZE:
            fewshot_counter+= 1
            continue


        i += 1

        question = sample['question']['text']
        answers = sample['annotations']['short_answers']
        if (len(answers[0]['text']) == 0):
            # we only consider samples with short_answers marked by annotator
            sample_size_changing += 1
            continue

        start_time = time() # should I do this call and latency call later within if statements?
        response = apis.call_default_api(question, model, prompt)
        latency = time() - start_time
        output_answer = response.choices[0].message.content
        num_tokens = response.usage.total_tokens

        f1 = 0
        answer = ""
        for answer in answers:
            # we take the f1 score of the output_answer combined with the answer that is the closest
            f1_temp = evaluation.f1_score(output_answer, answer['text'][0])
            if f1_temp > f1:
                f1 = f1_temp
                answer = answer['text'][0]
            #f1 = max(f1, f1_score(output_answer, answer['text'][0]))

        results.append(
            {
                'question': question,
                'answer': answer,
                'model_output': output_answer,
                'f1': f1,
                'latency': latency,
                'num_tokens': num_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens
            }
        )
        total_tokens += num_tokens
        total_time += latency
        average_f1 += f1
        counter += 1
        print(f"processed {counter}/{sample_size}")

    average_f1 = average_f1 / sample_size
    return results, total_tokens, total_time, average_f1, prompt
