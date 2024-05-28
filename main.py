# to run: python3 main.py
#from openai import OpenAI
import sys
from datetime import datetime
import pandas as pd
import math_main
import natural_questions_main
import apis

DATASET = "natural_questions"
"""
Options:
    1. math (hendrycks/competition_math)
    2. natural_questions
"""

MODEL = "gpt-3.5-turbo"
"""
Options:
    1. gpt-3.5-turbo
    2. gpt-4
    3. gpt-4-turbo
    4. gpt-4o
"""

PROMPTING_T = "COT_FS"
"""
Options:
    1. "FS" (few shot)
    2  "" (no prompting technique)
    3. "COT" (chain of thought)
    4. "TOT" (tree of thought)
    5. "COT_FS"
"""

SAMPLE_SIZE = 5 # how many data points to include

USE_ENSEMBLE = False # optionally use ensembling for math

ENSEMBLE_SIZE = 5 # number of ensembles to use

def main():
    
    if (DATASET == "math"):
        train, test = math_main.load_data()
        results, total_tokens, total_time, total_correct, prompt = math_main.run_model(train, test, MODEL, SAMPLE_SIZE, PROMPTING_T, USE_ENSEMBLE)
       
        accuracy = total_correct / SAMPLE_SIZE
        print("\nAccuracy:", accuracy)

    elif (DATASET == "natural_questions"):
        train, validation = natural_questions_main.load_data()
        results, total_tokens, total_time, average_f1, prompt = natural_questions_main.run_model(train, validation, MODEL, SAMPLE_SIZE, PROMPTING_T)
        print("\naverage f1 score:", average_f1)
    
    average_latency = total_time / SAMPLE_SIZE
    print("Average Latency (s):", average_latency)
    print("Total Tokens Used:", total_tokens)
    print("Number of datapoints used: ", SAMPLE_SIZE)
    if (PROMPTING_T == "FS"):
        print("number of examples used per few_shot: ", apis.FEW_SHOT_SIZE)

       
    df = pd.DataFrame(results)
    df.to_csv(f"results/result_{DATASET}_{MODEL}_{PROMPTING_T}_.csv")

    file_path = f'results/total_{DATASET}_{MODEL}_{PROMPTING_T}_.txt'
    with open(file_path, 'w') as file:
        file.write(f"Model Version: {MODEL}\n")
        file.write(f"Average Latency (s): {average_latency}\n")
        file.write(f"Total Tokens Used: {total_tokens}\n")
        file.write(f"Eval Sample Size: {SAMPLE_SIZE}\n")
        if DATASET == "math":
            file.write(f"Accuracy: {accuracy}\n")
        if DATASET == "natural_questions":
            file.write(f"F1 Score: {average_f1}\n")
        file.write(f"System Prompt: {prompt}\n")
        if PROMPTING_T == "FS" or PROMPTING_T == "COT_FS":
            name = f"{DATASET}_main.FEWSHOT_SIZE"
            file.write(f"Few Shot Examples: {name}\n")
        if USE_ENSEMBLE:
            var_name = f"{DATASET}_main.ENSEMBLE_SIZE"
            file.write(f"Ensemble Size: {var_name}\n")


if __name__ == "__main__":
    main()
