# to run: python3 main.py
#from openai import OpenAI
import sys
from datetime import datetime
import pandas as pd
import math_main
import natural_questions_main
import apis
import argparse

DATASET = "natural_questions"
"""
Options:
    1. math (hendrycks/competition_math)
    2. natural_questions
"""

MODEL = "gpt-4-turbo"
"""
Options:
    1. gpt-3.5-turbo
    2. gpt-4
    3. gpt-4-turbo
    4. gpt-4o
"""

PROMPTING_T = ""
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
    
    if USE_ENSEMBLE:
        df.to_csv(f"results/result_{DATASET}_{MODEL}_{PROMPTING_T}_ensemble.csv")
        file_path = f'results/total_{DATASET}_{MODEL}_{PROMPTING_T}_ensemble.txt'
    else:
        df.to_csv(f"results/result_{DATASET}_{MODEL}_{PROMPTING_T}.csv")
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
            if DATASET == "math":
                file.write(f"Few Shot Examples: {math_main.FEWSHOT_SIZE}\n")
            elif DATASET == "natural_questions":
                file.write(f"Few Shot Examples: {natural_questions_main.FEWSHOT_SIZE}\n")
        if USE_ENSEMBLE:
            var_name = f"{DATASET}_main.ENSEMBLE_SIZE"
            file.write(f"Ensemble Size: {var_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify pipeline properties')
    parser.add_argument(
        'model_version',
        type=str,
        choices=['gpt-3.5', 'gpt-4', 'gpt-4-turbo', 'gpt-4-o', 'all'],
        help='Specify the GPT model version to use.'
    )
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (math or natural_questions)')

    parser.add_argument('--model', type=str, required=True, choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'], help='LLM model to use (e.g., gpt-3.5-turbo, gpt-4)')
    parser.add_argument('--prompting_type', type=str, help='Type of prompting (FS, COT, TOT, COT_FS)')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of data points to include')
    parser.add_argument('--use_ensemble', action='store_true', help='Optionally use ensembling for math')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensembles to use')

    args = parser.parse_args()

    main()
