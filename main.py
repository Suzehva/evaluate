# to run: python3 main.py
#from openai import OpenAI
import sys
from datetime import datetime
import pandas as pd
import math_main
import natural_questions_main
import apis
import argparse

DATASET = "math"
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

FEWSHOT_SIZE = 5 # always set this to 0 if not using fewshot
NUM_EXPERTS = 3 # tree of thought
ENSEMBLE_SIZE = 4 # ensembling

def main():
    
    if (DATASET == "math"):
        train, test = math_main.load_data()
        results, total_tokens, total_time, total_correct, prompt = math_main.run_model(train, test)
       
        accuracy = total_correct / SAMPLE_SIZE
        print("\nAccuracy:", accuracy)

    elif (DATASET == "natural_questions"):
        train, validation = natural_questions_main.load_data()
        results, total_tokens, total_time, average_f1, prompt = natural_questions_main.run_model(train, validation)
        print("\naverage f1 score:", average_f1)
    
    average_latency = total_time / SAMPLE_SIZE
    print("Average Latency (s):", average_latency)
    print("Total Tokens Used:", total_tokens)
    print("Number of datapoints used: ", SAMPLE_SIZE)
    if (PROMPTING_T == "FS"):
        print("number of examples used per few_shot: ", FEWSHOT_SIZE)

       
    df = pd.DataFrame(results)
    
    if USE_ENSEMBLE:
        df.to_csv(f"results/result_{DATASET}_{MODEL}_{PROMPTING_T}_ensemble_{datetime.now()}.csv")
        file_path = f'results/total_{DATASET}_{MODEL}_{PROMPTING_T}_ensemble_{datetime.now()}.txt'
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
            file.write(f"Few Shot Examples: {FEWSHOT_SIZE}\n")
        if USE_ENSEMBLE:
            file.write(f"Ensemble Size: {ENSEMBLE_SIZE}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify pipeline properties')
    parser.add_argument('--dataset', type=str, required=True, choices=['math', 'natural_questions'], help='Dataset to use (math or natural_questions)')
    parser.add_argument('--model', type=str, required=True, choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'], help='LLM model to use (e.g., gpt-3.5-turbo, gpt-4)')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of data points to include')
    parser.add_argument('--prompting_type', type=str, choices=['FS', 'COT', 'TOT', 'COT_FS', ''], help='Type of prompting (FS, COT, TOT, COT_FS)')
    parser.add_argument('--prompting_arg', type=int, default=5, help='Optionally provide a number of prompts for COT_FS/FS, or number of experts for TOT')
    parser.add_argument('--use_ensemble', action='store_true', help='Optionally use ensembling')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of models to use for ensembling')
    args = parser.parse_args()


    DATASET = args.dataset
    MODEL = args.model
    SAMPLE_SIZE = args.sample_size
    PROMPTING_T = args.prompting_type 
    if PROMPTING_T == 'FS' or PROMPTING_T == 'COT_FS':
        FEWSHOT_SIZE = args.prompting_arg
    elif PROMPTING_T == 'TOT':
        NUM_EXPERTS = args.prompting_arg
    USE_ENSEMBLE = args.use_ensemble
    if USE_ENSEMBLE:
        ENSEMBLE_SIZE = args.ensemble_size
    
    main()
