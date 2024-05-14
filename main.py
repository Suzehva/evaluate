# to run: python3 main.py

from datasets import load_dataset
#from openai import OpenAI
import sys
from datetime import datetime
import pandas as pd
import math_pre
import natural_questions_pre
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
    2. ???
"""

PROMPTING_T = "few_shot"
"""
Options:
    1. "few_shot" 
    2  "" (no prompting technique)
    3. "COT"
"""

SAMPLE_SIZE = 10 # how many data points to include


def main():
    
    if (DATASET == "math"):
        # preprocess math
        train, test = math_pre.load_data()
        results, total_tokens, total_time, total_correct = math_pre.run_model(train, test, MODEL, SAMPLE_SIZE, PROMPTING_T)
       
        accuracy = total_correct / SAMPLE_SIZE
        print("\nAccuracy:", accuracy)

    elif (DATASET == "natural_questions"):
        train, validation = natural_questions_pre.load_data()
        if (train is None):
            print("train is none")
        if (validation is None):
            print ("validation is none")
        results, total_tokens, total_time, average_f1 = natural_questions_pre.run_model(train, validation, MODEL, SAMPLE_SIZE, PROMPTING_T)
        average_latency = total_time / SAMPLE_SIZE
        print("\naverage f1 score:", average_f1)

    print("Average Latency (s):", average_latency)
    print("Total Tokens Used:", total_tokens)
    print("Number of datapoints used: ", SAMPLE_SIZE)
    if (PROMPTING_T == "few_shot"):
        print("number of examples used per few_shot: ", apis.FEW_SHOT_SIZE)

       
    df = pd.DataFrame(results)
    df.to_csv(f"results/result_{DATASET}_{MODEL}_{PROMPTING_T}_.csv")


    # TODO: ask user whether it wants to use RAG, any tools/agents


if __name__ == "__main__":
    main()
