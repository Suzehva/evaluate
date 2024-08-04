# to run: python main.py --model gpt-3.5-turbo --sample_size 1 --dataset math --prompting_type ""
from datetime import datetime
import pandas as pd
import math_main
import natural_questions_main
import argparse
import config

def main():
    if (config.DATASET == "math"):
        train, test = math_main.load_data()
        results, total_tokens, total_time, total_correct, prompt = math_main.run_model(train, test)
        accuracy = total_correct / config.SAMPLE_SIZE

    elif (config.DATASET == "natural_questions"):
        train, validation = natural_questions_main.load_data()
        results, total_tokens, total_time, average_f1, prompt = natural_questions_main.run_model(train, validation)

    average_latency = total_time / config.SAMPLE_SIZE

    df = pd.DataFrame(results)
    
    if config.USE_ENSEMBLE:
        df.to_csv(f"results/result_{config.DATASET}_{config.MODEL}_{config.PROMPTING_T}_ensemble_{datetime.now()}.csv")
        file_path = f'results/total_{config.DATASET}_{config.MODEL}_{config.PROMPTING_T}_ensemble_{datetime.now()}.txt'
    else:
        df.to_csv(f"results/result_{config.DATASET}_{config.MODEL}_{config.PROMPTING_T}_{datetime.now()}.csv")
        file_path = f'results/total_{config.DATASET}_{config.MODEL}_{config.PROMPTING_T}_{datetime.now()}.txt'
    
    with open(file_path, 'w') as file:
        file.write(f"Model Version: {config.MODEL}\n")
        file.write(f"Average Latency (s): {average_latency}\n")
        file.write(f"Total Tokens Used: {total_tokens}\n")
        file.write(f"Eval Sample Size: {config.SAMPLE_SIZE}\n")
        if config.DATASET == "math":
            file.write(f"Accuracy: {accuracy}\n")
        if config.DATASET == "natural_questions":
            file.write(f"F1 Score: {average_f1}\n")
        file.write(f"System Prompt: {prompt}\n")
        if config.PROMPTING_T == "FS" or config.PROMPTING_T == "COT_FS":
            file.write(f"Few Shot Examples: {config.FEWSHOT_SIZE}\n")
        if config.USE_ENSEMBLE:
            file.write(f"Ensemble Size: {config.ENSEMBLE_SIZE}\n")


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
    config.initialize(args)    
    main()
