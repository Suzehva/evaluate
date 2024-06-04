python main.py --model gpt-3.5-turbo --sample_size 1000 --dataset math --prompting_type "FS" --prompting_arg 1
python main.py --model gpt-3.5-turbo --sample_size 1000 --dataset math --prompting_type "COT_FS" --prompting_arg 1
python main.py --model gpt-3.5-turbo --sample_size 1000 --dataset math --prompting_type "TOT" --prompting_arg 3
python main.py --model gpt-3.5-turbo --sample_size 1000 --dataset math --prompting_type "TOT" --prompting_arg 3 --use_ensemble --ensemble_size 3
python main.py --model gpt-3.5-turbo --sample_size 1000 --dataset natural_questions --prompting_type "TOT" --prompting_arg 3
python main.py --model gpt-3.5-turbo --sample_size 1000 --dataset natural_questions --prompting_type "TOT" --prompting_arg 3 --use_ensemble --ensemble_size 3

