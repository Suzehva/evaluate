# evaluate
evaluate is a benchmarking framework for evaluating various LLM pipelines.

# Setup
To manage dependencies, we recommend using a Python venv. First, ensure that you have Python installed (see https://cloud.google.com/python/docs/setup).

Then, set up a venv and install the required dependencies using pip within the environment.
```bash
cd evaluate/
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Next, because evaluate uses ChatGPT as the LLM to prompt, get an OpenAI key (see https://platform.openai.com/docs/quickstart).

# Running
To run evaluate, make sure you are in the evaluate directory on your computer. Try running the following to see if the setup was successfull:
```bash
python main.py --model gpt-3.5-turbo --sample_size 1 --dataset math --prompting_type ""
```
You should see two files pop up in the results folder: a results file and a total file. The results file has the data for each individual request and the total file has data common to all requests.

For a thorough check, run the following:
```bash
./scripts/error_check.sh
```
This will run all available pipelines once with gpt-3.5-turbo.


