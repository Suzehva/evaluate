from openai import OpenAI
client = OpenAI()

def call_default_api(input, model, system_prompt, temperature = 1):
    supported_models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "gpt-4"]
    if model not in supported_models:
        error("specified model is not supported")
    
    return client.chat.completions.create(
        model=model,
        temperature = temperature,
        messages=[
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": input}
        ]
    )
