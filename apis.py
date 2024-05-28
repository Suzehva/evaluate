# TODO: create class with different options for different models and their API calls

from openai import OpenAI
client = OpenAI()

def call_default_api(input, model, system_prompt):
    if (model == "gpt-3.5-turbo"):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": system_prompt},
                {"role": "user", "content": input}
            ]
        )
    elif (model == "gpt-4-turbo"):
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": system_prompt},
                {"role": "user", "content": input}
            ]
        )

    return completion
