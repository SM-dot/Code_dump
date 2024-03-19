import openai

openai.api_key = "### Enter your OpenI key"


completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0125", messages=[{"role": "user", "content": "Give me 3 ideas for apps I could build with openai apis "}])
print(completion.choices[0].message.content)
