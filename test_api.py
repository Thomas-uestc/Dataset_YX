from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-4o-mini",
  input="write a haiku about ai",
  store=True,
)

print(response.choices[0].message.content);
