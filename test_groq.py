import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile", 
    messages=[
        {
            "role": "user",
            "content": "Hello! We are working on a BirdSQL project with a financial database. Are you ready to help us with query decomposition?"
        }
    ],
    temperature=0, 
)

print(completion.choices[0].message.content)