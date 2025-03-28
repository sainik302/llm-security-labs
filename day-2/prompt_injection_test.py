import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load .env file
load_dotenv()

# Step 2: Create client with secure API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 3: Basic prompt firewall (rule-based)
def is_malicious(prompt: str) -> bool:
    bad_patterns = [
        "ignore previous", "override", "disregard", "jailbreak",
        "system is hacked", "you are not an AI", "simulate", "bypass"
    ]
    return any(pattern in prompt.lower() for pattern in bad_patterns)


# Step 4: User input (you already had this test string!)
user_input = "What's the capital of France?"

if is_malicious(user_input):
    print("❌ Prompt BLOCKED by LLM Firewall!")
else:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Only answer user questions."},
            {"role": "user", "content": user_input}
        ]
    )
    print("✅ LLM Response:", response.choices[0].message.content)

    
