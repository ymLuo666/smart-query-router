import openai
import os

def chat(query: str):
    model_name = 'qwen-plus'
    max_length = 100

    # Configure API credentials
    api_key = os.getenv('QIANWEN_API_KEY', None) or os.getenv("DASHSCOPE_API_KEY", "xxx")
    openai.api_key = api_key
    openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Create messages for the API
    messages = [
        {'role': 'system', 'content': "You are a research assistant. ONLY provide relevant background information, concepts, and definitions to understand the topic. Do NOT solve the problem or give direct answers. Always respond in English."},
        {'role': 'user', 'content': f"Find ONLY background information in English about this topic (DO NOT solve it): {query}"}
    ]

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=max_length,
    )

    if 'choices' in completion and len(completion['choices']) > 0:
        search_result = completion['choices'][0]['message']['content']
        return f"{search_result}"

    return f"{str(completion)}"

if __name__ == '__main__':
    chat('what is your name?')
