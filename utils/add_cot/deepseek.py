import os
from openai import OpenAI

def main():
    # test
    client = OpenAI(
        api_key="xxx", 
        base_url="xxx",
        timeout=1800,
        )
    response = client.chat.completions.create(
        # 替换 <Model> 为模型的Model ID
        model="xxx",  
        messages=[
            {"role": "user", "content": "我要有研究推理模型与非推理模型区别的课题，怎么体现我的专业性"}
        ]
    )
    # 当触发深度推理时，打印思维链内容
    if hasattr(response.choices[0].message, 'reasoning_content'):
        print("Thinking process:")
        print(response.choices[0].message.reasoning_content)
    print("Answer process:")
    print(response.choices[0].message.content)


def call_deepseek_V3(system_content, user_content):
    client = OpenAI(
        api_key="xxx", 
        base_url="xxx",
        timeout=1800,
    )
    
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    
    response = client.chat.completions.create(
        model="xxx",
        messages=messages
    )
    
    return response.choices[0].message.content

def call_deepseek_R1(system_content, user_content):
    client = OpenAI(
        api_key="xxx", 
        base_url="xxx",
        timeout=1800,
    )
    
    messages = []
    
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    try:
        response = client.chat.completions.create(
            model="xxx",
            messages=messages
        )
    except Exception as e:
        print(f"Error: {e}")
        return ""
    return response.choices[0].message.content, response.choices[0].message.reasoning_content


if __name__ == "__main__":
    main()