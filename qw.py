import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=[{"role": "user","content": [
            {"type": "text","text": "这是什么"},
            {"type": "image_url",
             "image_url": {"url": "https://demo-1739185756-9595.oss-cn-hangzhou.aliyuncs.com/image_1739185758342.jpg"}}
            ]}]
    )
print(completion.model_dump_json())