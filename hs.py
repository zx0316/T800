import os
from openai import OpenAI
from hs import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    openai_api_key=os.environ.get("ARK_API_KEY"),	# app_key
    model_name="ep-20250207192011-t99qr",	# 推理接入点
)

result = llm.invoke("你好，怎么称呼？")
print(result)

client = OpenAI(
    api_key = os.environ.get("ARK_API_KEY"),
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

# Image input:
response = client.chat.completions.create(
    model="ep-20250207192011-t99qr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是哪里？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
                    }
                },
            ],
        }
    ],
)

print(response.choices[0])