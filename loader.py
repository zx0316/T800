from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 OpenAI API 密钥
openai_api_key = os.getenv("OPENAI_API_KEY")

