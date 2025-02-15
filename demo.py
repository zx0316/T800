import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import os
import logging
import time
import random
import cv2
from openai import OpenAI
import pyaudio
import wave
import json
from vosk import Model, KaldiRecognizer
import numpy as np
import pyttsx3
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# 检查环境变量是否已设置
required_env_vars = ['OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET', 'DASHSCOPE_API_KEY']
for var in required_env_vars:
    if var not in os.environ:
        logging.error(f"Environment variable {var} is not set.")
        exit(1)

# 从环境变量中获取访问凭证
auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

# 设置 Endpoint 和 Region
endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
region = "cn-hangzhou"

def generate_unique_bucket_name():
    # 获取当前时间戳
    timestamp = int(time.time())
    # 生成 0 到 9999 之间的随机数
    random_number = random.randint(0, 9999)
    # 构建唯一的 Bucket 名称
    bucket_name = f"demo-{timestamp}-{random_number}"
    return bucket_name

# 生成唯一的 Bucket 名称
bucket_name = generate_unique_bucket_name()
bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

def create_bucket(bucket):
    try:
        bucket.create_bucket(oss2.models.BUCKET_ACL_PUBLIC_READ)
        logging.info("Bucket created successfully")
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to create bucket: {e}")

def upload_file(bucket, object_name, data):
    try:
        result = bucket.put_object(object_name, data)
        logging.info(f"File uploaded successfully, status code: {result.status}")
        # 生成公共读文件 URL
        public_url = f'https://{bucket.bucket_name}.{endpoint.replace("https://", "")}/{object_name}'
        return public_url
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to upload file: {e}")
        return None

# 定义对话历史列表
conversation_history = []
# 定义存储最近 2 次图片 URL 的列表
recent_image_urls = []

# 定义最大历史对话数量
MAX_CONVERSATION_HISTORY = 5

def call_large_model(image_urls, audio_text):
    global conversation_history
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        # 构建当前用户消息
        content = [{"type": "text", "text": f"'{audio_text}'"}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        user_message = {
            "role": "user",
            "content": content
        }
        # 将当前用户消息添加到对话历史中
        conversation_history.append(user_message)

        # 检查对话历史长度，若超过 5 条则移除最早的记录
        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
            conversation_history.pop(0)

        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=conversation_history
        )
        response_text = completion.choices[0].message.content.strip()
        print(response_text)

        # 将模型回复添加到对话历史中
        assistant_message = {
            "role": "assistant",
            "content": response_text
        }
        conversation_history.append(assistant_message)

        # 再次检查对话历史长度，若超过 5 条则移除最早的记录
        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
            conversation_history.pop(0)

        return response_text
    except Exception as e:
        logging.error(f"Failed to call large model: {e}")
        return None

def capture_audio_with_vosk(model_path):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SILENCE_THRESHOLD = 500  # 静音阈值
    SILENCE_DURATION = 1.5   # 静音持续时间（秒）

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    logging.info("* 开始录制音频")

    # 加载 Vosk 中文模型
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, RATE)

    frames = []
    silence_frames = 0
    is_speaking = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        # 使用 Vosk 检测语音
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_dict = json.loads(result)
            text = result_dict.get("text", "").strip()
            if text:
                logging.info(f"Detected speech: {text}")
                is_speaking = True

        # 检测静音
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
            silence_frames += 1
        else:
            silence_frames = 0

        # 如果检测到静音超过指定时间，则认为用户说完一句话
        if is_speaking and silence_frames > (SILENCE_DURATION * RATE / CHUNK):
            break

    logging.info("* 音频录制结束")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将音频数据保存为临时文件
    audio_filename = f"audio_{int(time.time() * 1000)}.wav"
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    with open(audio_filename, 'rb') as audio_file:
        audio_data = audio_file.read()
    os.remove(audio_filename)

    return audio_data

def audio_to_text_with_vosk(audio_data, model_path):
    model = Model(model_path)  # 根据需要选择语言模型
    recognizer = KaldiRecognizer(model, 16000)

    recognizer.AcceptWaveform(audio_data)
    result = recognizer.FinalResult()
    result_dict = json.loads(result)
    return result_dict.get("text", "").strip()

def text_to_speech(text):
    # 初始化 pyttsx3 引擎
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)  # 设置语速
    engine.setProperty('volume', 1.0)  # 设置音量
    engine.say(text)
    engine.runAndWait()
    logging.info("Text-to-speech completed")

def capture_images(cap, stop_event):
    global recent_image_urls
    last_capture_time = time.time()
    while not stop_event.is_set():
        current_time = time.time()
        if current_time - last_capture_time >= 10:  # 每隔 500ms 抓一张图片
            ret, frame = cap.read()
            if ret:
                # 将图像编码为 JPEG 格式
                _, buffer = cv2.imencode('.jpeg', frame)
                # 生成唯一的对象名称
                image_object_name = f"image_{int(time.time() * 1000)}.jpeg"
                # 上传图像文件并获取 URL
                image_url = upload_file(bucket, image_object_name, buffer.tobytes())
                if image_url:
                    recent_image_urls.append(image_url)
                    if len(recent_image_urls) > 2:
                        recent_image_urls.pop(0)  # 只保留最近 2 次的图片 URL
            last_capture_time = current_time
        time.sleep(1)

# 主流程
if __name__ == '__main__':
    # 1. 创建 Bucket
    create_bucket(bucket)

    # 打开摄像头，参数 0 表示默认摄像头，如果有多个摄像头可以尝试不同的索引
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 指定 Vosk 中文模型的路径
    model_path = "/Users/hobby/.cache/vosk/vosk-model-small-cn-0.22"  # 替换为您的模型路径

    try:
        stop_event = threading.Event()
        # 启动图像捕获线程
        image_thread = threading.Thread(target=capture_images, args=(cap, stop_event))
        image_thread.start()

        while True:
            # 捕获音频
            audio_data = capture_audio_with_vosk(model_path)
            # 音频转文字
            audio_text = audio_to_text_with_vosk(audio_data, model_path)

            if audio_text and recent_image_urls:
                # 调用大模型进行分析
                response_text = call_large_model(recent_image_urls, audio_text)

                if response_text:
                    # 文本转语音并播放
                    text_to_speech(response_text)

    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
        stop_event.set()
    finally:
        # 释放摄像头资源
        cap.release()
        # 关闭所有打开的窗口
        cv2.destroyAllWindows()