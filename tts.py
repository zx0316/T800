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
import dlib
from deepface import DeepFace
from collections import deque
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# 初始化人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# 指定 Vosk 中文模型的路径
vosk_model_path = "models/vosk-model-small-cn-0.22"

# 全局状态管理
class AgentState:
    def __init__(self):
        self.owner_face = None
        self.owner_name = None
        self.current_emotion = None
        self.emotion_history = deque(maxlen=10)
        self.conversation_history = []
        self.last_interaction_time = time.time()
        self.idle_time = 0
        self.owner_voice_features = None  # 主人的声纹特征
        self.is_speaking = False  # 标记是否正在播放语音
        self.play_queue = queue.Queue()  # 播放队列

agent_state = AgentState()

# 音频处理参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

# 初始化OSS
# 检查环境变量是否已设置
required_env_vars = ['OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET', 'DASHSCOPE_API_KEY']
for var in required_env_vars:
    if var not in os.environ:
        logging.error(f"环境变量 {var} 未设置，程序将退出。")
        exit(1)

auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
region = "cn-hangzhou"
bucket_name = f"ai-agent-{int(time.time())}-{random.randint(1000,9999)}"
bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

def setup_oss():
    try:
        logging.info("开始创建 OSS 存储桶...")
        bucket.create_bucket(oss2.models.BUCKET_ACL_PUBLIC_READ)
        logging.info("OSS 存储桶创建成功。")
    except Exception as e:
        logging.error(f"OSS 存储桶创建失败，错误信息: {e}")

def text_to_speech(text):
    try:
        pyttsx3.speak(text)
    except Exception as e:
        logging.error(f"语音合成失败，错误信息: {e}")

# 语音合成并添加到播放队列
def push_text_to_speech(text):
    agent_state.play_queue.put(text)

# 播放队列处理线程
def play_queue_worker():
    while True:
        text = agent_state.play_queue.get()
        agent_state.is_speaking = True
        try:
            pyttsx3.speak(text)
        except Exception as e:
            logging.error(f"语音播放出错: {e}")
        agent_state.is_speaking = False
        agent_state.play_queue.task_done()

def capture_audio_with_vosk():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SILENCE_THRESHOLD = 500  # 静音阈值
    SILENCE_DURATION = 1.5   # 静音持续时间（秒）

    p = pyaudio.PyAudio()

    logging.info("正在打开音频流以开始录制音频...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    logging.info("开始录制音频...")

    # 加载 Vosk 中文模型
    model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(model, RATE)

    frames = []
    silence_frames = 0
    is_speaking = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        # 检测静音
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
            silence_frames += 1
        else:
            silence_frames = 0
            if agent_state.is_speaking:
                # 如果正在播放语音，检测到说话则打断播放
                engine = pyttsx3.init()
                engine.stop()
                agent_state.is_speaking = False

        # 分块处理音频数据
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_dict = json.loads(result)
            text = result_dict.get("text", "").strip()
            if text:
                logging.info(f"检测到语音内容: {text}")
                is_speaking = True

        # 如果检测到静音超过指定时间，则认为用户说完一句话
        if is_speaking and silence_frames > (SILENCE_DURATION * RATE / CHUNK):
            break

    logging.info("音频录制结束。")

    logging.info("正在关闭音频流...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将音频数据保存为临时文件
    audio_filename = f"audio_{int(time.time() * 1000)}.wav"
    logging.info(f"将录制的音频数据保存到临时文件: {audio_filename}")
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    logging.info(f"从临时文件 {audio_filename} 读取音频数据...")
    with open(audio_filename, 'rb') as audio_file:
        audio_data = audio_file.read()
    logging.info(f"删除临时音频文件: {audio_filename}")
    os.remove(audio_filename)

    return audio_data

def audio_to_text_with_vosk(audio_data):
    logging.info("开始将音频数据转换为文本...")
    model = Model(vosk_model_path)  # 根据需要选择语言模型
    recognizer = KaldiRecognizer(model, 16000)

    recognizer.AcceptWaveform(audio_data)
    result = recognizer.FinalResult()
    result_dict = json.loads(result)
    text = result_dict.get("text", "").strip()
    logging.info(f"音频转文本结果: {text}")
    return text

def process_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None

    face = faces[0]
    shape = predictor(gray, face)
    face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
    return np.array(face_descriptor), face

def capture_reference_face(cap):
    logging.info("开始捕获主人的参考人脸图像...")
    ret, frame = cap.read()
    if not ret:
        logging.error("无法读取摄像头帧，捕获参考人脸失败。")
        return None

    descriptor, face = process_face(frame)
    if descriptor is None:
        logging.error("未能从捕获的图像中提取到有效的人脸特征。")
        return None

    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    filename = "reference_face.jpg"
    logging.info(f"将参考人脸图像保存到文件: {filename}")
    cv2.imwrite(filename, frame[y:y+h, x:x+w])
    logging.info("参考人脸图像保存成功。")
    return descriptor

def analyze_emotion(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        emotion_score = analysis[0]['emotion'][emotion]
        return emotion, emotion_score
    except Exception as e:
        logging.error(f"情绪分析失败，错误信息: {e}")
        return None, None

def emotion_based_response(emotion, emotion_score):
    responses = {
        'happy': [
            f"看你心情这么好，是不是有什么开心事要和我分享呀，你现在的情绪强度达到了{emotion_score:.2f}呢！",
            f"哇，你满脸笑容，心情超棒呀，情绪强度都有{emotion_score:.2f}了，能和我说说喜悦的来源不？"
        ],
        'sad': [
            f"感觉你有点低落呢，情绪强度有{emotion_score:.2f}，想和我倾诉一下吗？",
            f"你似乎心情不太好，情绪强度为{emotion_score:.2f}，说出来可能会让你好受些。"
        ],
        'angry': [
            f"你好像有点生气了，情绪强度是{emotion_score:.2f}，先消消气，和我说说怎么回事。",
            f"感觉你怒气值有点高，情绪强度达到{emotion_score:.2f}啦，深呼吸，和我讲讲。"
        ],
        'surprise': [
            f"你看起来很惊讶呀，情绪强度有{emotion_score:.2f}，是遇到什么意想不到的事了吗？",
            f"哇，满脸的惊讶，情绪强度{emotion_score:.2f}，快和我说说啥事儿这么惊人！"
        ],
        'fear': [
            f"你好像有点害怕，情绪强度{emotion_score:.2f}，别担心，有我陪着你。",
            f"感觉你有些恐惧，情绪强度达到{emotion_score:.2f}了，和我说说让你害怕的事儿。"
        ],
        'neutral': [
            f"今天过得怎么样？现在你的情绪比较平稳，强度为{emotion_score:.2f}。",
            f"感觉你状态还挺平和，情绪强度{emotion_score:.2f}，有没有什么想聊的？"
        ]
    }
    response = random.choice(responses.get(emotion, ["有什么我可以帮忙的吗？"]))
    return response

# 初始化声纹编码器
voice_encoder = VoiceEncoder()

def extract_voice_features(audio_data):
    try:
        logging.info("开始从音频数据中提取声纹特征...")
        audio_filename = f"temp_audio_{int(time.time() * 1000)}.wav"
        logging.info(f"将音频数据保存到临时文件: {audio_filename}")
        with open(audio_filename, 'wb') as audio_file:
            audio_file.write(audio_data)
        wav = preprocess_wav(audio_filename)
        features = voice_encoder.embed_utterance(wav)
        logging.info(f"删除临时音频文件: {audio_filename}")
        os.remove(audio_filename)
        logging.info("声纹特征提取完成。")
        return features
    except Exception as e:
        logging.error(f"声纹特征提取失败，错误信息: {e}")
        return None

def compare_voice_features(features1, features2):
    if features1 is None or features2 is None:
        logging.error("声纹特征为空，无法进行相似度比较。")
        return 0
    logging.info("开始比较两个声纹特征的相似度...")
    similarity = cosine_similarity([features1], [features2])[0][0]
    logging.info(f"声纹特征相似度计算结果为: {similarity}")
    return similarity

def initialize_owner_identity(cap):
    logging.info("开始初始化主人身份...")
    text_to_speech("您好，我是您的个人助理，请告诉我您是谁？")

    # 获取语音回应
    logging.info("正在捕获主人的语音以获取姓名...")
    audio_data = capture_audio_with_vosk()
    logging.info("开始将捕获的语音转换为文本以获取主人姓名...")
    name = audio_to_text_with_vosk(audio_data)

    # 提取主人的声纹特征
    logging.info("开始提取主人的声纹特征...")
    agent_state.owner_voice_features = extract_voice_features(audio_data)

    # 捕获面部特征
    text_to_speech("请面对摄像头让我记住您的样子...")
    logging.info("等待 2 秒后开始捕获主人的参考人脸图像...")
    time.sleep(2)
    logging.info("开始捕获主人的参考人脸图像...")
    face_descriptor = capture_reference_face(cap)

    if face_descriptor is not None and name:
        agent_state.owner_face = face_descriptor
        agent_state.owner_name = name
        text_to_speech(f"很高兴为您服务，{name}!")
        logging.info(f"主人身份初始化成功，主人姓名为: {name}")
        return True
    logging.error("主人身份初始化失败。")
    return False

def vision_processing(cap, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.error("无法从摄像头读取图像帧，视觉处理将跳过当前循环。")
            continue

        if agent_state.owner_face is not None:
            current_descriptor, face = process_face(frame)
            if current_descriptor is not None:
                similarity = np.linalg.norm(agent_state.owner_face - current_descriptor)
                if similarity < 0.6:
                    emotion, emotion_score = analyze_emotion(frame)
                    if emotion and emotion_score:
                        agent_state.emotion_history.append(emotion)
                        agent_state.current_emotion = max(
                            set(agent_state.emotion_history),
                            key=agent_state.emotion_history.count
                        )

                        idle_time = time.time() - agent_state.last_interaction_time
                        agent_state.idle_time = idle_time
                        if idle_time > 30:
                            if emotion_score > 0.7:
                                response = emotion_based_response(emotion, emotion_score)
                                push_text_to_speech(response)
                                agent_state.last_interaction_time = time.time()
                                logging.info("因主人情绪强度高触发主动交互。")
                            elif idle_time > 60:
                                suggestions = [
                                    "我知道一些有趣的故事，要不要听听？",
                                    "最近有很多热门电影，我可以给你推荐几部。",
                                    "你想不想了解一些新的知识，比如历史、科学之类的？"
                                ]
                                selected_suggestion = random.choice(suggestions)
                                push_text_to_speech(selected_suggestion)
                                agent_state.last_interaction_time = time.time()
                                logging.info("因主人长时间空闲触发主动交互。")
        time.sleep(0.5)

def generate_response(user_input, context=None):
    if not user_input:
        logging.info("用户输入为空，返回默认提示信息。")
        return "我没有听清楚，请再说一遍。"

    messages = []
    if context:
        emotion = context.get('emotion', '未知')
        emotion_score = context.get('emotion_score', '未知')
        messages.append({
            "role": "system",
            "content": f"你正在与{context.get('user', '用户')}对话。"
                       f"当前用户的情绪是{emotion}，情绪强度为{emotion_score}。"
        })

    for history in agent_state.conversation_history[-5:]:
        messages.append(history)

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        response_text = completion.choices[0].message.content.strip()
        logging.info(f"大模型生成的响应为: {response_text}")

        agent_state.conversation_history.append({"role": "user", "content": user_input})
        agent_state.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text
    except Exception as e:
        logging.error(f"调用大模型生成响应失败，错误信息: {e}，将返回默认错误提示。")
        return "抱歉，我现在无法回答这个问题。"
    
def main():
    logging.info("程序启动，开始初始化 OSS 存储桶...")
    # setup_oss()
    logging.info("尝试打开摄像头...")
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        logging.error("无法打开摄像头")
        exit()

    if not initialize_owner_identity(cap):
        logging.error("主人身份初始化失败，程序将终止。")
        exit()

    stop_event = threading.Event()

    # 启动播放队列处理线程
    play_thread = threading.Thread(target=play_queue_worker)
    play_thread.daemon = True
    play_thread.start()

    # 启动视觉处理线程
    logging.info("启动实时视觉处理线程...")
    vision_thread = threading.Thread(target=vision_processing, args=(cap, stop_event))
    vision_thread.start()

    # 主交互循环
    logging.info("进入主交互循环，等待用户语音输入...")
    while True:
        try:
            logging.info("开始捕获音频输入...")
            audio_data = capture_audio_with_vosk()
            # 提取当前音频的声纹特征
            current_features = extract_voice_features(audio_data)
            # 比较声纹特征
            if agent_state.owner_voice_features is not None:
                logging.info("开始比较当前声纹特征与主人声纹特征的相似度...")
                similarity = compare_voice_features(agent_state.owner_voice_features, current_features)
                # 可根据实际情况调整相似度阈值
                if similarity < 0.6:
                    logging.info("检测到非主人声音，忽略本次输入。")
                    continue

                # 若匹配到主人声音，打断当前播放
                if agent_state.is_speaking:
                    engine = pyttsx3.init()
                    engine.stop()
                    agent_state.is_speaking = False

            logging.info("将捕获的音频转换为文本...")
            text = audio_to_text_with_vosk(audio_data)

            if text:
                # 获取当前视觉上下文
                emotion = agent_state.current_emotion
                _, emotion_score = analyze_emotion(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
                context = {
                    "emotion": emotion,
                    "emotion_score": emotion_score if emotion_score else "未知",
                    "user": agent_state.owner_name
                }

                # 生成响应
                logging.info("根据用户输入和上下文信息生成响应...")
                response = generate_response(text, context)
                logging.info("将响应添加到播放队列...")
                push_text_to_speech(response)

                agent_state.last_interaction_time = time.time()
                agent_state.idle_time = 0

        except KeyboardInterrupt:
            logging.info("检测到用户手动中断（Ctrl+C），程序将停止。")
            stop_event.set()
            break

    logging.info("等待实时视觉处理线程结束...")
    vision_thread.join()
    logging.info("释放摄像头资源...")
    cap.release()
    logging.info("程序正常退出。")

if __name__ == "__main__":
    main()    