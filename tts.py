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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 全局状态管理
class AgentState:
    def __init__(self):
        self.owner_face = None
        self.owner_name = None
        self.current_emotion = None
        self.emotion_history = deque(maxlen=10)
        self.conversation_history = []
        self.last_interaction_time = time.time()

agent_state = AgentState()

# 音频处理参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

# 初始化OSS
auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
bucket_name = f"ai-agent-{int(time.time())}-{random.randint(1000,9999)}"
bucket = oss2.Bucket(auth, endpoint, bucket_name)

def setup_oss():
    try:
        bucket.create_bucket(oss2.models.BUCKET_ACL_PUBLIC_READ)
        logging.info("OSS bucket created")
    except Exception as e:
        logging.error(f"OSS setup failed: {e}")

def process_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    
    # 获取面部特征
    face = faces[0]
    shape = predictor(gray, face)
    face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
    
    return np.array(face_descriptor), face

def capture_reference_face(cap):
    """捕获主人的参考面部特征"""
    ret, frame = cap.read()
    if not ret:
        return None
    
    descriptor, face = process_face(frame)
    if descriptor is None:
        return None
    
    # 保存参考图像
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.imwrite("reference_face.jpg", frame[y:y+h, x:x+w])
    return descriptor

def analyze_emotion(frame):
    """使用DeepFace分析情绪"""
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        logging.error(f"Emotion analysis failed: {e}")
        return None

def emotion_based_response(emotion):
    """根据情绪生成主动问候"""
    responses = {
        'happy': ["今天有什么开心的事吗？", "你看上去心情不错！"],
        'sad': ["想和我聊聊吗？", "我在这里陪着你..."],
        'angry': ["需要我帮你冷静下来吗？", "深呼吸..."],
        'surprise': ["哇！有什么惊喜吗？"],
        'fear': ["别害怕，我在这里..."],
        'neutral': ["今天过得怎么样？"]
    }
    return random.choice(responses.get(emotion, ["有什么我可以帮忙的吗？"]))

def vision_processing(cap, stop_event):
    """实时视觉处理线程"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 人脸识别
        if agent_state.owner_face is not None:
            current_descriptor, face = process_face(frame)
            if current_descriptor is not None:
                # 计算相似度
                similarity = np.linalg.norm(agent_state.owner_face - current_descriptor)
                if similarity < 0.6:  # 相似度阈值
                    # 情绪分析
                    emotion = analyze_emotion(frame)
                    if emotion:
                        agent_state.emotion_history.append(emotion)
                        agent_state.current_emotion = max(
                            set(agent_state.emotion_history), 
                            key=agent_state.emotion_history.count
                        )
                        
                        # 主动交互逻辑
                        if time.time() - agent_state.last_interaction_time > 30:
                            response = emotion_based_response(agent_state.current_emotion)
                            text_to_speech(response)
                            agent_state.last_interaction_time = time.time()
        time.sleep(0.5)

def initialize_owner_identity(cap):
    """初始化主人身份流程"""
    text_to_speech("您好，我是您的个人助理，请告诉我您是谁？")
    
    # 获取语音回应
    audio_data = capture_audio()
    name = audio_to_text(audio_data)
    
    # 捕获面部特征
    text_to_speech("请面对摄像头让我记住您的样子...")
    time.sleep(2)
    face_descriptor = capture_reference_face(cap)
    
    if face_descriptor is not None and name:
        agent_state.owner_face = face_descriptor
        agent_state.owner_name = name
        text_to_speech(f"很高兴为您服务，{name}！")
        return True
    return False

def main():
    setup_oss()
    cap = cv2.VideoCapture(0)
    
    if not initialize_owner_identity(cap):
        logging.error("初始化失败")
        return

    stop_event = threading.Event()
    
    # 启动视觉处理线程
    vision_thread = threading.Thread(target=vision_processing, args=(cap, stop_event))
    vision_thread.start()

    # 主交互循环
    while True:
        try:
            audio_data = capture_audio()
            text = audio_to_text(audio_data)
            
            if text:
                # 获取当前视觉上下文
                context = {
                    "emotion": agent_state.current_emotion,
                    "user": agent_state.owner_name
                }
                
                # 生成响应
                response = generate_response(text, context)
                text_to_speech(response)
                
        except KeyboardInterrupt:
            stop_event.set()
            break
    
    vision_thread.join()
    cap.release()

if __name__ == "__main__":
    main()