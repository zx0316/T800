## 安装依赖
~~~shell
#安装python 3.9版本
brew install python@3.9

#创建虚拟环境
python3.9 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# pip安装依赖库
pip install oss2 opencv-python openai pyaudio vosk numpy pyttsx3 dlib DeepFace librosa noisereduce Resemblyzer

~~~


参考:
1. https://www.yunxin360.com/archives/434
2. https://www.cnblogs.com/XuXiaoCong/p/18555764