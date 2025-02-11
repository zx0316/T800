import cv2
import base64

# 打开摄像头，参数0表示默认摄像头，如果有多个摄像头可以尝试不同的索引
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取摄像头图像
while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 检查图像读取是否成功
    if not ret:
        print("无法读取图像")
        break

    # 将图像编码为 JPEG 格式
    _, buffer = cv2.imencode('.jpg', frame)
    
    # 将编码后的字节流转换为 Base64 编码
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 打印 Base64 编码的图像
    print("Base64 编码的图像：", img_base64)

    # 在窗口中显示图像
    cv2.imshow("Camera", frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭所有打开的窗口
cv2.destroyAllWindows()
