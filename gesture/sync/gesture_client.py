# gesture_client.py
import grpc
import os
import sys

# 手動加 proto 資料夾進 Python 模組搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proto')))

import gesture_pb2
import gesture_pb2_grpc
import base64

service_ip = "172.22.9.141"
service_port = 30501
service_url = f"{service_ip}:{service_port}"
IMAGE_PATH='/home/hiro/git_repo/ar_measure_performance/gesture/1920x1080_hand.jpg'
# 載入圖片一次即可重複使用
with open(IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes)

# 建立 stub（讓 benchmark.py 可以重複使用）
def get_stub():
    channel = grpc.insecure_channel(service_url)
    stub = gesture_pb2_grpc.GestureRecognitionStub(channel)
    return stub

# 封裝 request 發送邏輯，供 benchmark 使用
def send_request(stub):
    request = gesture_pb2.RecognitionRequest(image=image_base64)
    response = stub.Recognition(request)
    return response

if __name__ == "__main__":
    stub = get_stub()
    response = send_request(stub)
    print(f"Frame index: {response.frame_index}")
    print(f"Timestamp: {response.timestamp}")
    print(f"Action: {response.action}")
