# gesture_client_async.py

import grpc
import sys
import os

# 手動加 proto 資料夾進 Python 模組搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proto')))
import gesture_pb2
import gesture_pb2_grpc
import base64

# server 設定
service_ip = "172.22.9.141"
service_port = 30501
service_url = f"{service_ip}:{service_port}"

# 預讀圖片
image_path = "/home/hiro/git_repo/ar_measure_performance/gesture/1280hand.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes)

# 建立非同步 stub
async def get_stub():
    channel = grpc.aio.insecure_channel(service_url)
    stub = gesture_pb2_grpc.GestureRecognitionStub(channel)
    return stub, channel

# 非同步發送 request
async def send_request(stub):
    request = gesture_pb2.RecognitionRequest(image=image_base64)
    response = await stub.Recognition(request)
    return response
