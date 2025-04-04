import grpc
import os
import sys

# 手動加 proto 資料夾進 Python 模組搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proto')))
import inference_pb2
import inference_pb2_grpc
import json
import struct

# gRPC server 位址
service_ip = "172.22.9.227"
service_port = 30500
service_url = f"{service_ip}:{service_port}"
IMAGE_URL = "/home/hiro/git_repo/ar_measure_performance/1600x900_hand.jpg"
# 讀取圖片並加上模擬的 index
with open(IMAGE_URL, "rb") as f:
    image_bytes = bytearray(f.read())


# 建立 PredictionsRequest，需轉為 bytes
req = inference_pb2.PredictionsRequest(
    model_name="models-1",
    input={"data": bytes(image_bytes)} 
)

async def get_stub():
    # 建立 gRPC channel 和 stub
    channel = grpc.aio.insecure_channel(service_url)
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub, channel

async def send_request(stub):
    response = await stub.Predictions(req)
    return response


if __name__ == "__main__":
    stub = get_stub()
    try:
        # 測試呼叫
        result = send_request(stub)
        print("Object detection result:\n", result)
    except grpc.RpcError as e:
        print(f"gRPC 錯誤：{e.code()} - {e.details()}")