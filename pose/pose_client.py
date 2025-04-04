import grpc
import pose_pb2
import pose_pb2_grpc
import json
import struct

# gRPC server 位址
service_ip = "172.22.9.227"
service_port = 30500
service_url = f"{service_ip}:{service_port}"

# 讀取圖片並加上模擬的 index
with open("/home/hiro/git_repo/ar_measure_performance/object/1280hand.jpg", "rb") as f:
    image_bytes = bytearray(f.read())
# 建立 gRPC channel 和 stub
channel = grpc.insecure_channel(service_url)
stub = pose_pb2_grpc.InferenceAPIsServiceStub(channel)

# 建立 PredictionsRequest，需轉為 bytes
req = pose_pb2.PredictionsRequest(
    model_name="models-1",
    input={"data": bytes(image_bytes)} 
)

def get_stub():
    return stub

def send_request(stub):
    response = stub.Predictions(req)
    return response


if __name__ == "__main__":
    stub = get_stub()
    try:
        # 測試呼叫
        result = send_request(stub)
        print("Object detection result:\n", result)
    except grpc.RpcError as e:
        print(f"gRPC 錯誤：{e.code()} - {e.details()}")