import grpc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proto')))

import pose_pb2
import pose_pb2_grpc
import io
from PIL import Image
# gRPC server 位址

# 讀取圖片（不加 index）
image_path = "/home/hiro/git_repo/ar_measure_performance/pose/1280.jpg"
image = Image.open(image_path)
# 建立 BytesIO 物件並以 JPEG 格式儲存圖片
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
img_data = buffer.getvalue()
# 建立 FrameRequest（直接送圖片 bytes）


async def get_stub(service_url="172.22.9.141:30562"):
    channel = grpc.aio.insecure_channel(service_url)
    stub = pose_pb2_grpc.MirrorStub(channel)
    return stub, channel

async def send_request(stub):
    request = pose_pb2.FrameRequest(image_data=img_data)
    response = await stub.SkeletonFrame(request)
    return response

if __name__ == "__main__":
    stub = get_stub()
    try:
        result = send_request(stub)
        print("Pose detection result:\n", result)
    except grpc.RpcError as e:
        print(f"gRPC 錯誤：{e.code()} - {e.details()}")
