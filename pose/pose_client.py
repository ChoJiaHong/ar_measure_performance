import grpc
import pose_pb2
import pose_pb2_grpc
import io
from PIL import Image
# gRPC server 位址

# 讀取圖片（不加 index）
image_path = "1280.jpg"
image = Image.open(image_path)
# 建立 BytesIO 物件並以 JPEG 格式儲存圖片
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
img_data = buffer.getvalue()
# 建立 FrameRequest（直接送圖片 bytes）


def get_stub(service_ip="172.22.9.141", service_port = 30562):
    service_url = f"{service_ip}:{service_port}"
    channel = grpc.insecure_channel(service_url)
    stub = pose_pb2_grpc.MirrorStub(channel)
    return stub

def send_request(stub):
    request = pose_pb2.FrameRequest(image_data=img_data)
    response = stub.SkeletonFrame(request)
    return response

if __name__ == "__main__":
    stub = get_stub()
    try:
        result = send_request(stub)
        print("Pose detection result:\n", result)
    except grpc.RpcError as e:
        print(f"gRPC 錯誤：{e.code()} - {e.details()}")
