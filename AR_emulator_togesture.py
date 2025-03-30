import datetime
import json
import statistics
import sys
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import grpc
import gesture_recognition_pb2
import gesture_recognition_pb2_grpc

#python AR_emulator_togesture.py serviceIP servicePort freqency
ServiceIP = sys.argv[1]
ServicePort = sys.argv[2]
freq = int(sys.argv[3])
output_folder = sys.argv[4]

#read image to byte array
with open("1280hand.jpg", "rb") as image:
    image_byte = image.read()

Service_channel = grpc.insecure_channel(ServiceIP + ":" + str(ServicePort))
Service_stub = gesture_recognition_pb2_grpc.GestureRecognitionStub(Service_channel)

request = gesture_recognition_pb2.RecognitionRequest(image = image_byte)

request_num = 1000

sending_time = []
recv_time = []
inference_time = []

def detection():
    send = time.time()
    sending_time.append(send)
    response = Service_stub.Recognition(request)
    recv = time.time()
    recv_time.append(recv)
    inference_time.append(recv - send)
    actiondata = json.loads(response.action)
    print(f"Left: {actiondata['Left']}, Right: {actiondata['Right']}")

for i in range(request_num):
    starttime = time.time()

    t = threading.Thread(target=detection)
    t.daemon = True
    t.start()

    sleeptime = 1 / freq - time.time() + starttime
    if sleeptime > 0:
        time.sleep(sleeptime)

print('inference_time')
print(f'min = {min(inference_time)}')
print(f'max = {max(inference_time)}')
print(f'avg = {statistics.mean(inference_time)}')
print(f'std = {statistics.stdev(inference_time)}')

filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
output_path = os.path.join(output_folder, f"record_{filename}.txt")

with open(output_path, 'w') as f:
    for i in range(request_num):
        f.write(f"{sending_time[i]} | {recv_time[i]} | {inference_time[i]}\n")
    f.write(f"first send : {sending_time[0]}, last receive : {recv_time[-1]}\n")
    f.write(f"total time : {recv_time[-1] - sending_time[0]}\n")
    f.write(f"average : {(recv_time[-1] - sending_time[0]) / request_num}\n")

x = np.linspace(0, request_num, request_num)
plt.plot(x, inference_time, 'r')
plt.show()
