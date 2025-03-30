import os
import datetime
import json
import statistics
import sys
import threading
import time
import grpc
import numpy as np
import matplotlib.pyplot as plt
import inference_pb2
import inference_pb2_grpc


#python AR_emulator_toservice.py serviceIP servicePort
#python AR_emulator_toservice.py 10.52.52.25 30602
ServiceIP = sys.argv[1]
ServicePort = sys.argv[2]
freq = int(sys.argv[3])
output_folder = sys.argv[4]

#read image to byte array
with open("1280hand.jpg", "rb") as image:
    image_byte = image.read()

Object_detection_channel = grpc.insecure_channel(ServiceIP + ":" + str(ServicePort))
Object_detection_stub = inference_pb2_grpc.InferenceAPIsServiceStub(Object_detection_channel)

request = inference_pb2.PredictionsRequest(
            model_name='models-1',
            input={'data': image_byte}
        )

sending_time = []
recv_time = []
inference_time = []

def objdet():
    send = time.time()
    sending_time.append(send)
    response = Object_detection_stub.Predictions(request)
    recv = time.time()
    recv_time.append(recv)
    inference_time.append(recv - send)
    result = response.prediction.decode('utf-8')
    result = json.loads(result)
    retstr = ""
    for i in range(0, 5):
        retstr = retstr + result[0]['parts'][i] + " "
    print(retstr)

request_num = 1000

for i in range(request_num):
    tt = time.time()

    t = threading.Thread(target=objdet)
    t.daemon = True
    t.start()

    sleeptime = 1 / freq - time.time() + tt
    if sleeptime > 0:
        time.sleep(sleeptime)

print('inference_time')
print(f'min = {min(inference_time)}')
print(f'max = {max(inference_time)}')
print(f'avg = {statistics.mean(inference_time)}')
print(f'std = {statistics.stdev(inference_time)}')

filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
output_path = os.path.join(output_folder, f"record_{filename}.txt")

with open(output_path , 'w') as f:
    for i in range(request_num):
        f.write(f"{sending_time[i]} | {recv_time[i]} | {inference_time[i]}\n")
    f.write(f"first send : {sending_time[0]}, last receive : {recv_time[-1]}\n")
    f.write(f"total time : {recv_time[-1] - sending_time[0]}\n")
    f.write(f"average : {(recv_time[-1] - sending_time[0]) / request_num}\n")

x = np.linspace(0, request_num, request_num)
plt.plot(x, inference_time, 'r')
plt.show()