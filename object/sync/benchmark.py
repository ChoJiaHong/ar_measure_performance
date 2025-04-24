import time
import os
import grpc
import json
import statistics
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from object_client import get_stub, send_request

def benchmark(n_requests=100, output_folder="./results/sync"):
    stub = get_stub()
    print(f"開始進行 {n_requests} 次請求...")

    sending_time = []
    recv_time = []
    inference_time = []

    start_total = time.time()
    success = 0

    for i in range(n_requests):
        try:
            send = time.time()
            sending_time.append(send)

            response = send_request(stub)

            recv = time.time()
            recv_time.append(recv)
            duration = recv - send
            inference_time.append(duration)

            #action = json.loads(response.action)
            #print(f"[{i}] Left: {action['Left']}, Right: {action['Right']}")
            success += 1
        except grpc.RpcError as e:
            recv = time.time()
            recv_time.append(recv)
            inference_time.append(0)
            print(f"[{i}] gRPC 錯誤：{e.code()} - {e.details()}")

    end_total = time.time()
    total_duration = end_total - start_total
    qps = success / total_duration if total_duration > 0 else 0

    # 統計
    valid_times = [t for t in inference_time if t > 0]
    min_val = min(valid_times) if valid_times else 0
    max_val = max(valid_times) if valid_times else 0
    avg_val = statistics.mean(valid_times) if valid_times else 0
    std_val = statistics.stdev(valid_times) if len(valid_times) > 1 else 0

    print(f"總耗時：{total_duration:.2f} 秒")
    print(f"平均每秒吞吐量（QPS）：{qps:.2f} 次請求/秒")

    # 準備輸出
    os.makedirs(output_folder, exist_ok=True)
    filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]

    # 1️⃣ 輸出 CSV：詳細請求紀錄
    csv_path = os.path.join(output_folder, f"record_{filename}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "sending_time", "recv_time", "inference_time"])
        for i in range(n_requests):
            writer.writerow([i, sending_time[i], recv_time[i], inference_time[i]])

    # 2️⃣ 輸出 TXT：總體摘要
    summary_path = os.path.join(output_folder, f"summary_{filename}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"requests         : {n_requests}\n")
        f.write(f"successes        : {success}\n")
        f.write(f"total_time       : {total_duration:.6f} s\n")
        f.write(f"average_time     : {(total_duration / n_requests):.6f} s\n")
        f.write(f"qps              : {qps:.2f}\n")
        f.write(f"min              : {min_val:.4f}\n")
        f.write(f"max              : {max_val:.4f}\n")
        f.write(f"avg              : {avg_val:.4f}\n")
        f.write(f"std              : {std_val:.4f}\n")

    # 3️⃣ 畫圖
    x = np.linspace(0, n_requests - 1, n_requests)
    plt.plot(x, inference_time, color='blue')
    plt.title("Sync Inference Time per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Inference Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"plot_{filename}.png"))
    plt.show()

if __name__ == '__main__':
    benchmark(n_requests=10000)
