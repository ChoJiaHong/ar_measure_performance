import asyncio
import time
import grpc
import json
import statistics
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv

from gesture_client_async import get_stub, send_request

async def run_single(i, stub, sending_time, recv_time, inference_time):
    try:
        send = time.time()
        sending_time[i] = send

        response = await send_request(stub)

        recv = time.time()
        recv_time[i] = recv
        inference_time[i] = recv - send

        action = json.loads(response.action)
        print(f"[{i}] Left: {action['Left']}, Right: {action['Right']}")
    except grpc.RpcError as e:
        recv_time[i] = sending_time[i]
        inference_time[i] = 0
        print(f"[{i}] gRPC 錯誤：{e.code()} - {e.details()}")

async def benchmark_async(n_requests=100, concurrency=10, output_folder="./results/async"):
    stub, channel = await get_stub()

    sending_time = [0] * n_requests
    recv_time = [0] * n_requests
    inference_time = [0] * n_requests

    print(f"開始以併發 {concurrency} 模擬 {n_requests} 次請求...")

    start = time.time()

    sem = asyncio.Semaphore(concurrency)

    async def limited_run(i):
        async with sem:
            await run_single(i, stub, sending_time, recv_time, inference_time)

    await asyncio.gather(*(limited_run(i) for i in range(n_requests)))

    end = time.time()
    total_duration = end - start

    await channel.close()

    # 統計
    valid_times = [t for t in inference_time if t > 0]
    success_count = len(valid_times)

    min_val = min(valid_times) if valid_times else 0
    max_val = max(valid_times) if valid_times else 0
    avg_val = statistics.mean(valid_times) if valid_times else 0
    std_val = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
    qps = success_count / total_duration if total_duration > 0 else 0

    # 輸出檔案
    os.makedirs(output_folder, exist_ok=True)
    filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]

    # 1️⃣ 輸出 CSV 詳細資料
    csv_path = os.path.join(output_folder, f"record_{filename}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "sending_time", "recv_time", "inference_time"])
        for i in range(n_requests):
            writer.writerow([i, sending_time[i], recv_time[i], inference_time[i]])

    # 2️⃣ 輸出 TXT 統計摘要
    summary_path = os.path.join(output_folder, f"summary_{filename}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"requests         : {n_requests}\n")
        f.write(f"successes        : {success_count}\n")
        f.write(f"total_time       : {total_duration:.6f} s\n")
        f.write(f"average_time     : {(total_duration / n_requests):.6f} s\n")
        f.write(f"qps              : {qps:.2f}\n")
        f.write(f"min              : {min_val:.4f}\n")
        f.write(f"max              : {max_val:.4f}\n")
        f.write(f"avg              : {avg_val:.4f}\n")
        f.write(f"std              : {std_val:.4f}\n")

    # 3️⃣ 繪圖
    x = np.linspace(0, n_requests - 1, n_requests)
    plt.plot(x, inference_time, color='red')
    plt.title("Async Inference Time per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Inference Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"plot_{filename}.png"))
    plt.show()

    # 4️⃣ 顯示統計結果
    print("\n=== 統計資訊 ===")
    print(f"requests         : {n_requests}")
    print(f"successes        : {success_count}")
    print(f"total_time       : {total_duration:.6f} s")
    print(f"average_time     : {(total_duration / n_requests):.6f} s")
    print(f"qps              : {qps:.2f}")
    print(f"min              : {min_val:.4f}")
    print(f"max              : {max_val:.4f}")
    print(f"avg              : {avg_val:.4f}")
    print(f"std              : {std_val:.4f}")

if __name__ == '__main__':
    asyncio.run(benchmark_async(n_requests=100, concurrency=10, output_folder="./results/async"))
