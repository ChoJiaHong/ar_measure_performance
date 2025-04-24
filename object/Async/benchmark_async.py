import asyncio
import time
import grpc
import statistics
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
import requests

from object_client_async import get_stub, send_request
from util.query_prometheus import query_metric_range


# -----------------------------
# 請求部分
# -----------------------------
async def run_single(i, stub):
    """
    發送單個 gRPC 請求並測量延遲。

    參數:
        i (int): 請求索引
        stub: gRPC stub 物件

    回傳:
        tuple: (sending_time, recv_time, inference_time)
    """
    try:
        send = time.time()
        await send_request(stub)
        recv = time.time()
        return (send, recv, recv - send)
    except grpc.RpcError as e:
        print(f"[{i}] gRPC 錯誤：{e.code()} - {e.details()}")
        now = time.time()
        return (now, now, 0)

async def benchmark_async(n_requests, concurrency):
    """
    非同步執行多次請求並收集每次請求的時間數據。

    參數:
        n_requests (int): 請求總數
        concurrency (int): 同時併發請求數上限

    回傳:
        list: 每個請求結果的 tuple 列表 (sending_time, recv_time, inference_time)
    """
    stub, channel = await get_stub()
    sem = asyncio.Semaphore(concurrency)

    async def limited_run(i):
        async with sem:
            return await run_single(i, stub)

    results = await asyncio.gather(*(limited_run(i) for i in range(n_requests)))
    await channel.close()
    return results

# -----------------------------
# 統計與數據處理部分
# -----------------------------
def compute_statistics(results):
    """
    從結果列表中計算統計數據。

    參數:
        results (list): 每個元素為 (sending_time, recv_time, inference_time) 的 tuple

    回傳:
        dict: 統計數據字典，包含：
              - sending_times, recv_times, inference_times (列表)
              - total_duration, success_count, min_val, max_val, avg_val, std_val, qps
    """
    sending_times = list(map(lambda r: r[0], results))
    recv_times = list(map(lambda r: r[1], results))
    inference_times = list(map(lambda r: r[2], results))
    total_duration = max(recv_times) - min(sending_times)
    valid_times = list(filter(lambda t: t > 0, inference_times))
    success_count = len(valid_times)
    min_val = min(valid_times) if valid_times else 0
    max_val = max(valid_times) if valid_times else 0
    avg_val = statistics.mean(valid_times) if valid_times else 0
    std_val = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
    qps = success_count / total_duration if total_duration > 0 else 0
    return {
        "sending_times": sending_times,
        "recv_times": recv_times,
        "inference_times": inference_times,
        "total_duration": total_duration,
        "success_count": success_count,
        "min_val": min_val,
        "max_val": max_val,
        "avg_val": avg_val,
        "std_val": std_val,
        "qps": qps
    }

# -----------------------------
# 輸出與繪圖部分
# -----------------------------
def output_csv(data, n_requests, output_folder, filename):
    """
    將請求結果輸出為 CSV 檔案。

    參數:
        data (dict): 包含 sending_times, recv_times, inference_times 的字典
        n_requests (int): 請求總數
        output_folder (str): 輸出目錄
        filename (str): 用於檔案命名的字串
    """
    csv_path = os.path.join(output_folder, f"record_{filename}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "sending_time", "recv_time", "inference_time"])
        for i, (s, r, inf) in enumerate(zip(data["sending_times"], data["recv_times"], data["inference_times"])):
            writer.writerow([i, s, r, inf])

def output_txt(data, n_requests, output_folder, filename):
    """
    將統計摘要輸出到 TXT 檔案。

    參數:
        data (dict): 統計數據字典
        n_requests (int): 請求總數
        output_folder (str): 輸出目錄
        filename (str): 用於檔案命名的字串
    """
    summary_path = os.path.join(output_folder, f"summary_{filename}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"requests         : {n_requests}\n")
        f.write(f"successes        : {data['success_count']}\n")
        f.write(f"total_time       : {data['total_duration']:.6f} s\n")
        f.write(f"average_time     : {(data['total_duration'] / n_requests):.6f} s\n")
        f.write(f"qps              : {data['qps']:.2f}\n")
        f.write(f"min              : {data['min_val']:.4f}\n")
        f.write(f"max              : {data['max_val']:.4f}\n")
        f.write(f"avg              : {data['avg_val']:.4f}\n")
        f.write(f"std              : {data['std_val']:.4f}\n")

def plot_results(data, n_requests, output_folder, filename):
    """
    使用 matplotlib 繪製延遲數據折線圖，並儲存圖片。

    參數:
        data (dict): 包含 inference_times 的字典
        n_requests (int): 請求總數
        output_folder (str): 輸出目錄
        filename (str): 用於檔案命名的字串
    """
    x = np.linspace(0, n_requests - 1, n_requests)
    plt.plot(x, data["inference_times"], color='red')
    plt.title("Async Inference Time per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Inference Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"plot_{filename}.png")
    plt.savefig(plot_path)
    plt.show()

# -----------------------------
# Prometheus 數據查詢部分
# -----------------------------
def query_prometheus_data(prometheus_url, metric, start, end, step):
    """
    使用 query_metric_range 從 Prometheus 查詢指定指標在一個時間區間內的數據。

    參數:
        prometheus_url (str): Prometheus 服務的 URL
        metric (str): 查詢的 Prometheus 指標，例如 "DCGM_FI_DEV_GPU_UTIL"
        start (int): 起始 Unix 時間戳（秒）
        end (int): 結束 Unix 時間戳（秒）
        step (str): 查詢間隔（秒）

    回傳:
        list: 每個數據點為 [timestamp, value] 的列表
    """
    return query_metric_range(metric, start, end, step, prometheus_url)

def output_prometheus_data(ts_data, metric_name):
    """
    將 Prometheus 返回的時間序列數據格式化輸出。

    參數:
        ts_data (list): 每個元素為 [timestamp, value] 的列表
        metric_name (str): 指標名稱，用於輸出標題
    """
    for ts, value in ts_data:
        dt = datetime.datetime.fromtimestamp(float(ts))
        print(f"{metric_name} at {dt}: {value}")

# -----------------------------
# 主流程
# -----------------------------
async def main(n_requests=1000, concurrency=10, output_folder="./results/async",
               prometheus_url="http://your_prometheus_server:9090"):
    """
    主函式：執行非同步基準測試、計算統計數據、輸出結果、並查詢 Prometheus 獲取指定時間區間內的 GPU 指標數據。

    參數:
        n_requests (int, optional): 請求總數，預設為 1000
        concurrency (int, optional): 併發請求數上限，預設為 10
        output_folder (str, optional): 結果輸出目錄，預設 "./results/async"
        prometheus_url (str, optional): Prometheus 服務的 URL，請根據你的環境修改

    回傳:
        None
    """
    # 執行基準測試並收集結果（返回結果列表）
    results = await benchmark_async(n_requests, concurrency)
    # 計算統計數據（純函式）
    stats = compute_statistics(results)
    # 生成唯一檔案名稱
    filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
    os.makedirs(output_folder, exist_ok=True)
    # 輸出 CSV 與 TXT 及繪圖（副作用函式）
    output_csv(stats, n_requests, output_folder, filename)
    output_txt(stats, n_requests, output_folder, filename)
    plot_results(stats, n_requests, output_folder, filename)

    # 查詢 Prometheus 數據：定義查詢區間（例如過去 1 小時）
    end_time = int(time.time())
    start_time = end_time - 3600  # 一小時前
    step = "30"  # 每 30 秒一個數據點

    gpu_util_ts = query_prometheus_data(prometheus_url, "DCGM_FI_DEV_GPU_UTIL", start_time, end_time, step)
    sm_clock_ts = query_prometheus_data(prometheus_url, "DCGM_FI_DEV_SM_CLOCK", start_time, end_time, step)

    print("\nPrometheus GPU Utilization Time Series:")
    output_prometheus_data(gpu_util_ts, "GPU Utilization")

    print("\nPrometheus SM Clock Time Series:")
    output_prometheus_data(sm_clock_ts, "SM Clock")

if __name__ == '__main__':
    asyncio.run(main(n_requests=1000, concurrency=10, output_folder="./results/async",
                      prometheus_url="http://172.22.9.249:9090"))
