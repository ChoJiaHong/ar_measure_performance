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

from gesture.Async.gesture_client_async import get_stub, send_request

from util.prometheus_client import PrometheusClient
from zoneinfo import ZoneInfo  # Python 3.9+
# 取得台灣時間（Asia/Taipei）
now_tw = datetime.datetime.now(ZoneInfo("Asia/Taipei"))

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

async def benchmark_async_duration(test_duration_seconds, concurrency):
    print(f"Starting benchmark for {test_duration_seconds} seconds with concurrency={concurrency}")
    stub, channel = await get_stub()
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def limited_loop(i):
        async with sem:
            send = time.time()
            try:
                print(f"[{i}] Sending request")
                await send_request(stub)
                recv = time.time()
                print(f"[{i}] Received response in {recv - send:.4f}s")
                return (send, recv, recv - send)
            except grpc.RpcError as e:
                now = time.time()
                print(f"[{i}] gRPC error: {e.code()} - {e.details()}")
                return (now, now, 0)

    start_time = time.time()
    task_id = 0
    tasks = []

    while time.time() - start_time < test_duration_seconds:
        print(f"[{task_id}] Scheduling new request")
        task = asyncio.create_task(limited_loop(task_id))
        tasks.append(task)
        task_id += 1
        await asyncio.sleep(0.001)

    print(f"All tasks scheduled, awaiting completion of {len(tasks)} tasks")
    results = await asyncio.gather(*tasks)
    print(f"Benchmark complete. Total requests sent: {len(results)}")
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
              - sending_times (list of float): 發送時間戳列表
              - recv_times (list of float): 接收時間戳列表
              - inference_times (list of float): 延遲列表
              - total_duration (float): 總耗時 (最後回應時間 - 最早發送時間)
              - success_count (int): 延遲 > 0 的請求數
              - success_rate (float): 成功率 = success_count / 總請求數
              - min_val (float): 有效延遲最小值
              - max_val (float): 有效延遲最大值
              - avg_val (float): 有效延遲平均值
              - std_val (float): 有效延遲標準差
              - qps (float): 每秒成功請求數 = success_count / total_duration
    """
    sending_times = [r[0] for r in results]
    recv_times = [r[1] for r in results]
    inference_times = [r[2] for r in results]
    total_duration = max(recv_times) - min(sending_times) if results else 0
    valid_times = [t for t in inference_times if t > 0]
    success_count = len(valid_times)
    total_requests = len(results)
    success_rate = success_count / total_requests if total_requests > 0 else 0.0
    min_val = min(valid_times) if valid_times else 0.0
    max_val = max(valid_times) if valid_times else 0.0
    avg_val = statistics.mean(valid_times) if valid_times else 0.0
    std_val = statistics.stdev(valid_times) if len(valid_times) > 1 else 0.0
    qps = success_count / total_duration if total_duration > 0 else 0.0

    return {
        "sending_times": sending_times,
        "recv_times": recv_times,
        "inference_times": inference_times,
        "total_duration": total_duration,
        "success_count": success_count,
        "success_rate": success_rate,
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


def plot_results(x, y, title, xlabel, ylabel, folder, prefix,
                fmt='-o', grid=True, rotate_xticks=False, figsize=(10,5)):
    """
    通用折線圖函式：繪製 X vs Y，並將圖表儲存為 PNG。

    參數:
        x (Sequence): X 軸數值列表（索引或時間點）。
        y (Sequence): Y 軸數值列表，與 x 長度相同。
        title (str): 圖表標題。
        xlabel (str): X 軸標籤文字。
        ylabel (str): Y 軸標籤文字。
        folder (str): 圖檔輸出目錄。若不存在將自動建立。
        prefix (str): 檔名開頭，用於區分不同圖表。
        fmt (str, optional): Matplotlib 繪圖格式字串，預設 '-o'（線 + 圓點）。
        grid (bool, optional): 是否顯示網格線，預設 True。
        rotate_xticks (bool, optional): 是否將 X 軸刻度標籤旋轉 45 度並右對齊，預設 False。
        figsize (tuple, optional): 圖片尺寸 (width, height)，單位英吋，預設 (10, 5)。

    回傳:
        str: 儲存後的圖檔完整路徑。
    """
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, fmt)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid: ax.grid(True)
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(folder, f"{prefix}_{title.replace(' ','_')}_.png")
    fig.savefig(path); plt.close(fig)
    return path

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
    prometheusClient = PrometheusClient(prometheus_url)
    start_time= prometheusClient.get_current_time()
    
    # 執行基準測試並收集結果（返回結果列表）
    results = await benchmark_async_duration(n_requests, concurrency)
    print(f"跑完{n_requests}請求")
    # 計算統計數據（純函式）
    stats = compute_statistics(results)
    # 生成唯一檔案名稱
    filename = now_tw.strftime("%m%d_%H:%M")
    #output_folder再加一個folder名字為filename
    
    output_folder=f"{output_folder}/{filename}_concurrency:{concurrency}"
    os.makedirs(output_folder, exist_ok=True)
    # 輸出 CSV 與 TXT 及繪圖（副作用函式）
    output_csv(stats, n_requests, output_folder, filename)
    output_txt(stats, n_requests, output_folder, filename)
    plot_results(x=range(len(stats["inference_times"])),
                 y=stats["inference_times"],
                 title="Inference Latency",
                 xlabel="Request Index",
                 ylabel="Latency (s)",
                 folder=output_folder,
                 prefix=f"latency_{filename}")

    # 查詢 Prometheus 數據：定義查詢區間（例如過去 1 小時）
    end_time = prometheusClient.get_current_time()+40
    
    print(f"等待30秒，觀察硬體資源")
    time.sleep(40)
    
    step = "1"  # 每 1 秒一個數據點
    gpu_util_query='DCGM_FI_DEV_GPU_UTIL{job="prometheus/dcgm-exporter", instance="10.244.3.95:9400", gpu="0"}'
    gpu_util_ts = prometheusClient.query_range( gpu_util_query, start_time, end_time, step)
    sm_clock_ts = prometheusClient.query_range( "DCGM_FI_DEV_SM_CLOCK", start_time, end_time, step)

    print("\nPrometheus GPU Utilization Time Series:")
    if gpu_util_ts:
        timestamps, values = zip(*gpu_util_ts)
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="GPU Utilization",
                 xlabel="Timestamp",
                 ylabel="Utilization",
                 folder=output_folder,
                 prefix="gpu_util")
    else:
        print(f"no gpu_util_ts metric")
        

    print("\nPrometheus SM Clock Time Series:")
    if sm_clock_ts:
        timestamps, values = zip(*sm_clock_ts)
        plot_results(x=timestamps,
                 y=values,
                 title="SM Clock",
                 xlabel="Timestamp",
                 ylabel="SM Clock",
                 folder=output_folder,
                 prefix="sm_clock")
    else:
        print(f"no SM Clock Time Series metric")
    

if __name__ == '__main__':
    asyncio.run(main(n_requests=10, concurrency=30, output_folder="./results/async/gesture",
                      prometheus_url="http://172.22.9.249:30000"))
