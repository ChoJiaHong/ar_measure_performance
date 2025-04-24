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

from pose.Async.pose_client_async import get_stub, send_request

from util.prometheus_client import PrometheusClient
import util.output
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
async def benchmark_async_duration_ver(test_duration_seconds, concurrency, service_url):
    stub, channel = await get_stub(service_url=service_url)
    sem = asyncio.Semaphore(concurrency)
    results = []

    start_time = time.time()

    async def limited_loop(i):
        async with sem:
            send = time.time()
            try:
                await send_request(stub)
                recv = time.time()
                duration = recv - send

                # ✅ 只保留在時間限制內完成的請求
                if recv - start_time <= test_duration_seconds:
                    return (send, recv, duration)
                else:
                    return None
            except grpc.RpcError as e:
                now = time.time()
                return None

    task_id = 0
    tasks = []
    
    while time.time() - start_time < test_duration_seconds:
        task = asyncio.create_task(limited_loop(task_id))
        tasks.append(task)
        task_id += 1
        #  # 每 1ms 
        await asyncio.sleep(0.001)

    print(f"taskID{task_id}")
    all_results = await asyncio.gather(*tasks)

    # ✅ 過濾掉 None（未在時間內完成或錯誤的）
    filtered_results = [r for r in all_results if r is not None]

    await channel.close()
    return filtered_results
async def benchmark_async(n_requests, concurrency, service_url):
    """
    非同步執行多次請求並收集每次請求的時間數據。

    參數:
        n_requests (int): 請求總數
        concurrency (int): 同時併發請求數上限

    回傳:
        list: 每個請求結果的 tuple 列表 (sending_time, recv_time, inference_time)
    """
    stub, channel = await get_stub(service_url=service_url)
    sem = asyncio.Semaphore(concurrency)

    async def limited_run(i):
        async with sem:
            return await run_single(i, stub)

    results = await asyncio.gather(*(limited_run(i) for i in range(n_requests)))
    await channel.close()
    return results

async def benchmark_async_duration(test_duration_seconds, concurrency,service_url):
    print(f"Starting benchmark for {test_duration_seconds} seconds with concurrency={concurrency}")
    stub, channel = await get_stub(service_url=service_url)
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def limited_loop(i):
        async with sem:
            send = round(time.time(), 4)
            try:
                print(f"[{i}] Sending request")
                await send_request(stub)
                recv = round(time.time(), 4)  # 單位是秒 (seconds)，縮減到小數點後四位
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
                fmt='-o', grid=True, rotate_xticks=False, figsize=(10,5),
                show_values=False, fixed_ylim=None):
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
    #ax.plot(x, y, fmt)
    ax.plot(x, y, fmt[0], linewidth=1)
    

    # 標題與座標軸
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 固定 Y 軸範圍（如果有指定）
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)
    elif y:
        ymin, ymax = min(y), max(y)
        ax.set_ylim(ymin - 1, ymax + 1)
    # 顯示每個資料點的數值
    if show_values:
        previous_y = None
        for xi, yi in zip(x, y):
            current_y = int(float(yi))
            if previous_y is None or current_y != previous_y:
                ax.annotate(
                    str(current_y),
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha='center',
                    fontsize=8,
                    
                )
            previous_y = current_y
    if grid:
        ax.grid(True)
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(folder, f"{prefix}_{title.replace(' ','_')}_.png")
    fig.savefig(path)
    plt.close(fig)
    return path

def plot_results_gpu(x, y, title, xlabel, ylabel, folder, prefix,
                fmt='-o', grid=True, rotate_xticks=False, figsize=(10,5), fix_ylim=False):
    """
    繪製 X vs Y 並儲存圖表為 PNG。
    新增參數 fix_ylim，若為 True，則將 Y 軸固定為 0~100。
    """
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, fmt)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if fix_ylim:
        ax.set_ylim(0, 100)  # << 固定 Y 軸範圍
    if grid:
        ax.grid(True)
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(folder, f"{prefix}_{title.replace(' ','_')}_.png")
    fig.savefig(path)
    plt.close(fig)
    return path

def save_timeseries_to_csv(timeseries, output_file):
    """
    timeseries: List of [timestamp, value]
    output_file: CSV 檔案名稱
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_iso", "timestamp_unix", "value"])  # 標題列

        for ts, val in timeseries:
            ts_iso = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow([ts_iso, ts, val])

    print(f"✅ 已儲存時序資料到 {output_file}")
# -----------------------------
# 主流程

        

    
# -----------------------------
async def main(n_requests=1000, concurrency=10, output_folder="./results/async",
               prometheus_url="http://your_prometheus_server:9090",wait_time=30):
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
    results = await benchmark_async_duration(n_requests, concurrency,"172.22.9.141:30562")
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
                 prefix=f"latency_{filename}",
                 fixed_ylim=(0,2))

    # 查詢 Prometheus 數據：定義查詢區間（例如過去 1 小時）

    end_time = prometheusClient.get_current_time()+wait_time
    
    print(f"等待{wait_time}秒，觀察硬體資源")
    time.sleep(wait_time)
    
    step = "2s"  # 每 1 秒一個數據點
    gpu_util_query='DCGM_FI_DEV_GPU_UTIL{job="prometheus/dcgm-exporter", instance="10.244.2.108:9400", gpu="0"}'
    sm_clock_query='DCGM_FI_DEV_SM_CLOCK{job="prometheus/dcgm-exporter", instance="10.244.2.108:9400", gpu="0"}'
    gpu_util_ts = prometheusClient.query_range( gpu_util_query, start_time, end_time, step)
    print(f"Prometheus GPU Utilization Time Series:{gpu_util_ts}")
    sm_clock_ts = prometheusClient.query_range( sm_clock_query, start_time, end_time, step)

    print("\nPrometheus GPU Utilization Time Series:")
    if gpu_util_ts:
        timestamps, values = zip(*gpu_util_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="GPU Utilization",
                 xlabel="Timestamp",
                 ylabel="Utilization",
                 folder=output_folder,
                 prefix="gpu_util",
                 show_values=True,
                 fixed_ylim=(0, 100)
                 )
    else:
        print(f"no gpu_util_ts metric")
        

    print("\nPrometheus SM Clock Time Series:")
    print(f"SM Clock Time Series: {sm_clock_ts}")
    if sm_clock_ts:
        timestamps, values = zip(*sm_clock_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="SM Clock",
                 xlabel="Timestamp",
                 ylabel="MHz",
                 folder=output_folder,
                 prefix="sm_clock",
                 show_values=True,
                 fixed_ylim=(0, 3000)
                 )
    else:
        print(f"no SM Clock Time Series metric")
    
    
    gpu_mem_query='DCGM_FI_DEV_FB_USED{job="prometheus/dcgm-exporter", instance="10.244.2.108:9400", gpu="0"}'
    gpu_mem_ts = prometheusClient.query_range( gpu_mem_query, start_time, end_time, step)
    print("\nPrometheus SM Clock Time Series:")
    if sm_clock_ts:
        timestamps, values = zip(*gpu_mem_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="gpu_mem",
                 xlabel="Timestamp",
                 ylabel="mib",
                 folder=output_folder,
                 prefix="gpu_mem",
                 show_values=True)
    else:
        print(f"no SM Clock Time Series metric")
        
    power_usage_query='DCGM_FI_DEV_POWER_USAGE{job="prometheus/dcgm-exporter", instance="10.244.2.108:9400", gpu="0"}'
    power_usage_ts = prometheusClient.query_range( power_usage_query, start_time, end_time, step)
    print("\nPrometheus SM Clock Time Series:")
    if sm_clock_ts:
        timestamps, values = zip(*power_usage_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="power_usage",
                 xlabel="Timestamp",
                 ylabel="W",
                 folder=output_folder,
                 prefix="power_usage",
                 show_values=True,
                 fixed_ylim=(0, 500)
                 )
    else:
        print(f"no SM Clock Time Series metric")
    
    GPU_TEMP_query='DCGM_FI_DEV_GPU_TEMP{job="prometheus/dcgm-exporter", instance="10.244.2.108:9400", gpu="0"}'
    GPU_TEMP_ts = prometheusClient.query_range( GPU_TEMP_query, start_time, end_time, step)
    print("\nPrometheus SM Clock Time Series:")
    if sm_clock_ts:
        timestamps, values = zip(*GPU_TEMP_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="GPU_TEMP",
                 xlabel="Timestamp",
                 ylabel="°C",
                 folder=output_folder,
                 prefix="GPU_TEMP",
                 show_values=True,
                 fixed_ylim=(0,80)
                 )
    else:
        print(f"no SM Clock Time Series metric")
        
    
async def hiro_duration(execute_time=1000, concurrency=10, output_folder="./results/async",
               prometheus_url="http://your_prometheus_server:9090",wait_time=20,service_urls=""):
    prometheusClient = PrometheusClient(prometheus_url)
    start_time= prometheusClient.get_current_time()-1
    #avg_concurrency=concurrency/len(service_urls)
    avg_concurrency=concurrency
    results = []
    tasks = []

    for service_url in service_urls:
        print(f"service_url: {service_url}")
        task = asyncio.create_task(benchmark_async_duration_ver(execute_time, avg_concurrency, service_url))
        tasks.append(task)

    # 等所有 benchmark_async_duration 完成
    
    all_results = await asyncio.gather(*tasks)
    
    # 合併所有結果（因為每個 result 是 list）
    for res in all_results:
        results.extend(res)    
    time.sleep(wait_time)
    total_time = prometheusClient.get_current_time() - start_time
    end_time=start_time+execute_time+wait_time
    step="1"
    #region gpu_util
    gpu_util_query='DCGM_FI_DEV_GPU_UTIL{job="prometheus/dcgm-exporter", instance="10.244.2.108:9400", gpu="0"}'
    gpu_util_ts = prometheusClient.query_range( gpu_util_query, start_time, end_time, step)
    print(f"Prometheus GPU Utilization Time Series:{gpu_util_ts}")
    if gpu_util_ts:
        timestamps, values = zip(*gpu_util_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        plot_results(x=time_labels,
                 y=values,
                 title="GPU Utilization",
                 xlabel="Timestamp",
                 ylabel="Utilization",
                 folder=output_folder,
                 prefix="gpu_util",
                 show_values=True,
                 fixed_ylim=(0, 100)
                 )
    else:
        print(f"no gpu_util_ts metric")
    #endregion
    #region cpu_util
    cpu_util_query = '100 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100'
    cpu_util_ts = prometheusClient.query_range(cpu_util_query, start_time, end_time, step)

    if cpu_util_ts:
        timestamps, values = zip(*cpu_util_ts)
        values = list(map(float, values))
        time_labels = [now_tw.fromtimestamp(float(ts)) for ts in timestamps]
        
        plot_results(
            x=time_labels,
            y=values,
            title="CPU Usage (%)",
            xlabel="Timestamp",
            ylabel="Usage (%)",
            folder=output_folder,
            prefix="cpu_util",
            show_values=True,
            fixed_ylim=(0, 100)
        )
    else:
        print("No CPU utilization metric found.")
    #endregion
    
    
    
    
    
    print(f"整個程式花了:{total_time}")
    print(f"start_time:{start_time}")
    print(f"總共花了{execute_time}秒")
    print(f"跑完{len(results)}請求")
    #print最後一個值
    print(f"最後一個值: {results[-1]}")
    print(f"吞吐量: {len(results)/execute_time} QPS")
    print(f"總體平均延遲: {execute_time/len(results)} 秒")
    print(f"單位平均延遲: {statistics.mean([r[2] for r in results])} 秒")
    print(f"最大延遲: {max([r[2] for r in results])} 秒")
    print(f"Throughput/服務數量: {len(results)/execute_time/len(service_urls)} QPS")
async def hiro(n_requests=1000, concurrency=10, output_folder="./results/async",
               prometheus_url="http://your_prometheus_server:9090",wait_time=20,service_urls=""):
    
    #avg_concurrency=int(concurrency/len(service_urls))
    avg_concurrency=concurrency
    print(f"avg_concurrency: {avg_concurrency}")
    avg_requests=int(n_requests/len(service_urls))
    print(f"avg_requests: {avg_requests}")
    results = []
    tasks = []
    real_start_time=time.time()
    for service_url in service_urls:
        print(f"service_url: {service_url}")
        task = asyncio.create_task(benchmark_async(avg_requests, avg_concurrency, service_url))
        tasks.append(task)

    print(f"taskID{len(tasks)}")
    # 等所有 benchmark_async_duration 完成
    all_results = await asyncio.gather(*tasks)
    # 合併所有結果（因為每個 result 是 list）
    for res in all_results:
        results.extend(res)    
    
    real_end_time=time.time()
    
    total_duration = real_end_time - real_start_time
  
  
    
    
   
    
    
    
    
    
    print(f"總共花了{total_duration}秒")
    print(f"跑完{len(results)}請求")
    #print最後一個值
    print(f"最後一個值: {results[-1]}")
    print(f"吞吐量: {len(results)/total_duration} QPS")
    print(f"總體平均延遲: {total_duration/len(results)} 秒")
    print(f"單位平均延遲: {statistics.mean([r[2] for r in results])} 秒")
    print(f"最大延遲: {max([r[2] for r in results])} 秒")
    print(f"Throughput/服務數量: {len(results)/total_duration/len(service_urls)} QPS")
async def hiro_pro(execute_time=1000, concurrency=10, output_folder="./results/async",
               prometheus_url="http://your_prometheus_server:9090",wait_time=20,service_urls=""):
    prometheusClient = PrometheusClient(prometheus_url)
    start_time= prometheusClient.get_current_time()
    
    # 等所有 benchmark_async_duration 完成
    
    
    
    # 合併所有結果（因為每個 result 是 list）
   
    
    total_time = prometheusClient.get_current_time() - start_time
    end_time=start_time+execute_time+wait_time
        
    print(f"start_time:{start_time}")
    print(f"end_time:{end_time}")
    print(f"step:")
    print(f"整個程式花了:{total_time}")
    print(f"start_time:{start_time}")
    print(f"總共花了{execute_time}秒")
    
if __name__ == '__main__':
    service_set=["172.22.9.141:30561",
                  "172.22.9.141:30562",
                  "172.22.9.141:30563",
                  "172.22.9.141:30564",
                  "172.22.9.141:30565",
                  "172.22.9.141:30566",
                  "172.22.9.141:30567",
                  "172.22.9.141:30578",
                  "172.22.9.141:30579",
                  "172.22.9.141:30580",
                  "172.22.9.141:30569",
                  "172.22.9.141:30571"]
    
    service_urls=service_set[11:12]
    num=len(service_urls)
    filename = now_tw.strftime("%m%d_%H:%M")
    
#14760, 23520, 28800, 31440, 33000
    asyncio.run(hiro(n_requests=500 ,concurrency=20, output_folder=f"./results/multi/{num}/{filename}/async/pose",
                      prometheus_url="http://172.22.9.249:30000",wait_time=0,service_urls=service_urls))
   