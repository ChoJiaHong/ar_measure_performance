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
from zoneinfo import ZoneInfo  # Python 3.9+
# 取得台灣時間（Asia/Taipei）
now_tw = datetime.datetime.now(ZoneInfo("Asia/Taipei"))

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