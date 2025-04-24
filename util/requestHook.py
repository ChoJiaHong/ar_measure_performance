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

async def benchmark_async_duration(test_duration_seconds, concurrency,service_url):
    print(f"Starting benchmark for {test_duration_seconds} seconds with concurrency={concurrency}")
    stub, channel = await get_stub(service_url=service_url)
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