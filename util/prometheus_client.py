import time
import requests

class PrometheusClient:
    def __init__(self, prometheus_url: str):
        """
        Prometheus 查詢用戶端

        參數：
            prometheus_url (str): Prometheus server 位址，格式如 "http://<ip>:<port>"
        """
        self.prometheus_url = prometheus_url.rstrip("/")

    def get_current_time(self) -> float:
        """
        查詢 Prometheus 的現在時間（使用內建 time()）

        回傳：
            float: Prometheus 回傳的目前 UTC 時間戳（秒）
        """
        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": "time()"}
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("data", {}).get("result", [])
            return float(result[1])  # 取出 string timestamp 並轉為 float
        except Exception as e:
            print("Error getting Prometheus current time:", e)
            return time.time()  # fallback 使用本地時間

    def query_range(self, metric: str, start: int, end: int, step: str) -> list:
        """
        查詢某段時間內的時間序列資料（query_range）

        參數：
            metric (str): PromQL 查詢語句，例如 "DCGM_FI_DEV_GPU_UTIL{gpu='0'}"
            start (int): 起始時間戳（秒）
            end (int): 結束時間戳（秒）
            step (str): 每筆資料的間隔（例如 "5", "30", "1m"）

        回傳：
            list: 每筆資料為 [timestamp, value]，若無資料回傳空列表
        """
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": metric,
            "start": start,
            "end": end,
            "step": step
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            result = resp.json()["data"]["result"]
            return result[0]["values"] if result else []
        except Exception as e:
            print("Error querying metric range:", e)
            return []
