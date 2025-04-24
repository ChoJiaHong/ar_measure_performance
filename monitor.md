非常明確，👍你希望從 **Node Exporter** 中取得「簡潔版」的關鍵系統資訊，不要看到每個 CPU core / mode / label 組合的一堆線。

以下是我幫你整理的 **Prometheus Query 清單**，每個查詢都會只回傳一條或幾條「有意義」的整體數據，適合用來觀察系統資源是否可能是 AI 推理的瓶頸。

---

## ✅ 1. 總體 CPU 使用率（百分比）

```promql
100 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100
```

- 顯示每個節點的 CPU 平均使用率（忽略細節的 user/system/etc）。
- 若超過 90%，可能是 CPU 飽和的徵兆。

---

## ✅ 2. CPU 等待 I/O 的時間比例（iowait）

```promql
avg by (instance) (rate(node_cpu_seconds_total{mode="iowait"}[1m])) * 100
```

- 如果這個值高，代表 CPU 花很多時間在等待磁碟/網路 I/O → 可能是資料存取的瓶頸。

---

## ✅ 3. 記憶體可用量（剩餘 RAM）

```promql
node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100
```

- 顯示剩餘記憶體的百分比（不是已用，是可用），低於 10% 要注意。
- 可配合這個看實際值（MiB）：
  ```promql
  node_memory_MemAvailable_bytes / 1024 / 1024
  ```

---

## ✅ 4. Swap 使用量（用到代表記憶體不足）

```promql
node_memory_SwapUsed_bytes / 1024 / 1024
```

- 有使用 Swap 表示實體記憶體可能不足，系統效能會下降。

---

## ✅ 5. 磁碟 I/O 使用率（磁碟忙碌程度 %）

```promql
rate(node_disk_io_time_seconds_total[1m]) * 100
```

- 若某些裝置接近 100%，表示磁碟正在飽和。
- 可進一步這樣聚合每個節點：
  ```promql
  sum by (instance) (rate(node_disk_io_time_seconds_total[1m])) * 100
  ```

---

## ✅ 6. 每秒磁碟讀寫量（MB/s）

```promql
sum by (instance) (
  rate(node_disk_read_bytes_total[1m]) +
  rate(node_disk_written_bytes_total[1m])
) / 1024 / 1024
```

- 觀察磁碟傳輸量，判斷是否正在大量讀寫資料。

---

## ✅ 7. 網路接收 + 傳送流量（MB/s）

```promql
sum by (instance) (
  rate(node_network_receive_bytes_total[1m]) +
  rate(node_network_transmit_bytes_total[1m])
) / 1024 / 1024
```

- 判斷該節點是否有大量進出流量（例如推理服務接收圖片 / 傳回結果）。
- 若某服務主要靠 HTTP 接收推理任務，可以看此是否對應負載增加。

---

## ✅ 8. 節點 Load Average（1分鐘負載）

```promql
node_load1
```

- 值若高於 CPU 核心數，代表系統有排隊壓力。
- 可以這樣寫出「每核平均 Load」：
  ```promql
  node_load1 / count(count(node_cpu_seconds_total{mode="idle"}) by (cpu, instance))
  ```

---

## 📦 Bonus：整合到 Grafana Panel

這些 query 非常適合做出以下面板：

| 資源 | Query | 可設警示？ |
|------|-------|------------|
| CPU 使用率 | ✅ | ✅ |
| CPU iowait | ✅ | ✅ |
| 記憶體使用率 | ✅ | ✅ |
| 磁碟忙碌程度 | ✅ | ✅ |
| Swap 使用量 | ✅ | ✅ |
| 網路流量 | ✅ | 可選 |
| Load 平均 | ✅ | ✅ |

---

如果你希望我幫你產出一份對應的 **Grafana JSON dashboard 匯入檔**，或整合成一份 K8s 服務性能觀測用的「一頁總覽」，我也可以幫你製作！

需要我幫你整理一份嗎？