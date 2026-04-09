
import json
import time

import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
dummy = torch.zeros(1, 3, 640, 640).cuda()
half = True
if half:
    model.model.half()
    dummy = dummy.half()

# Warm-up
for _ in range(10):
    model(dummy, verbose=False)

latencies = []
for _ in range(200):
    t0 = time.perf_counter()
    model(dummy, verbose=False)
    latencies.append((time.perf_counter() - t0) * 1000)  # ms

lat = sorted(latencies)
n = len(lat)
print(json.dumps({
    "fps":        round(1000 / (sum(latencies)/n), 2),
    "latency_p50_ms": round(lat[int(n*0.50)], 3),
    "latency_p95_ms": round(lat[int(n*0.95)], 3),
    "latency_p99_ms": round(lat[int(n*0.99)], 3),
    "latency_min_ms": round(lat[0], 3),
    "latency_max_ms": round(lat[-1], 3),
}))
