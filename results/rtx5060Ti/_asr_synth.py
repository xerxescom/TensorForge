
import time, json, numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda",
                     compute_type="float16")
# 合成 5 秒音频
audio = np.random.randn(5 * 16000).astype(np.float32)
latencies = []
for _ in range(5):
    t0 = time.perf_counter()
    segs, _ = model.transcribe(audio, language="en")
    list(segs)
    latencies.append(time.perf_counter() - t0)

mean_lat = sum(latencies) / len(latencies)
print(json.dumps({
    "rtf": round(mean_lat / 5.0, 4),
    "latency_mean_s": round(mean_lat, 3),
    "audio_duration_s": 5.0,
    "note": "synthetic_audio_benchmark",
}))
