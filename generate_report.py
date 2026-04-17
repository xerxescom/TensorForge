#!/usr/bin/env python3
"""
将 final_report.json 渲染为独立 HTML 报告
用法: python report/generate_report.py results/final_report.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def load_report(path: str) -> dict:
    return json.loads(Path(path).read_text())


def extract_metric(phases: list, phase_name: str, *keys):
    """从 phases 列表中安全取值"""
    for p in phases:
        if p.get("phase") == phase_name:
            result = p.get("result", p.get("concurrent_metrics", {}))
            if isinstance(result, dict):
                val = result
                for k in keys:
                    if isinstance(val, dict):
                        val = val.get(k)
                    else:
                        return None
                return val
    return None


def build_chart_data(report: dict) -> dict:
    phases = report.get("phases", [])

    def phase_result(name):
        for p in phases:
            if p.get("phase") == name:
                return p.get("result", {})
        return {}

    llm_q4 = phase_result("llm")
    llm_fp16 = phase_result("llm_fp16")
    diffusion = phase_result("diffusion")
    cv = phase_result("cv")
    asr = phase_result("asr")
    ctx_scale = phase_result("llm_context_scale")
    concurrent = phase_result("concurrent")

    def m(d, *keys, default=0):
        v = d
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, default)
            else:
                return default
        return v if v is not None else default

    # 吞吐量对比
    throughput = {
        "labels": ["LLM Q4 (tok/s)", "LLM FP16 (tok/s)", "Diffusion (it/s)", "CV (FPS)"],
        "values": [
            m(llm_q4, "metrics", "tokens_per_s_mean"),
            m(llm_fp16, "metrics", "tokens_per_s_mean"),
            m(diffusion, "metrics", "it_per_s"),
            m(cv, "metrics", "fps"),
        ],
    }

    # 功耗
    power = {
        "labels": ["LLM Q4", "LLM FP16", "Diffusion", "CV", "ASR"],
        "mean": [
            m(llm_q4, "gpu_stats", "power_w", "mean"),
            m(llm_fp16, "gpu_stats", "power_w", "mean"),
            m(diffusion, "gpu_stats", "power_w", "mean"),
            m(cv, "gpu_stats", "power_w", "mean"),
            m(asr, "gpu_stats", "power_w", "mean"),
        ],
        "max": [
            m(llm_q4, "gpu_stats", "power_w", "max"),
            m(llm_fp16, "gpu_stats", "power_w", "max"),
            m(diffusion, "gpu_stats", "power_w", "max"),
            m(cv, "gpu_stats", "power_w", "max"),
            m(asr, "gpu_stats", "power_w", "max"),
        ],
    }

    # 温度
    temp = {
        "labels": power["labels"],
        "max": [
            m(llm_q4, "gpu_stats", "temp_c", "max"),
            m(llm_fp16, "gpu_stats", "temp_c", "max"),
            m(diffusion, "gpu_stats", "temp_c", "max"),
            m(cv, "gpu_stats", "temp_c", "max"),
            m(asr, "gpu_stats", "temp_c", "max"),
        ],
    }

    # 上下文长度衰减
    ctx_curve = m(ctx_scale, "metrics", "context_scale_curve", default=[])

    # 效率 (tokens/J)
    efficiency = {
        "labels": ["LLM Q4", "LLM FP16", "Diffusion"],
        "values": [
            m(llm_q4, "metrics", "tokens_per_joule"),
            m(llm_fp16, "metrics", "tokens_per_joule"),
            m(diffusion, "metrics", "steps_per_joule"),
        ],
    }

    # 并发衰减
    degradation = m(concurrent, "degradation_ratios", default={})
    concurrent_metrics = m(concurrent, "concurrent_metrics", default=[])

    split_rows = []
    for item in concurrent_metrics:
        if not isinstance(item, dict):
            continue
        task_type = item.get("type", "unknown")
        model = item.get("model", "")
        split_rows.append(
            {
                "task": task_type,
                "model": model,
                "measurement_mode": item.get("measurement_mode", ""),
                "e2e": item.get("tokens_per_s_end_to_end", item.get("it_per_s_end_to_end", 0)),
                "infer": item.get(
                    "tokens_per_s_inference_only", item.get("it_per_s_inference_only", 0)
                ),
                "unit": "tok/s" if task_type == "llm" else ("it/s" if task_type == "diffusion" else ""),
            }
        )

    # 雷达图（归一化评分，简单线性缩放）
    def score(val, max_val, invert=False):
        if max_val == 0:
            return 0
        s = min(val / max_val, 1.0) * 100
        return round(100 - s if invert else s, 1)

    tok_s = m(llm_q4, "metrics", "tokens_per_s_mean")
    radar = {
        "labels": ["LLM 速度", "图像速度", "CV 速度", "功耗效率", "低延迟"],
        "values": [
            score(tok_s, 150),
            score(m(diffusion, "metrics", "it_per_s"), 10),
            score(m(cv, "metrics", "fps"), 500),
            score(m(llm_q4, "metrics", "tokens_per_joule"), 1),
            score(m(llm_q4, "metrics", "ttft_s_min"), 2.0, invert=True),
        ],
    }

    return {
        "throughput": throughput,
        "power": power,
        "temp": temp,
        "ctx_curve": ctx_curve,
        "efficiency": efficiency,
        "degradation": degradation,
        "concurrent_split": split_rows,
        "radar": radar,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU AI Benchmark Report — {gpu_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d0d0f;
    --surface: #17171a;
    --surface2: #1e1e22;
    --border: rgba(255,255,255,0.08);
    --text: #e8e8ea;
    --muted: #7a7a82;
    --accent: #7c6af7;
    --accent2: #3ec9a0;
    --warn: #f0884a;
    --danger: #e24b4a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.6;
  }}
  header {{
    padding: 2rem 2.5rem 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    font-size: 20px;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: var(--accent);
  }}
  header p {{
    color: var(--muted);
    margin-top: 4px;
    font-size: 12px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1px;
    background: var(--border);
    border-top: 1px solid var(--border);
  }}
  .card {{
    background: var(--surface);
    padding: 1.5rem;
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1.2rem;
  }}
  canvas {{ max-height: 240px; }}
  .stat-row {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }}
  .stat {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    min-width: 140px;
  }}
  .stat .label {{ color: var(--muted); font-size: 11px; margin-bottom: 4px; }}
  .stat .value {{ font-size: 22px; font-weight: 500; color: var(--accent); }}
  .stat .unit  {{ font-size: 11px; color: var(--muted); margin-left: 3px; }}
  .degrade-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .degrade-table td, .degrade-table th {{
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    text-align: left;
  }}
  .degrade-table th {{ color: var(--muted); font-weight: 400; }}
  .ok   {{ color: var(--accent2); }}
  .warn {{ color: var(--warn); }}
  .bad  {{ color: var(--danger); }}
  footer {{
    padding: 1.5rem 2.5rem;
    color: var(--muted);
    font-size: 11px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
<header>
  <h1>GPU Benchmark Report — {gpu_name}</h1>
  <p>Generated {date} &nbsp;·&nbsp; Results dir: {output_dir}</p>
</header>

<div class="grid">

  <div class="card full">
    <h2>Key stats</h2>
    <div class="stat-row" id="key-stats"></div>
  </div>

  <div class="card">
    <h2>Throughput by task</h2>
    <canvas id="chart-throughput"></canvas>
  </div>

  <div class="card">
    <h2>Power draw (W)</h2>
    <canvas id="chart-power"></canvas>
  </div>

  <div class="card">
    <h2>Peak temperature (°C)</h2>
    <canvas id="chart-temp"></canvas>
  </div>

  <div class="card">
    <h2>Efficiency (perf / joule)</h2>
    <canvas id="chart-eff"></canvas>
  </div>

  <div class="card">
    <h2>LLM context length scaling</h2>
    <canvas id="chart-ctx"></canvas>
  </div>

  <div class="card">
    <h2>Capability radar</h2>
    <canvas id="chart-radar"></canvas>
  </div>

  <div class="card">
    <h2>Concurrent load degradation</h2>
    <div id="degrade-content"></div>
  </div>

  <div class="card full">
    <h2>Concurrent throughput split (end-to-end vs inference-only)</h2>
    <div id="concurrent-split"></div>
  </div>

</div>
<footer>
  Auto-generated by GPU Benchmark Suite &nbsp;·&nbsp; raw data in final_report.json
</footer>

<script>
const DATA = {chart_data_json};

const palette = {{
  accent:  'rgba(124,106,247,0.85)',
  accent2: 'rgba(62,201,160,0.85)',
  warn:    'rgba(240,136,74,0.85)',
  danger:  'rgba(226,75,74,0.85)',
  muted:   'rgba(122,122,130,0.5)',
  grid:    'rgba(255,255,255,0.06)',
  text:    '#e8e8ea',
}};

Chart.defaults.color = palette.text;
Chart.defaults.borderColor = palette.grid;
Chart.defaults.font.family = "'SF Mono','Fira Code',monospace";
Chart.defaults.font.size = 11;

function bar(id, labels, datasets, opts={{}}) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10 }} }} }},
      scales: {{
        x: {{ grid: {{ color: palette.grid }} }},
        y: {{ grid: {{ color: palette.grid }}, beginAtZero: true }},
      }},
      ...opts,
    }}
  }});
}}

function line(id, labels, datasets) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10 }} }} }},
      scales: {{
        x: {{ grid: {{ color: palette.grid }} }},
        y: {{ grid: {{ color: palette.grid }}, beginAtZero: true }},
      }},
    }}
  }});
}}

// Key stats
const ks = [
  ['LLM Q4 speed',  (DATA.throughput.values[0]||0).toFixed(1), 'tok/s'],
  ['LLM FP16 speed',(DATA.throughput.values[1]||0).toFixed(1), 'tok/s'],
  ['Diffusion',     (DATA.throughput.values[2]||0).toFixed(2), 'it/s'],
  ['CV FPS',        (DATA.throughput.values[3]||0).toFixed(0), 'fps'],
];
document.getElementById('key-stats').innerHTML = ks.map(([l,v,u]) =>
  `<div class="stat"><div class="label">${{l}}</div><span class="value">${{v}}</span><span class="unit">${{u}}</span></div>`
).join('');

// Throughput
bar('chart-throughput',
  DATA.throughput.labels,
  [{{ label: 'throughput', data: DATA.throughput.values,
     backgroundColor: [palette.accent, palette.accent2, palette.warn, palette.danger] }}]
);

// Power
bar('chart-power',
  DATA.power.labels,
  [
    {{ label: 'mean W', data: DATA.power.mean, backgroundColor: palette.accent }},
    {{ label: 'max W',  data: DATA.power.max,  backgroundColor: palette.warn  }},
  ]
);

// Temp
bar('chart-temp',
  DATA.temp.labels,
  [{{ label: 'max °C', data: DATA.temp.max, backgroundColor: palette.warn }}]
);

// Efficiency
bar('chart-eff',
  DATA.efficiency.labels,
  [{{ label: 'units/J', data: DATA.efficiency.values, backgroundColor: palette.accent2 }}]
);

// Context scale
const ctx = DATA.ctx_curve;
if (ctx && ctx.length) {{
  line('chart-ctx',
    ctx.map(r => r.context_tokens + ' tok'),
    [{{ label: 'tok/s', data: ctx.map(r => r.tokens_per_s),
       borderColor: palette.accent, backgroundColor: 'rgba(124,106,247,0.1)',
       fill: true, tension: 0.3 }}]
  );
}} else {{
  document.getElementById('chart-ctx').parentElement.innerHTML +=
    '<p style="color:var(--muted);font-size:12px;margin-top:8px">No context scale data</p>';
}}

// Radar
new Chart(document.getElementById('chart-radar'), {{
  type: 'radar',
  data: {{
    labels: DATA.radar.labels,
    datasets: [{{
      label: 'score (0–100)',
      data: DATA.radar.values,
      backgroundColor: 'rgba(124,106,247,0.15)',
      borderColor: palette.accent,
      pointBackgroundColor: palette.accent,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    scales: {{ r: {{ beginAtZero: true, max: 100,
      grid: {{ color: palette.grid }}, pointLabels: {{ color: palette.text }} }} }},
    plugins: {{ legend: {{ position: 'bottom' }} }},
  }}
}});

// Degradation table
const dg = DATA.degradation;
const keys = Object.keys(dg);
if (keys.length) {{
  const rows = keys.map(k => {{
    const v = dg[k];
    const cls = v >= 0.9 ? 'ok' : v >= 0.7 ? 'warn' : 'bad';
    return `<tr><td>${{k}}</td><td class="${{cls}}">${{(v*100).toFixed(1)}}%</td></tr>`;
  }}).join('');
  document.getElementById('degrade-content').innerHTML =
    `<table class="degrade-table"><thead><tr><th>metric</th><th>concurrent / solo</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}} else {{
  document.getElementById('degrade-content').innerHTML =
    '<p style="color:var(--muted);font-size:12px">No concurrent test data</p>';
}}

// Concurrent throughput split table
const cs = DATA.concurrent_split || [];
if (cs.length) {{
  const rows = cs.map(r =>
    `<tr><td>${{r.task}}</td><td>${{r.model||'-'}}</td><td>${{r.measurement_mode||'-'}}</td><td>${{Number(r.e2e||0).toFixed(2)}} ${{r.unit}}</td><td>${{Number(r.infer||0).toFixed(2)}} ${{r.unit}}</td></tr>`
  ).join('');
  document.getElementById('concurrent-split').innerHTML =
    `<table class="degrade-table"><thead><tr><th>task</th><th>model</th><th>mode</th><th>end-to-end</th><th>inference-only</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}} else {{
  document.getElementById('concurrent-split').innerHTML =
    '<p style="color:var(--muted);font-size:12px">No concurrent throughput split data</p>';
}}
</script>
</body>
</html>"""


def generate(report_path: str, out_html: str | None = None):
    report = load_report(report_path)
    chart_data = build_chart_data(report)

    html = HTML_TEMPLATE.format(
        gpu_name=report.get("gpu_name", "Unknown GPU"),
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        output_dir=str(Path(report_path).parent),
        chart_data_json=json.dumps(chart_data),
    )

    out = out_html or str(Path(report_path).parent / "report.html")
    Path(out).write_text(html, encoding="utf-8")
    print(f"Report generated → {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <final_report.json> [output.html]")
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
