#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isaac 训练离线可视化服务（纯本地、零外网依赖，默认端口 8600）。

生产环境无外网时，云端 wandb / `wandb server`（首次需拉镜像）都不可用。本服务
直接解析训练日志（<output>/train.log，由 finetune_isaac05.py 自动 tee 生成），
在浏览器里实时看 loss/指标曲线。只依赖 Python 标准库（http.server），无需联网、
无需 docker、无需 wandb。

用法:
    python viz_server.py --log-dir /data/algorithm/repo/outputs/isaac-finetune/smoke \
        --host 0.0.0.0 --port 8600

    浏览器打开 http://<host>:8600 即可。
    数据接口:
      GET /metrics -> { "latest": {...}, "series": {"<metric>": [[step, val], ...]}, "mtime": ... }
      GET /log     -> 最近 --tail 行原始日志（text/plain）
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# lerobot MetricsTracker.__str__ 每行形如:
#   step:20 smpl:32 ep:2 epch:0.02 loss:1.2345 loss/flow_loss:1.1 ...
METRIC_RE = re.compile(r"(\S+):([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

# 需要忽略的非数值/非指标字段（step 单独作为 x 轴，其它都作为曲线）
SKIP_KEYS = {"smpl", "ep", "epch", "samples", "episodes", "epochs"}
# 只保留这些指标曲线（可调；None = 全部数值字段）
KEEP_PREFIXES = ("loss", "lr", "grad", "mem", "flow", "text", "timestep")


def parse_metric_line(line: str) -> dict[str, float] | None:
    if "step:" not in line:
        return None
    fields: dict[str, float] = {}
    for key, value in METRIC_RE.findall(line):
        if key in SKIP_KEYS or key == "step":
            continue
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    if not fields:
        return None
    step = None
    m = re.search(r"step:(\d+)", line)
    if m:
        step = int(m.group(1))
    return {"step": step, **fields}


def tail(path: Path, lines: int = 5000) -> list[str]:
    if not path.is_file():
        return []
    with open(path, "rb") as fh:
        # 只读最后 lines 行
        fh.seek(0, 2)
        size = fh.tell()
        block = 1 << 16
        data = b""
        while size > 0 and len(data.splitlines()) < lines:
            size = max(0, size - block)
            fh.seek(size)
            data = fh.read(block) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def build_metrics(log_path: Path, tail_lines: int = 5000) -> dict:
    series: dict[str, list[list[float]]] = {}
    latest: dict = {}
    for line in tail(log_path, tail_lines):
        parsed = parse_metric_line(line)
        if not parsed:
            continue
        step = parsed["step"]
        for key, value in parsed.items():
            if key == "step":
                continue
            if not (KEEP_PREFIXES and key.startswith(KEEP_PREFIXES)):
                continue
            series.setdefault(key, []).append([float(step), value])
        latest = {"step": step, **{k: v for k, v in parsed.items() if k != "step"}}
    return {"latest": latest, "series": series, "mtime": log_path.stat().st_mtime if log_path.is_file() else 0.0}


class VizHandler(BaseHTTPRequestHandler):
    log_path: Path = Path("train.log")
    tail_lines: int = 5000

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            payload = json.dumps(build_metrics(self.log_path, self.tail_lines)).encode("utf-8")
            self._send(200, payload, "application/json")
        elif path == "/log":
            body = "\n".join(tail(self.log_path, self.tail_lines)).encode("utf-8")
            self._send(200, body, "text/plain; charset=utf-8")
        else:
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def log_message(self, fmt: str, *args):  # noqa: A003
        print(f"[viz] {self.client_address[0]} {fmt % args}")


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Isaac 训练可视化</title>
<style>
 body{font:14px/1.5 system-ui;margin:16px;background:#111;color:#ddd}
 h1{font-size:18px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
 .card{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px}
 .card h2{font-size:13px;margin:0 0 6px;color:#9cf}
 canvas{width:100%;height:220px;display:block}
 #latest{font-family:monospace;white-space:pre-wrap;background:#0d0d0d;padding:8px;border-radius:6px;font-size:12px}
</style>
<h1>Isaac 0.5 训练可视化（离线）</h1>
<div id="latest">加载中…</div><div class="grid" id="cards"></div>
<script>
const KEEP = ["loss","loss/flow_loss","loss/text_loss","flow_loss","text_loss","lr","grad_norm","gpu_mem_gb"];
async function refresh(){
  try{
    const r = await fetch("/metrics"); const d = await r.json();
    if(!d.series){ return; }
    const latest=d.latest||{};
    document.getElementById("latest").textContent =
      "step: "+(latest.step??"-")+"   "+
      ["loss","loss/flow_loss","loss/text_loss","lr","grad_norm"].map(k=>k+"="+(latest[k]!=null?latest[k].toFixed(4):"-")).join("   ")+
      "   (更新于 "+new Date().toLocaleTimeString()+")";
    const grid=document.getElementById("cards"); grid.innerHTML="";
    for(const key of Object.keys(d.series).sort()){
      if(!KEEP.some(p=>key.includes(p))) continue;
      const pts=d.series[key]; if(!pts.length) continue;
      const card=document.createElement("div"); card.className="card";
      const h=document.createElement("h2"); h.textContent=key;
      const cv=document.createElement("canvas");
      card.appendChild(h); card.appendChild(cv); grid.appendChild(card);
      draw(cv,pts);
    }
  }catch(e){ console.error(e); }
}
function draw(cv,pts){
  const c=cv.getContext("2d"), W=cv.width=cv.clientWidth*devicePixelRatio, H=cv.height=cv.clientHeight*devicePixelRatio;
  c.clearRect(0,0,W,H);
  if(pts.length<2) return;
  let x0=pts[0][0],x1=pts[pts.length-1][0]; if(x1===x0) x1=x0+1;
  let y0=Math.min(...pts.map(p=>p[1])), y1=Math.max(...pts.map(p=>p[1]));
  if(y1===y0){y0-=1;y1+=1;}
  const pad=8, px=x=>pad+(x-x0)/(x1-x0)*(W-2*pad), py=y=>H-pad-(y-y0)/(y1-y0)*(H-2*pad);
  c.strokeStyle="#333";c.strokeRect(pad,pad,W-2*pad,H-2*pad);
  c.strokeStyle="#4af"; c.beginPath();
  pts.forEach((p,i)=> i? c.lineTo(px(p[0]),py(p[1])) : c.moveTo(px(p[0]),py(p[1])));
  c.stroke();
}
setInterval(refresh, 3000); refresh();
</script></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac 训练离线可视化（默认端口 8600）")
    parser.add_argument("--log-dir", required=True,
                        help="输出目录（finetune_isaac05.py 的 --output-dir，含 train.log）")
    parser.add_argument("--log-path", default=None, help="直接指定 train.log 路径（优先于 --log-dir）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--tail", type=int, default=5000, help="解析的最近日志行数")
    args = parser.parse_args()

    log_path = Path(args.log_path) if args.log_path else Path(args.log_dir) / "train.log"
    # 兼容：finetune_isaac05.py 现在把日志写到 <output>.train.log（输出目录旁），
    # 不在 <output>/train.log。--log-dir 时若目录内没有 train.log，自动尝试旁边的同名文件。
    if not log_path.is_file() and not args.log_path:
        sibling = Path(args.log_dir).parent / (Path(args.log_dir).name + ".train.log")
        if sibling.is_file():
            log_path = sibling
    VizHandler.log_path = log_path
    VizHandler.tail_lines = args.tail
    if not log_path.is_file():
        print(f"[警告] 日志文件尚不存在: {log_path}（训练开始写日志后自动可用）")

    server = ThreadingHTTPServer((args.host, args.port), VizHandler)
    print(f"离线可视化服务: http://{args.host}:{args.port}   (日志: {log_path})")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
