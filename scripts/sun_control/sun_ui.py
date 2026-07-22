"""Interactive sun-position control for a RENI++ latent channel.

Serves a single-page UI: the 2:1 ERP box shows the live decoded environment,
and dragging the dot sets the supervised latent channel's direction to the
dot's sphere direction (the box IS the sphere, so the dot marks where the sun
is being placed). Controls: render resolution, base latent, channel norm,
azimuth offset (for emergent channels whose learnt frame is rotated relative
to the world), and channel index.

    # existing core model, emergent channel 9
    PYTHONPATH=.:scripts/figures python scripts/sun_control/sun_ui.py --port 8765

    # sun-supervised synthetic model once trained
    PYTHONPATH=.:scripts/figures python scripts/sun_control/sun_ui.py \
        --run-dir outputs/reni_sun_synth_d100 --port 8765
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

sys.path.insert(0, "scripts/figures")
from _common import MODEL_DIRS, equirect_ray_bundle, load_model  # noqa: E402
from reni.field_components.field_heads import RENIFieldHeadNames  # noqa: E402
from reni.utils.colourspace import linear_to_sRGB  # noqa: E402
from reni.utils.tonemap import two_bracket_to_linear  # noqa: E402

STATE: dict = {}
LOCK = threading.Lock()


def load(args) -> None:
    spec = Path(args.run_dir) if args.run_dir else MODEL_DIRS[args.model_key][100]
    _, _, model = load_model(spec, device=args.device)
    model.eval()
    STATE["model"] = model
    STATE["device"] = args.device
    STATE["bundles"] = {}
    bank = model.field.train_mu.detach()
    STATE["bases"] = {"mean": bank.mean(0)}
    eval_bank = getattr(model.field, "eval_mu", None)
    if eval_bank is not None:
        for i in range(min(eval_bank.shape[0], 10)):
            STATE["bases"][f"env{i}"] = eval_bank[i].detach()
    STATE["default_norm"] = float(bank.norm(dim=-1).median())
    print(f"[ui] model loaded; bases: {list(STATE['bases'])}; "
          f"median channel norm {STATE['default_norm']:.3f}")


def bundle(height: int):
    if height not in STATE["bundles"]:
        STATE["bundles"][height] = equirect_ray_bundle(
            STATE["device"], idx=0, height=height)
    return STATE["bundles"][height]


@torch.no_grad()
def render(ax: float, ay: float, height: int, base: str, norm: float,
           azoff_deg: float, channel: int) -> bytes:
    from PIL import Image

    model = STATE["model"]
    rb = bundle(height)
    W = height * 2
    col = min(int(ax * W), W - 1)
    row = min(int(ay * height), height - 1)
    d = rb.directions[row * W + col].clone()
    # optional azimuth offset in the horizontal plane (y up; planar
    # components live in x/z)
    a = math.radians(azoff_deg)
    x, z = float(d[0]), float(d[2])
    d[0] = math.cos(a) * x - math.sin(a) * z
    d[2] = math.sin(a) * x + math.cos(a) * z
    d = d / d.norm()

    z_lat = STATE["bases"][base].clone().to(STATE["device"])
    z_lat[channel] = norm * d
    z_lat = z_lat.unsqueeze(0)

    outs = []
    for start in range(0, rb.origins.shape[0], 65536):
        end = start + 65536
        samples = model.create_ray_samples(
            rb.origins[start:end], rb.directions[start:end],
            rb.camera_indices[start:end])
        out = model.field.forward(
            samples, rotation=None,
            latent_codes=z_lat.repeat(samples.shape[0], 1, 1),
        )[RENIFieldHeadNames.RGB]
        if getattr(model, "two_bracket", False):
            outs.append(two_bracket_to_linear(
                out, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log))
        else:
            outs.append(model.field.unnormalise(out))
    lin = torch.cat(outs).reshape(height, W, 3)
    img = linear_to_sRGB(lin, use_quantile=True)
    arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


PAGE = """<!doctype html>
<meta charset="utf-8"><title>RENI++ sun control</title>
<style>
 body{font-family:system-ui,sans-serif;background:#14161a;color:#dfe3e8;
      display:flex;flex-direction:column;align-items:center;gap:12px;padding:24px}
 #wrap{position:relative;cursor:crosshair;border:1px solid #3a3f46;
       border-radius:6px;overflow:hidden}
 #erp{display:block;image-rendering:auto}
 #dot{position:absolute;width:14px;height:14px;border-radius:50%;
      border:2.5px solid #fff;box-shadow:0 0 6px #000;transform:translate(-50%,-50%);
      pointer-events:none;background:rgba(255,200,60,.55)}
 .row{display:flex;gap:16px;align-items:center;flex-wrap:wrap;justify-content:center}
 label{font-size:13px;color:#9aa3ad}
 select,input{background:#1e2126;color:#dfe3e8;border:1px solid #3a3f46;
      border-radius:4px;padding:2px 6px}
 #lat{font-size:12px;color:#6b7480}
</style>
<h3>RENI++ latent sun control</h3>
<div id="wrap"><img id="erp" width="768" height="384"><div id="dot"></div></div>
<div class="row">
 <label>resolution <select id="res"><option>64</option><option selected>128</option>
   <option>192</option><option>256</option></select></label>
 <label>base <select id="base">__BASES__</select></label>
 <label>channel <input id="ch" type="number" value="__CH__" min="0" max="99"
   style="width:56px"></label>
 <label>norm <input id="norm" type="range" min="0" max="3" step="0.05"
   value="__NORM__" style="width:120px"><span id="normv"></span></label>
 <label>azimuth offset <input id="azoff" type="range" min="-180" max="180"
   step="5" value="0" style="width:140px"><span id="azv">0°</span></label>
 <span id="lat"></span>
</div>
<script>
const img=document.getElementById('erp'),dot=document.getElementById('dot'),
 wrap=document.getElementById('wrap');
let ax=0.25, ay=0.3, busy=false, queued=null;
function place(){const r=img.getBoundingClientRect();
 dot.style.left=(ax*r.width)+'px';dot.style.top=(ay*r.height)+'px';}
async function update(){
 const p=new URLSearchParams({ax,ay,h:document.getElementById('res').value,
  base:document.getElementById('base').value,
  norm:document.getElementById('norm').value,
  azoff:document.getElementById('azoff').value,
  ch:document.getElementById('ch').value});
 if(busy){queued=p;return}
 busy=true;const t0=performance.now();
 const r=await fetch('/render?'+p);const b=await r.blob();
 img.src=URL.createObjectURL(b);
 document.getElementById('lat').textContent=(performance.now()-t0).toFixed(0)+' ms';
 busy=false;if(queued){const q=queued;queued=null;update();}
}
function fromEvent(e){const r=wrap.getBoundingClientRect();
 ax=Math.min(Math.max((e.clientX-r.left)/r.width,0),0.999);
 ay=Math.min(Math.max((e.clientY-r.top)/r.height,0),0.999);
 place();update();}
let drag=false;
wrap.addEventListener('pointerdown',e=>{drag=true;fromEvent(e)});
window.addEventListener('pointermove',e=>{if(drag)fromEvent(e)});
window.addEventListener('pointerup',()=>drag=false);
for(const id of ['res','base','ch','norm','azoff'])
 document.getElementById(id).addEventListener('input',()=>{
  document.getElementById('normv').textContent=document.getElementById('norm').value;
  document.getElementById('azv').textContent=document.getElementById('azoff').value+'°';
  update();});
place();update();
document.getElementById('normv').textContent=document.getElementById('norm').value;
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            bases = "".join(f"<option>{b}</option>" for b in STATE["bases"])
            page = (PAGE.replace("__BASES__", bases)
                        .replace("__CH__", str(STATE["channel"]))
                        .replace("__NORM__", f"{STATE['default_norm']:.2f}"))
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/render":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            with LOCK:
                png = render(
                    float(q.get("ax", 0.25)), float(q.get("ay", 0.3)),
                    int(q.get("h", 128)), q.get("base", "mean"),
                    float(q.get("norm", STATE["default_norm"])),
                    float(q.get("azoff", 0.0)),
                    int(q.get("ch", STATE["channel"])))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
        else:
            self.send_response(404)
            self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="vnjoint_ortho_2cyc")
    parser.add_argument("--run-dir", default=None,
                        help="Run directory override (e.g. the sun-supervised "
                             "synthetic model).")
    parser.add_argument("--channel", type=int, default=9)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    STATE["channel"] = args.channel
    load(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[ui] serving on http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
