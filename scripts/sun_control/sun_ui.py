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
    # Default command magnitude must match the SUN channel's trained norms;
    # the all-channel median can be several times smaller and renders
    # commands too faintly to see.
    STATE["default_norm"] = float(bank[:, args.channel].norm(dim=-1).median())
    if args.bases_file:
        payload = torch.load(
            args.bases_file, map_location="cpu", weights_only=True)
        stored = payload["bases"] if "bases" in payload else payload
        if args.bases_only:
            STATE["bases"] = {}
        for name, latent in stored.items():
            if tuple(latent.shape) != tuple(bank.shape[1:]):
                raise ValueError(
                    f"Base {name!r} has shape {tuple(latent.shape)}, "
                    f"expected {tuple(bank.shape[1:])}")
            STATE["bases"][name] = latent.detach().cpu()
        if "norm" in payload:
            STATE["default_norm"] = float(payload["norm"])
        print(f"[ui] loaded {len(stored)} fitted bases from "
              f"{args.bases_file}")
    STATE["exposure"] = {}
    print(f"[ui] model loaded; bases: {list(STATE['bases'])}; "
          f"ch{args.channel} median norm {STATE['default_norm']:.3f} "
          f"(all-channel {float(bank.norm(dim=-1).median()):.3f})")
    if args.purify_bases:
        purify_bases(args.purify_bases)


def _decode_small(model, rb, z_lat):
    outs = []
    for s in range(0, rb.origins.shape[0], 65536):
        e = s + 65536
        sm = model.create_ray_samples(rb.origins[s:e], rb.directions[s:e],
                                      rb.camera_indices[s:e])
        out = model.field.forward(
            sm, rotation=None,
            latent_codes=z_lat.unsqueeze(0).repeat(sm.shape[0], 1, 1),
        )[RENIFieldHeadNames.RGB]
        if getattr(model, "two_bracket", False):
            out = two_bracket_to_linear(
                out, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
        else:
            out = model.field.unnormalise(out)
        outs.append(out)
    return torch.cat(outs)


def purify_bases(steps: int) -> None:
    """Strip ghost suns from the env bases. Eval latents are fitted by pure
    reconstruction, so their content channels re-encode the sun; commanding
    ch9 then fights that ghost. Pin ch9 at the base's detected sun and refit
    only the other channels against the base's own decode: the doubly
    rendered sun overshoots the target, so the optimiser removes the
    content-channel sun pathway."""
    from reni.utils.tonemap import luminance

    model, dev, ch = STATE["model"], STATE["device"], STATE["channel"]
    rb = bundle(64)
    for name in [b for b in STATE["bases"] if b.startswith("env")]:
        H64, W64 = 64, 128
        v = (torch.arange(H64) + 0.5) / H64 * math.pi
        u = (torch.arange(W64) + 0.5) / W64 * 2.0 * math.pi - math.pi
        pol, azm = torch.meshgrid(v, u, indexing="ij")
        analytic = torch.stack([
            torch.sin(pol) * torch.sin(azm), torch.cos(pol),
            torch.sin(pol) * torch.cos(azm)], -1).reshape(-1, 3).to(dev)
        base = STATE["bases"][name].clone().to(dev)
        with torch.no_grad():
            target = _decode_small(model, rb, base)
            lum = luminance(target.reshape(-1, 3)).reshape(-1)
            sel = lum >= torch.quantile(lum, 0.999)
            d = torch.nn.functional.normalize(
                (analytic[sel] * lum[sel][:, None]).sum(0), dim=0)
        z = base.clone()
        z[ch] = STATE["default_norm"] * d
        others = [c for c in range(z.shape[0]) if c != ch]
        free = z[others].clone().requires_grad_(True)
        opt = torch.optim.Adam([free], lr=3e-2)
        for _ in range(steps):
            opt.zero_grad()
            z_full = z.detach().clone()
            z_full[others] = free
            pred = _decode_small(model, rb, z_full)
            loss = torch.nn.functional.mse_loss(
                torch.log1p(pred.clamp(min=0)), torch.log1p(target))
            loss.backward()
            opt.step()
        with torch.no_grad():
            z[others] = free.detach()
        STATE["bases"][name] = z.detach().cpu()
        print(f"[ui] purified {name}: sun pinned, refit loss {loss.item():.5f}")


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
    # Command in the GENERATOR's analytic ERP frame (y-up, top row =
    # zenith): the training labels live there, so ch9 semantics do too.
    # rb.directions uses the nerfstudio camera convention, which maps the
    # same pixel to a very different direction (top-center reads ~-18 deg
    # elevation) and silently breaks dot tracking.
    pol = (row + 0.5) / height * math.pi
    azm = (col + 0.5) / W * 2.0 * math.pi - math.pi
    d = torch.tensor([math.sin(pol) * math.sin(azm), math.cos(pol),
                      math.sin(pol) * math.cos(azm)],
                     device=STATE["device"], dtype=torch.float32)
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
    # Fixed exposure per (base, height), measured once from the UNMODIFIED
    # base decode: per-frame quantile auto-exposure otherwise rescales the
    # whole image whenever the commanded sun brightens the peak, which
    # reads as a scene-wide colour "snap" while dragging.
    key = (base, height)
    if key not in STATE["exposure"]:
        if base == "mean" or norm == 0.0:
            ref = lin
        else:
            ref = None
        if ref is None:
            z_ref = STATE["bases"][base].clone().to(STATE["device"]).unsqueeze(0)
            refs = []
            for start in range(0, rb.origins.shape[0], 65536):
                end = start + 65536
                samples = model.create_ray_samples(
                    rb.origins[start:end], rb.directions[start:end],
                    rb.camera_indices[start:end])
                out = model.field.forward(
                    samples, rotation=None,
                    latent_codes=z_ref.repeat(samples.shape[0], 1, 1),
                )[RENIFieldHeadNames.RGB]
                if getattr(model, "two_bracket", False):
                    refs.append(two_bracket_to_linear(
                        out, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log))
                else:
                    refs.append(model.field.unnormalise(out))
            ref = torch.cat(refs).reshape(height, W, 3)
        STATE["exposure"][key] = float(
            torch.quantile(ref.reshape(-1, 3).max(-1).values, 0.97).clamp(min=1e-6))
    img = (lin / STATE["exposure"][key]).clamp(0, 1) ** (1.0 / 2.2)
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
    parser.add_argument("--bases-file", type=Path, default=None,
                        help="Optional .pt file containing {'bases': "
                             "{name: latent}, 'norm': float}.")
    parser.add_argument("--bases-only", action="store_true",
                        help="With --bases-file, hide the checkpoint's mean "
                             "and raw evaluation-bank bases.")
    parser.add_argument("--purify-bases", type=int, default=0, metavar="STEPS",
                        help="Refit env bases with ch pinned at their detected "
                             "sun to strip content-channel ghost suns.")
    args = parser.parse_args()
    STATE["channel"] = args.channel
    load(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[ui] serving on http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
