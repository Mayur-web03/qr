"""POST /api/decode  (raw image bytes, Content-Type: application/octet-stream)
Returns {"text": "..."} or {"text": null}.
Engine: zxing-cpp (the C++ ZXing reader). Finds a QR anywhere in a full photo or
screenshot, handles inverted / dark-mode codes, rotation, and iPhone HEIC files."""
import io
import json
import time
from http.server import BaseHTTPRequestHandler

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import zxingcpp


_BIN = getattr(zxingcpp, "Binarizer", None)
BINARIZERS = [None] + [getattr(_BIN, n) for n in ("GlobalHistogram", "LocalAverage", "FixedThreshold")
                       if _BIN is not None and hasattr(_BIN, n)]


def _read(arr):
    """Run zxing-cpp on one uint8 array (several binarizers); return text or None."""
    for b in BINARIZERS:
        kw = dict(formats=zxingcpp.BarcodeFormat.QRCode, try_rotate=True, try_downscale=True)
        if b is not None:
            kw["binarizer"] = b
        try:
            res = zxingcpp.read_barcodes(arr, **kw)
        except TypeError:
            try:
                res = zxingcpp.read_barcodes(arr)
            except Exception:
                res = []
        except Exception:
            res = []
        for r in res:
            if r.text:
                return r.text
    return None


def _views(gray):
    """Normal, inverted and contrast-stretched copies of one grayscale array."""
    yield gray
    yield 255 - gray
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    if hi - lo > 20:
        st = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        yield st
        yield 255 - st


def _try_gray(gray):
    for v in _views(np.ascontiguousarray(gray)):
        t = _read(v)
        if t:
            return t
    return None


def _windows(w, h):
    m = min(w, h)
    out = []
    for f in (0.8, 0.55, 0.38):
        s = int(m * f)
        step = max(1, s // 2)
        y = 0
        while True:
            yy = min(y, h - s)
            x = 0
            while True:
                xx = min(x, w - s)
                out.append((xx, yy, s))
                if xx >= w - s:
                    break
                x += step
            if yy >= h - s:
                break
            y += step
    cx, cy = w / 2, h / 2
    out.sort(key=lambda a: (a[0] + a[2] / 2 - cx) ** 2 + (a[1] + a[2] / 2 - cy) ** 2)
    return out[:20]


def _qr_boxes(lum, max_boxes=4):
    """Locate QR-like regions (busy, mixed light/dark blocks) in a full photo.
    Returns boxes (x0, y0, x1, y1) in `lum` coordinates, biggest first."""
    h, w = lum.shape
    sc = min(1.0, 640.0 / max(h, w))
    sm = np.asarray(Image.fromarray(lum).resize((max(8, int(w * sc)), max(8, int(h * sc))), Image.BILINEAR)).astype(np.float32)
    lo, hi = np.percentile(sm, 5), np.percentile(sm, 95)
    if hi - lo < 25:
        return []
    mid = (lo + hi) / 2.0
    gx = np.abs(np.diff(sm, axis=1, prepend=sm[:, :1]))
    gy = np.abs(np.diff(sm, axis=0, prepend=sm[:1, :]))
    edge = ((gx + gy) > (hi - lo) * 0.25).astype(np.float32)
    bright = (sm > mid).astype(np.float32)
    B = 8
    bh, bw = sm.shape[0] // B, sm.shape[1] // B
    if bh < 3 or bw < 3:
        return []
    e = edge[:bh * B, :bw * B].reshape(bh, B, bw, B).mean(axis=(1, 3))
    b = bright[:bh * B, :bw * B].reshape(bh, B, bw, B).mean(axis=(1, 3))
    mask = (e > 0.12) & (b > 0.15) & (b < 0.85)
    seen = np.zeros_like(mask)
    comps = []
    for sy in range(bh):
        for sx in range(bw):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack, cells = [(sy, sx)], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < bh and 0 <= nx < bw and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) >= 12:
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                comps.append((len(cells), min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    comps.sort(reverse=True)
    boxes = []
    k = B / sc
    for _, x0, y0, x1, y1 in comps[:max_boxes]:
        boxes.append((int(x0 * k), int(y0 * k), min(w, int(x1 * k)), min(h, int(y1 * k))))
    return boxes


def _try_quiet(lum, box):
    """Some designs put the QR on a dark panel, so the finder patterns merge with the
    background and no light quiet zone exists. Trim inward by several amounts and
    add a white border so the reader sees a proper quiet zone."""
    x0, y0, x1, y1 = box
    h, w = lum.shape
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    for grow in (1.15, 1.0, 1.3):
        half = int(side * grow / 2)
        ax0, ay0 = max(0, cx - half), max(0, cy - half)
        ax1, ay1 = min(w, cx + half), min(h, cy + half)
        base = lum[ay0:ay1, ax0:ax1]
        if base.size == 0:
            continue
        for inset in (0.0, 0.04, 0.07, 0.10, 0.13, 0.16):
            i = int(min(base.shape) * inset)
            c = base[i:base.shape[0] - i, i:base.shape[1] - i]
            if min(c.shape) < 40:
                continue
            im = Image.fromarray(np.ascontiguousarray(c))
            tgt = 480
            im = im.resize((tgt, int(tgt * c.shape[0] / c.shape[1]) or 1), Image.LANCZOS)
            arr = np.pad(np.asarray(im), 48, mode="constant", constant_values=255)
            t = _read(arr) or _read(255 - arr)
            if t:
                return t
    return None


def find_qr(im, budget=20.0, live=False):
    """budget: seconds. Live camera frames use a short budget and fewer channels
    so results stay quick."""
    t_end = time.time() + budget
    im = im.convert("RGB")
    im.thumbnail((2000, 2000), Image.LANCZOS)
    rgb = np.asarray(im)
    lum = np.asarray(im.convert("L"))

    # 1) whole image: luminance (live), or luminance + single colour channels (upload)
    chans = (lum,) if live else (lum, rgb[:, :, 1], rgb[:, :, 2], rgb[:, :, 0])
    for g in chans:
        if time.time() > t_end:
            return None
        t = _try_gray(g)
        if t:
            return t

    # 1b) locate QR-like regions and retry with a forced white quiet zone
    #     (fixes codes drawn on dark/coloured panels with no light border)
    try:
        for box in _qr_boxes(lum):
            if time.time() > t_end:
                return None
            t = _try_quiet(lum, box)
            if t:
                return t
    except Exception:
        pass

    # 2) smaller copy (reduces screen moire and noise)
    small = im.copy()
    small.thumbnail((800, 800), Image.LANCZOS)
    t = _try_gray(np.asarray(small.convert("L")))
    if t:
        return t
    for edge in (1200, 500):   # more sizes: the best scale differs per photo
        sm = im.copy()
        sm.thumbnail((edge, edge), Image.LANCZOS)
        t = _try_gray(np.asarray(sm.convert("L")))
        if t:
            return t
        if time.time() > t_end:
            return None

    # 3) zoomed windows, centre first, upscaled if tiny
    h, w = lum.shape
    for x, y, s in _windows(w, h):
        if time.time() > t_end:
            break
        crop = Image.fromarray(lum[y:y + s, x:x + s])
        if s < 500:
            crop = crop.resize((500, 500), Image.LANCZOS)
        elif s > 900:
            crop = crop.resize((900, 900), Image.LANCZOS)
        t = _try_gray(np.asarray(crop))
        if t:
            return t
    return None


def open_image(body):
    im = Image.open(io.BytesIO(body))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(405, {"error": "Use POST"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            if not body:
                return self._send(400, {"error": "Empty upload"})
            try:
                im = open_image(body)
            except Exception:
                return self._send(422, {"error": "Could not read that image file."})
            live = self.headers.get("X-Mode") == "live"
            self._send(200, {"text": find_qr(im, 4.0 if live else 20.0, live)})
        except Exception:
            self._send(500, {"error": "Server error while scanning."})
