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


def find_qr(im, budget=20.0):
    """budget: seconds. Live camera frames use a short budget so results stay quick."""
    t_end = time.time() + budget
    im = im.convert("RGB")
    im.thumbnail((2000, 2000), Image.LANCZOS)
    rgb = np.asarray(im)
    lum = np.asarray(im.convert("L"))

    # 1) whole image: luminance, then single colour channels (helps coloured backgrounds)
    for g in (lum, rgb[:, :, 1], rgb[:, :, 2], rgb[:, :, 0]):
        if time.time() > t_end:
            return None
        t = _try_gray(g)
        if t:
            return t

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
            self._send(200, {"text": find_qr(im, 4.0 if live else 20.0)})
        except Exception:
            self._send(500, {"error": "Server error while scanning."})
