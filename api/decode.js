// POST /api/decode  (body: raw image bytes, Content-Type: application/octet-stream)
// Returns { text: "<decoded>" } or { text: null }.
// Handles JPG, PNG, WebP, GIF, AVIF and iPhone HEIC. Tries normal + inverted QR,
// several crops and sizes, so dark-mode / inverted codes decode too.
const sharp = require('sharp');
const jsQR = require('jsqr');
const Z = require('@zxing/library');

const hints = new Map();
hints.set(Z.DecodeHintType.TRY_HARDER, true);
hints.set(Z.DecodeHintType.POSSIBLE_FORMATS, [Z.BarcodeFormat.QR_CODE]);

function readBody(req) {
  if (Buffer.isBuffer(req.body)) return Promise.resolve(req.body);
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function rgba(buf, maxDim) {
  const { data, info } = await sharp(buf, { failOn: 'none' })
    .rotate()
    .flatten({ background: '#ffffff' })
    .resize({ width: maxDim, height: maxDim, fit: 'inside', withoutEnlargement: true })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, w: info.width, h: info.height };
}

// Decode to pixels; if the format is unsupported (HEIC), convert to JPEG first.
async function load(buf, maxDim) {
  try {
    return { px: await rgba(buf, maxDim), buf };
  } catch (e) {
    const convert = require('heic-convert');
    const jpg = Buffer.from(await convert({ buffer: buf, format: 'JPEG', quality: 0.92 }));
    return { px: await rgba(jpg, maxDim), buf: jpg };
  }
}

function crop(px, x, y, cw, ch) {
  const out = Buffer.alloc(cw * ch * 4);
  for (let r = 0; r < ch; r++) {
    px.data.copy(out, r * cw * 4, ((y + r) * px.w + x) * 4, ((y + r) * px.w + x + cw) * 4);
  }
  return { data: out, w: cw, h: ch };
}

function luminance(px, invert) {
  const n = px.w * px.h, l = new Uint8ClampedArray(n);
  let mn = 255, mx = 0;
  for (let i = 0; i < n; i++) {
    const v = (px.data[i * 4] * 77 + px.data[i * 4 + 1] * 150 + px.data[i * 4 + 2] * 29) >> 8;
    l[i] = v; if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const k = mx > mn ? 255 / (mx - mn) : 1;
  for (let i = 0; i < n; i++) { const v = (l[i] - mn) * k; l[i] = invert ? 255 - v : v; }
  return l;
}

function decodeView(px) {
  const r = jsQR(new Uint8ClampedArray(px.data.buffer, px.data.byteOffset, px.data.length),
                 px.w, px.h, { inversionAttempts: 'attemptBoth' });
  if (r && r.data) return r.data;
  for (const inv of [false, true]) {
    try {
      const reader = new Z.MultiFormatReader();
      reader.setHints(hints);
      const bmp = new Z.BinaryBitmap(new Z.HybridBinarizer(
        new Z.RGBLuminanceSource(luminance(px, inv), px.w, px.h)));
      const res = reader.decode(bmp);
      if (res && res.getText()) return res.getText();
    } catch (e) { /* not found in this view */ }
  }
  return null;
}

function windows(w, h) {
  const m = Math.min(w, h), list = [];
  [0.85, 0.6, 0.42].forEach((f) => {
    const s = Math.round(m * f), step = Math.max(1, Math.round(s / 2));
    for (let y = 0; ; y += step) {
      const yy = Math.min(y, h - s);
      for (let x = 0; ; x += step) {
        const xx = Math.min(x, w - s);
        list.push([xx, yy, s]);
        if (xx >= w - s) break;
      }
      if (yy >= h - s) break;
    }
  });
  const cx = w / 2, cy = h / 2;
  list.sort((a, b) => Math.hypot(a[0] + a[2] / 2 - cx, a[1] + a[2] / 2 - cy) -
                      Math.hypot(b[0] + b[2] / 2 - cx, b[1] + b[2] / 2 - cy));
  return list.slice(0, 18);
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') { res.setHeader('Allow', 'POST'); return res.status(405).json({ error: 'Use POST' }); }
  try {
    const body = await readBody(req);
    if (!body || !body.length) return res.status(400).json({ error: 'Empty upload' });
    const deadline = Date.now() + 20000;

    const big = await load(body, 1600);
    let t = decodeView(big.px);                       // 1. whole image
    if (t) return res.status(200).json({ text: t });

    try {                                             // 2. smaller copy (helps screen moire/noise)
      const small = await rgba(big.buf, 700);
      t = decodeView(small);
      if (t) return res.status(200).json({ text: t });
    } catch (e) {}

    for (const [x, y, s] of windows(big.px.w, big.px.h)) {   // 3. zoomed crops, centre first
      if (Date.now() > deadline) break;
      t = decodeView(crop(big.px, x, y, s, s));
      if (t) return res.status(200).json({ text: t });
    }
    return res.status(200).json({ text: null });
  } catch (e) {
    return res.status(422).json({ error: 'Could not read that image file.' });
  }
};
