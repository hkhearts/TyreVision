/**
 * ar-overlay.js — Custom Canvas AR Camera Overlay
 * Tire Vision — Fleet Tire Intelligence System
 *
 * Renders zone guides, scan animations, quality feedback
 * directly onto a canvas overlaid on the live camera feed.
 */

export class AROverlay {
  constructor(videoEl, canvasEl, options = {}) {
    this.video    = videoEl;
    this.canvas   = canvasEl;
    this.ctx      = canvasEl.getContext('2d');
    this.zone     = options.zone || 'tread';    // tread | sidewall | shoulder | dot | valve
    this.onCapture = options.onCapture || null;
    this.quality  = 0;
    this.scanning = false;
    this.scanY    = 0;
    this.scanDir  = 1;
    this.captured = false;
    this.autoCapture = options.autoCapture !== false;
    this.qualityThreshold = options.qualityThreshold || 0.45;
    this._raf     = null;
    this._stream  = null;
    this._frameCount = 0;
  }

  // ─── Start camera ─────────────────────────────────────────
  async start() {
    try {
      const constraints = {
        video: {
          facingMode: { ideal: 'environment' },
          width:  { ideal: 1280 },
          height: { ideal: 720 },
        }
      };
      this._stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this._stream;
      await this.video.play();
      this._resizeCanvas();
      window.addEventListener('resize', () => this._resizeCanvas());
      this._loop();
      return true;
    } catch(err) {
      console.error('[AR] Camera error:', err);
      return false;
    }
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._stream) this._stream.getTracks().forEach(t => t.stop());
    this._stream = null;
    this._raf    = null;
  }

  _resizeCanvas() {
    const rect = this.video.getBoundingClientRect();
    this.canvas.width  = rect.width  || this.video.videoWidth  || 640;
    this.canvas.height = rect.height || this.video.videoHeight || 480;
  }

  // ─── Main render loop ─────────────────────────────────────
  _loop() {
    this._raf = requestAnimationFrame(() => this._loop());
    this._frameCount++;
    this._render();

    // Assess quality every 10 frames
    if (this._frameCount % 10 === 0 && !this.captured) {
      this._assessQuality();
    }
  }

  _render() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (this.captured) {
      this._drawCaptureFlash(ctx, W, H);
      return;
    }

    const t = Date.now() / 1000;

    switch (this.zone) {
      case 'tread':    this._drawTreadZone(ctx, W, H, t); break;
      case 'sidewall': this._drawSidewallZone(ctx, W, H, t); break;
      case 'shoulder': this._drawShoulderZone(ctx, W, H, t); break;
      case 'dot':      this._drawDotZone(ctx, W, H, t); break;
      case 'valve':    this._drawValveZone(ctx, W, H, t); break;
      default:         this._drawTreadZone(ctx, W, H, t);
    }

    this._drawScanLine(ctx, W, H, t);
    this._drawQualityBar(ctx, W, H);
    this._drawCornerBrackets(ctx, W, H, t);
  }

  // ─── Tread Zone ───────────────────────────────────────────
  _drawTreadZone(ctx, W, H, t) {
    const mx = W * 0.1, my = H * 0.15, mw = W * 0.8, mh = H * 0.7;
    const pulse = 0.5 + 0.5 * Math.sin(t * 2);

    // Main guide box
    ctx.strokeStyle = `rgba(0, 212, 255, ${0.6 + pulse * 0.3})`;
    ctx.lineWidth   = 2;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(mx, my, mw, mh);
    ctx.setLineDash([]);

    // Tread groove indicators (horizontal lines)
    ctx.strokeStyle = `rgba(0, 212, 255, 0.25)`;
    ctx.lineWidth   = 1;
    const numGrooves = 5;
    for (let i = 1; i < numGrooves; i++) {
      const y = my + (mh * i) / numGrooves;
      ctx.beginPath(); ctx.moveTo(mx, y); ctx.lineTo(mx + mw, y); ctx.stroke();
    }

    // Depth measurement markers (left side)
    const depthLabels = ['Min', '4mm', '8mm', '12mm', 'Max'];
    ctx.fillStyle = 'rgba(0, 212, 255, 0.8)';
    ctx.font      = '11px JetBrains Mono, monospace';
    depthLabels.forEach((label, i) => {
      const y = my + (mh * i) / (depthLabels.length - 1);
      ctx.fillText(label, mx - 35, y + 4);
      ctx.beginPath();
      ctx.arc(mx, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Center cross
    ctx.strokeStyle = `rgba(0, 212, 255, 0.5)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W/2 - 15, H/2); ctx.lineTo(W/2 + 15, H/2);
    ctx.moveTo(W/2, H/2 - 15); ctx.lineTo(W/2, H/2 + 15);
    ctx.stroke();

    // Label
    this._drawLabel(ctx, W/2, my - 12, '● TREAD FACE — POSITION TIRE FLAT');
  }

  // ─── Sidewall Zone ────────────────────────────────────────
  _drawSidewallZone(ctx, W, H, t) {
    const cx = W/2, cy = H/2;
    const rx = W * 0.38, ry = H * 0.42;
    const pulse = 0.5 + 0.5 * Math.sin(t * 2);

    // Ellipse guide
    ctx.strokeStyle = `rgba(255, 204, 0, ${0.6 + pulse * 0.3})`;
    ctx.lineWidth   = 2.5;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Inner ring
    ctx.strokeStyle = `rgba(255, 204, 0, 0.2)`;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx * 0.7, ry * 0.7, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Zone labels
    ctx.fillStyle = 'rgba(255, 204, 0, 0.9)';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText('OUTER', cx + rx * 0.35, cy - ry * 0.6);
    ctx.fillText('INNER', cx - rx * 0.7,  cy + ry * 0.1);

    // Crack detection region indicators
    const crackZones = [
      {a: -0.3, label: 'Check Upper'},
      {a: 0.8,  label: 'Check Side'},
      {a: 2.2,  label: 'Check Lower'},
    ];
    crackZones.forEach(({a, label}) => {
      const x = cx + Math.cos(a) * rx;
      const y = cy + Math.sin(a) * ry;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 204, 0, 0.7)`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    this._drawLabel(ctx, cx, cy - ry - 16, '● SIDEWALL — SCAN FOR CRACKS & BULGES');
  }

  // ─── Shoulder Zone ────────────────────────────────────────
  _drawShoulderZone(ctx, W, H, t) {
    const mx = W * 0.05, my = H * 0.2, mw = W * 0.9, mh = H * 0.6;
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.5);

    // Trapezoidal shoulder guide
    ctx.strokeStyle = `rgba(255, 107, 0, ${0.6 + pulse * 0.3})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(mx + mw * 0.1, my);
    ctx.lineTo(mx + mw * 0.9, my);
    ctx.lineTo(mx + mw, my + mh);
    ctx.lineTo(mx, my + mh);
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // Shoulder-to-tread transition arc
    ctx.strokeStyle = `rgba(255, 107, 0, 0.35)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(W/2, my + mh * 0.15, mw * 0.35, Math.PI * 0.1, Math.PI * 0.9);
    ctx.stroke();

    this._drawLabel(ctx, W/2, my - 14, '● SHOULDER — CAPTURE TREAD-TO-SIDEWALL CURVE');
  }

  // ─── DOT Code Zone ────────────────────────────────────────
  _drawDotZone(ctx, W, H, t) {
    const bx = W * 0.15, by = H * 0.3, bw = W * 0.7, bh = H * 0.35;
    const pulse = 0.5 + 0.5 * Math.sin(t * 3);

    // OCR target box
    ctx.strokeStyle = `rgba(0, 230, 118, ${0.7 + pulse * 0.25})`;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.strokeRect(bx, by, bw, bh);

    // Text target region
    ctx.fillStyle = 'rgba(0, 230, 118, 0.06)';
    ctx.fillRect(bx, by, bw, bh);

    // Character boxes (DOT format: 12 chars typically)
    const charW = bw / 12;
    for (let i = 0; i < 12; i++) {
      ctx.strokeStyle = `rgba(0, 230, 118, 0.25)`;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx + i * charW + 1, by + bh * 0.25, charW - 2, bh * 0.5);
    }

    // Magnify icon
    ctx.strokeStyle = 'rgba(0, 230, 118, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(bx - 20, by + bh / 2, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(bx - 12, by + bh/2 + 7);
    ctx.lineTo(bx - 5, by + bh/2 + 14);
    ctx.stroke();

    this._drawLabel(ctx, W/2, by - 14, '● DOT CODE — 4-DIGIT DATE CODE e.g. 3024');
  }

  // ─── Valve Zone ───────────────────────────────────────────
  _drawValveZone(ctx, W, H, t) {
    const cx = W/2, cy = H/2;
    const r  = Math.min(W, H) * 0.22;
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.5);

    // Target reticle
    ctx.strokeStyle = `rgba(0, 212, 255, ${0.6 + pulse * 0.3})`;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, r * 0.5, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(0, 212, 255, 0.4)`;
    ctx.stroke();

    // Crosshair
    ctx.strokeStyle = `rgba(0, 212, 255, ${0.4 + pulse * 0.2})`;
    ctx.lineWidth = 1;
    [[cx, cy-r*1.2, cx, cy-r*0.6],
     [cx, cy+r*0.6, cx, cy+r*1.2],
     [cx-r*1.2, cy, cx-r*0.6, cy],
     [cx+r*0.6, cy, cx+r*1.2, cy]].forEach(([x1,y1,x2,y2]) => {
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    });

    // Rim inspection zones (four quadrants)
    ['↑ Cap', '→ Stem', '↓ Rim', '← Seal'].forEach((label, i) => {
      const angle = (i * Math.PI / 2) - Math.PI / 4;
      const lx = cx + Math.cos(angle) * r * 1.4;
      const ly = cy + Math.sin(angle) * r * 1.4;
      ctx.fillStyle = 'rgba(0, 212, 255, 0.7)';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, lx, ly);
    });
    ctx.textAlign = 'left';

    this._drawLabel(ctx, W/2, cy - r - 20, '● VALVE & RIM — CENTER VALVE STEM IN TARGET');
  }

  // ─── Scan Line ─────────────────────────────────────────────
  _drawScanLine(ctx, W, H, t) {
    const scanY = ((t * 0.4) % 1) * H;
    const grad = ctx.createLinearGradient(0, scanY - 15, 0, scanY + 15);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(0.5, 'rgba(0, 212, 255, 0.3)');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.fillRect(0, scanY - 15, W, 30);
  }

  // ─── Corner Brackets ──────────────────────────────────────
  _drawCornerBrackets(ctx, W, H, t) {
    const len   = 28;
    const margin = 16;
    const color = this.quality > this.qualityThreshold
      ? `rgba(0, 230, 118, ${0.7 + 0.3 * Math.sin(t * 4)})`
      : `rgba(0, 212, 255, 0.7)`;

    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.lineCap     = 'square';

    const corners = [
      [margin, margin, 1, 1],
      [W - margin, margin, -1, 1],
      [margin, H - margin, 1, -1],
      [W - margin, H - margin, -1, -1],
    ];
    corners.forEach(([x, y, dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(x + dx * len, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * len);
      ctx.stroke();
    });
  }

  // ─── Quality Bar ──────────────────────────────────────────
  _drawQualityBar(ctx, W, H) {
    const bw = 120, bh = 8, bx = W - bw - 12, by = H - 28;
    const qPct = this.quality;
    const color = qPct > 0.7 ? '#00e676' : qPct > 0.4 ? '#ffcc00' : '#ff1744';

    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.beginPath();
    ctx.roundRect(bx - 4, by - 14, bw + 8, bh + 22, 4);
    ctx.fill();

    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle = color;
    ctx.fillRect(bx, by, bw * Math.min(1, qPct), bh);

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(`QUALITY ${Math.round(qPct * 100)}%`, bx, by - 3);
  }

  // ─── Label ────────────────────────────────────────────────
  _drawLabel(ctx, x, y, text) {
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    const metrics = ctx.measureText(text);
    const tw = metrics.width;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(x - tw/2 - 8, y - 13, tw + 16, 18);
    ctx.fillStyle = 'rgba(0, 212, 255, 0.95)';
    ctx.fillText(text, x, y);
    ctx.textAlign = 'left';
  }

  // ─── Capture flash ────────────────────────────────────────
  _drawCaptureFlash(ctx, W, H) {
    const t = Date.now();
    if (!this._captureTime) this._captureTime = t;
    const elapsed = (t - this._captureTime) / 1000;
    if (elapsed < 0.15) {
      ctx.fillStyle = `rgba(255,255,255,${0.8 - elapsed * 5})`;
      ctx.fillRect(0, 0, W, H);
    }

    // ✓ mark
    const alpha = Math.min(1, elapsed * 4);
    ctx.strokeStyle = `rgba(0, 230, 118, ${alpha})`;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(W/2 - 20, H/2);
    ctx.lineTo(W/2 - 5, H/2 + 18);
    ctx.lineTo(W/2 + 25, H/2 - 15);
    ctx.stroke();
  }

  // ─── Quality assessment ───────────────────────────────────
  _assessQuality() {
    if (!this.video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = 80; canvas.height = 60;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.video, 0, 0, 80, 60);
    const data = ctx.getImageData(0, 0, 80, 60).data;
    const n = 80 * 60;
    let sum = 0, sumSq = 0;
    for (let i = 0; i < data.length; i += 4) {
      const lum = (0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]) / 255;
      sum += lum; sumSq += lum * lum;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    const contrastScore = Math.min(1, Math.sqrt(variance) * 5);
    const brightnessScore = 1 - Math.abs(mean - 0.4) * 2;
    this.quality = Math.max(0, Math.min(1, contrastScore * 0.6 + brightnessScore * 0.4));

    // Auto-capture when quality is sufficient
    if (this.autoCapture && this.quality >= this.qualityThreshold && !this.captured) {
      if (!this._qualityFrames) this._qualityFrames = 0;
      this._qualityFrames++;
      if (this._qualityFrames >= 3) this.capture();
    } else {
      this._qualityFrames = 0;
    }
  }

  // ─── Manual or auto capture ───────────────────────────────
  async capture() {
    if (this.captured) return null;
    this.captured = true;
    this._captureTime = null;

    const canvas = document.createElement('canvas');
    canvas.width  = this.video.videoWidth  || 1280;
    canvas.height = this.video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);

    // Haptic feedback
    if ('vibrate' in navigator) navigator.vibrate(50);

    return new Promise(resolve => {
      canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          if (this.onCapture) this.onCapture(img, blob, url);
          resolve({ img, blob, url });
        };
        img.src = url;
      }, 'image/jpeg', 0.92);
    });
  }

  // Allow re-capture
  reset() {
    this.captured = false;
    this._captureTime = null;
    this._qualityFrames = 0;
  }

  setZone(zone) {
    this.zone = zone;
    this.reset();
  }
}
