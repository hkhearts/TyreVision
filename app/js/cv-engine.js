/**
 * cv-engine.js — Computer Vision Analysis Engine
 * Tire Vision — Fleet Tire Intelligence System
 *
 * Simulated rule-based CV engine using:
 * - Image pixel analysis (brightness, variance, edge density)
 * - Mileage-based feature derivation from Updated Tyre Dataset
 * - Defect signature matching from Tyre dataset
 */

import { classifyRisk } from './risk-engine.js';
import { calcReliabilityScore, calcShelfLife } from './reliability.js';

const API_BASE = window.TV_API_BASE || 'http://localhost:5000/api';

// ─── Image Quality Assessment ────────────────────────────────
export function assessImageQuality(imageElement) {
  const canvas = document.createElement('canvas');
  const maxSize = 320;
  const ratio = Math.min(maxSize / imageElement.naturalWidth, maxSize / imageElement.naturalHeight, 1);
  canvas.width  = Math.round(imageElement.naturalWidth  * ratio);
  canvas.height = Math.round(imageElement.naturalHeight * ratio);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const n = data.length / 4;
  let sumR = 0, sumG = 0, sumB = 0;
  for (let i = 0; i < data.length; i += 4) {
    sumR += data[i]; sumG += data[i+1]; sumB += data[i+2];
  }
  const avgBrightness = (sumR + sumG + sumB) / (3 * n * 255);

  // Variance (contrast proxy)
  let variance = 0;
  for (let i = 0; i < data.length; i += 4) {
    const lum = (0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]) / 255;
    variance += (lum - avgBrightness) ** 2;
  }
  variance /= n;

  // Edge density (blur detection proxy via simple diff)
  let edgeSum = 0;
  const gray = new Float32Array(n);
  for (let i = 0; i < data.length; i += 4) {
    gray[i/4] = (0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]) / 255;
  }
  const W = canvas.width;
  for (let y = 1; y < canvas.height - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const gx = gray[y*W+x+1] - gray[y*W+x-1];
      const gy = gray[(y+1)*W+x] - gray[(y-1)*W+x];
      edgeSum += Math.sqrt(gx*gx + gy*gy);
    }
  }
  const edgeDensity = edgeSum / n;
  const blurScore   = Math.min(1, edgeDensity * 8);  // 0=blurry, 1=sharp
  const overallQuality = Math.min(1,
    blurScore * 0.5 +
    Math.min(1, avgBrightness * 1.5) * 0.25 +
    Math.min(1, Math.sqrt(variance) * 4) * 0.25
  );

  return {
    blur:     Math.round(blurScore     * 1000) / 1000,
    brightness: Math.round(avgBrightness * 1000) / 1000,
    contrast:   Math.round(Math.sqrt(variance) * 1000) / 1000,
    overall:    Math.round(overallQuality * 1000) / 1000,
    pass:       overallQuality > 0.3 && avgBrightness > 0.1 && avgBrightness < 0.95,
  };
}

// ─── Extract pixel features from canvas ─────────────────────
function extractPixelFeatures(imageElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 224; canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageElement, 0, 0, 224, 224);
  const data = ctx.getImageData(0, 0, 224, 224).data;
  const n = 224 * 224;

  // Channel means
  let rSum = 0, gSum = 0, bSum = 0, lumSum = 0;
  let darkPx = 0, lightPx = 0, brightPx = 0;
  const histo = new Int32Array(256);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i+1], b = data[i+2];
    const lum = Math.round(0.299*r + 0.587*g + 0.114*b);
    rSum += r; gSum += g; bSum += b; lumSum += lum;
    histo[lum]++;
    if (lum < 50)  darkPx++;
    if (lum > 200) lightPx++;
    if (lum > 220) brightPx++;
  }

  const meanLum = lumSum / n;
  let variance = 0;
  for (let i = 0; i < data.length; i += 4) {
    const lum = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
    variance += (lum - meanLum) ** 2;
  }
  variance /= n;

  const darkRatio  = darkPx  / n;
  const lightRatio = lightPx / n;
  const brightRatio= brightPx / n;

  return { meanLum, variance, darkRatio, lightRatio, brightRatio, n };
}

// ─── Build Feature Vector (client-side fallback) ─────────────
export function buildFeatureVector(capturedImages, inspectionData) {
  const { mileageKm = 50000, position = 'drive', dotCode = '', ambientTempC = 28 } = inspectionData;

  const BASELINES = {
    steer:   { newDepth: 19, minLegal: 5, lifespan: 150000, wearRate: 0.093 },
    drive:   { newDepth: 24, minLegal: 5, lifespan: 250000, wearRate: 0.076 },
    trailer: { newDepth: 16, minLegal: 5, lifespan: 180000, wearRate: 0.061 },
  };
  const pos = (position.includes('steer') || position.includes('front')) ? 'steer'
            : position.includes('trailer') ? 'trailer' : 'drive';
  const baseline = BASELINES[pos];

  // Mileage-based baseline
  const wearMm = (mileageKm / 1000) * baseline.wearRate;
  let treadDepth = Math.max(0.5, baseline.newDepth - wearMm);

  // Image-based adjustments
  let wearVariance = 0.1, crackDensity = 0.05;
  let heatDamage   = 0.05, ozoneAging  = 0.05;
  let lightQuality = 0.7;

  if (capturedImages?.tread) {
    try {
      const pf = extractPixelFeatures(capturedImages.tread);
      // Bright rubber → more worn
      const brightModifier = (pf.meanLum / 255 - 0.35) * 6;
      treadDepth = Math.max(0.5, treadDepth - brightModifier);
      // High variance → uneven wear
      wearVariance = Math.min(0.9, (pf.variance / 2000) * 1.5);
      lightQuality = Math.min(1, (pf.variance / 500) * 0.5 + 0.5);
    } catch(e) { console.warn('[CV] Tread analysis error:', e); }
  }

  if (capturedImages?.sidewall) {
    try {
      const pf = extractPixelFeatures(capturedImages.sidewall);
      // Dark fine lines → cracks
      crackDensity = Math.min(0.9, pf.darkRatio * 4);
      ozoneAging   = Math.min(0.9, pf.darkRatio * 2.5 + pf.variance / 8000);
    } catch(e) { console.warn('[CV] Sidewall analysis error:', e); }
  }

  if (capturedImages?.shoulder) {
    try {
      const pf = extractPixelFeatures(capturedImages.shoulder);
      // High brightness at shoulders → heat glazing
      heatDamage = Math.min(0.9, pf.lightRatio * 3 + pf.variance / 12000);
    } catch(e) { console.warn('[CV] Shoulder analysis error:', e); }
  }

  // DOT parsing
  const dotInfo = parseDOT(dotCode);
  const ageMonths = dotInfo.ageMonths;

  // Deterministic noise seed from mileage
  const seed = mileageKm % 997;
  const noise = (n) => ((seed * n * 2654435761 % 65536) / 65536 - 0.5) * 0.1;

  const wearPct = mileageKm / baseline.lifespan;
  const bulgeConf = Math.max(0, Math.min(1, wearPct * 0.6 - 0.25 + noise(3)));

  const features = {
    tread_depth_mm:         Math.round(treadDepth * 100) / 100,
    tread_wear_variance:    Math.round(Math.max(0, Math.min(0.9, wearVariance + noise(1))) * 1000) / 1000,
    shoulder_wear_idx:      Math.round(Math.max(0, Math.min(0.9, wearPct * 0.55 + noise(2))) * 1000) / 1000,
    sidewall_crack_density: Math.round(Math.max(0, Math.min(0.9, crackDensity + noise(4))) * 1000) / 1000,
    bulge_detected:         bulgeConf > 0.85,
    bulge_confidence:       Math.round(Math.max(0, bulgeConf) * 1000) / 1000,
    ozone_aging_score:      Math.round(Math.max(0, Math.min(0.9, ozoneAging + ageMonths/200 + noise(5))) * 1000) / 1000,
    cut_puncture_count:     Math.max(0, Math.round(wearPct * 2.5 + noise(6) * 3)),
    tire_age_months:        Math.round(ageMonths * 10) / 10,
    retread_flag:           dotInfo.retread,
    heat_damage_idx:        Math.round(Math.max(0, Math.min(0.9, heatDamage + noise(7))) * 1000) / 1000,
    rim_corrosion_lvl:      Math.round(Math.max(0, Math.min(0.6, wearPct * 0.2 + noise(8))) * 1000) / 1000,
    valve_integrity:        Math.round(Math.max(0.2, Math.min(1, 1 - wearPct * 0.25 + noise(9))) * 1000) / 1000,
    tire_position:          pos,
    mileage_since_last:     mileageKm,
    capture_light_quality:  Math.round(lightQuality * 1000) / 1000,
    ambient_temp_c:         ambientTempC,
    exposed_cords:          treadDepth < 1.5,
    sidewall_crack_mm:      Math.round(crackDensity * 35 * 10) / 10,
  };

  return { features, dotInfo, position: pos };
}

// ─── DOT Code Parser ─────────────────────────────────────────
export function parseDOT(dotString = '') {
  const clean = dotString.replace(/\s/g, '');
  const match = clean.match(/(\d{4})/);
  if (match) {
    const ww = parseInt(match[1].slice(0, 2), 10);
    const yy = parseInt(match[1].slice(2, 4), 10);
    if (ww >= 1 && ww <= 52 && yy >= 0 && yy <= 99) {
      const year = yy <= 30 ? 2000 + yy : 1900 + yy;
      const mfgDate = new Date(year, 0, 1);
      mfgDate.setDate(mfgDate.getDate() + (ww - 1) * 7);
      const ageMonths = (Date.now() - mfgDate.getTime()) / (1000 * 60 * 60 * 24 * 30.44);
      return {
        valid: true, week: ww, year, raw: match[1],
        mfgDate: mfgDate.toISOString().slice(0, 10),
        ageMonths: Math.max(0, Math.round(ageMonths * 10) / 10),
        retread: false,
      };
    }
  }
  return { valid: false, week: 0, year: 0, raw: clean, mfgDate: null, ageMonths: 0, retread: false };
}

// ─── Main Analysis Pipeline ───────────────────────────────────
export async function analyzeTire(capturedImages, inspectionData) {
  const startTime = Date.now();

  let result;

  // Try server-side analysis first
  try {
    const serverResult = await analyzeViaServer(capturedImages, inspectionData);
    if (serverResult) {
      result = serverResult;
      result.analysisMethod = 'server';
    }
  } catch(e) {
    console.warn('[CV] Server analysis failed, using client-side engine:', e.message);
  }

  // Fallback: client-side
  if (!result) {
    const { features, dotInfo, position } = buildFeatureVector(capturedImages, inspectionData);
    const risk        = classifyRisk(features);
    const reliability = calcReliabilityScore(features, position);
    const shelf       = calcShelfLife(features, inspectionData.mileageKm, position);

    result = {
      features, dotInfo,
      risk_flag:        risk.flag,
      risk_triggers:    risk.triggers,
      risk_action:      risk.action,
      risk_color:       risk.color,
      risk_icon:        risk.icon,
      reliability_score: reliability.score,
      reliability_grade: reliability.grade,
      reliability_components: reliability.components,
      shelf_life_pct:   shelf.shelfLifePct,
      remaining_km:     shelf.remainingKm,
      replacement_date: shelf.replacementDate,
      wear_rate:        shelf.wearRate,
      position,
      analysisMethod:   'client',
    };
  }

  result.totalTimeMs = Date.now() - startTime;
  result.timestamp   = new Date().toISOString();
  result.inspectionData = inspectionData;

  return result;
}

// ─── Server Analysis ─────────────────────────────────────────
async function analyzeViaServer(capturedImages, inspectionData) {
  const treadImage = capturedImages?.tread;
  if (!treadImage) return null;

  const canvas = document.createElement('canvas');
  canvas.width = 640; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(treadImage, 0, 0, 640, 480);
  const imageB64 = canvas.toDataURL('image/jpeg', 0.85);

  const payload = {
    image:        imageB64,
    mileage_km:   inspectionData.mileageKm,
    position:     inspectionData.position,
    dot_code:     inspectionData.dotCode || '',
    ambient_temp_c: inspectionData.ambientTempC || 28,
    capture_zone: 'tread',
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);

  const resp = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: controller.signal,
  });
  clearTimeout(timeout);

  if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
  const data = await resp.json();

  // Re-run client-side risk + reliability on server features for consistency
  const features = data.features || {};
  const risk = classifyRisk(features);
  const reliability = calcReliabilityScore(features, data.position || 'drive');
  const shelf = calcShelfLife(features, inspectionData.mileageKm, data.position || 'drive');

  return {
    features,
    dotInfo: data.dot_info,
    quality: data.quality,
    risk_flag:   risk.flag,
    risk_triggers: risk.triggers,
    risk_action: risk.action,
    risk_color:  risk.color,
    risk_icon:   risk.icon,
    reliability_score: reliability.score,
    reliability_grade: reliability.grade,
    reliability_components: reliability.components,
    shelf_life_pct:  shelf.shelfLifePct,
    remaining_km:    shelf.remainingKm,
    replacement_date: shelf.replacementDate,
    wear_rate:       shelf.wearRate,
    position:        data.position,
    server_time_ms:  data.analysis_time_ms,
  };
}

// ─── OCR DOT via Server ───────────────────────────────────────
export async function ocrDOT(imageElement, hint = '') {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 400; canvas.height = 300;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, 400, 300);
    const imageB64 = canvas.toDataURL('image/jpeg', 0.9);

    const resp = await fetch(`${API_BASE}/ocr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageB64, hint }),
    });
    const data = await resp.json();
    return data;
  } catch(e) {
    // Fallback
    return parseDOT(hint);
  }
}
