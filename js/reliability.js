/**
 * reliability.js — Reliability Score & Shelf Life Calculator
 * Tire Vision — Fleet Tire Intelligence System
 */

// ─── Position Baselines ──────────────────────────────────────
export const POSITION_BASELINES = {
  steer:   { newDepth: 19.0, minLegal: 5.0, lifespanKm: 150000, wearRate: 0.093 },
  drive:   { newDepth: 24.0, minLegal: 5.0, lifespanKm: 250000, wearRate: 0.076 },
  trailer: { newDepth: 16.0, minLegal: 5.0, lifespanKm: 180000, wearRate: 0.061 },
};

// ─── Score Component Ranges per Position ─────────────────────
const NORM_RANGES = {
  tread_depth_mm:       { steer: [5, 19],   drive: [5, 24],   trailer: [5, 16]  },
  sidewall_crack_density:{ steer: [0, 1],   drive: [0, 1],    trailer: [0, 1]   },
  tread_wear_variance:  { steer: [0, 0.8],  drive: [0, 0.8],  trailer: [0, 0.8] },
  heat_damage_idx:      { steer: [0, 1],    drive: [0, 1],    trailer: [0, 1]   },
  tire_age_months:      { steer: [0, 120],  drive: [0, 120],  trailer: [0, 120] },
  valve_integrity:      { steer: [0, 1],    drive: [0, 1],    trailer: [0, 1]   },
};

function normalize(val, lo, hi) {
  return Math.max(0, Math.min(1, (val - lo) / (hi - lo)));
}
function normalizeInv(val, lo, hi) {
  return 1 - normalize(val, lo, hi);
}

/**
 * Calculate Reliability Score (0–100)
 * Formula:
 *   40 × f(tread_depth)
 * + 20 × f_inv(crack_density)
 * + 15 × f_inv(wear_variance)
 * + 10 × f_inv(heat_damage)
 * + 10 × f_inv(age_months)
 * +  5 × f(valve_integrity)
 */
export function calcReliabilityScore(features, position = 'drive') {
  const f = features || {};
  const pos = position.toLowerCase();
  const ranges = {
    tread:   NORM_RANGES.tread_depth_mm[pos]        || NORM_RANGES.tread_depth_mm.drive,
    crack:   NORM_RANGES.sidewall_crack_density[pos] || [0, 1],
    wear:    NORM_RANGES.tread_wear_variance[pos]    || [0, 0.8],
    heat:    NORM_RANGES.heat_damage_idx[pos]        || [0, 1],
    age:     NORM_RANGES.tire_age_months[pos]        || [0, 120],
    valve:   NORM_RANGES.valve_integrity[pos]        || [0, 1],
  };

  const treadScore = normalize   (f.tread_depth_mm        ?? 8,   ranges.tread[0],  ranges.tread[1]);
  const crackScore = normalizeInv(f.sidewall_crack_density ?? 0,   ranges.crack[0],  ranges.crack[1]);
  const wearScore  = normalizeInv(f.tread_wear_variance    ?? 0.1, ranges.wear[0],   ranges.wear[1]);
  const heatScore  = normalizeInv(f.heat_damage_idx        ?? 0,   ranges.heat[0],   ranges.heat[1]);
  const ageScore   = normalizeInv(f.tire_age_months        ?? 0,   ranges.age[0],    ranges.age[1]);
  const valveScore = normalize   (f.valve_integrity        ?? 0.9, ranges.valve[0],  ranges.valve[1]);

  const score =
    40 * treadScore +
    20 * crackScore +
    15 * wearScore  +
    10 * heatScore  +
    10 * ageScore   +
     5 * valveScore;

  const components = {
    tread:  Math.round(treadScore * 40 * 10) / 10,
    crack:  Math.round(crackScore * 20 * 10) / 10,
    wear:   Math.round(wearScore  * 15 * 10) / 10,
    heat:   Math.round(heatScore  * 10 * 10) / 10,
    age:    Math.round(ageScore   * 10 * 10) / 10,
    valve:  Math.round(valveScore *  5 * 10) / 10,
  };

  return {
    score:      Math.round(Math.max(0, Math.min(100, score)) * 10) / 10,
    components,
    grade:      scoreToGrade(score),
  };
}

function scoreToGrade(score) {
  if (score >= 85) return 'A';
  if (score >= 70) return 'B';
  if (score >= 55) return 'C';
  if (score >= 40) return 'D';
  return 'F';
}

/**
 * Calculate Shelf Life (remaining usable life %)
 */
export function calcShelfLife(features, mileageKm, position = 'drive', priorData = null) {
  const f = features || {};
  const pos = POSITION_BASELINES[position.toLowerCase()] || POSITION_BASELINES.drive;

  const currentDepth = f.tread_depth_mm ?? pos.newDepth;
  let wearRate = pos.wearRate; // mm per 1000 km

  // Use historical data if available
  if (priorData?.treadDepth && priorData?.mileageKm) {
    const depthDelta = priorData.treadDepth - currentDepth;
    const kmDelta    = mileageKm - priorData.mileageKm;
    if (kmDelta > 0 && depthDelta > 0) {
      wearRate = (depthDelta / kmDelta) * 1000;
    }
  }

  const remainingDepth = Math.max(0, currentDepth - pos.minLegal);
  const remainingKm    = wearRate > 0 ? (remainingDepth / wearRate) * 1000 : pos.lifespanKm;
  let shelfLifePct     = (remainingKm / pos.lifespanKm) * 100;

  // Age penalty
  const ageMo = f.tire_age_months ?? 0;
  if (ageMo > 96) shelfLifePct *= 0.70;
  else if (ageMo > 72) shelfLifePct *= 0.85;

  // Heat damage penalty
  const heat = f.heat_damage_idx ?? 0;
  if (heat > 0.5) shelfLifePct *= (1 - heat * 0.3);

  // Bulge / crack penalty
  if (f.bulge_detected) shelfLifePct *= 0.5;

  shelfLifePct = Math.round(Math.max(0, Math.min(100, shelfLifePct)) * 10) / 10;
  const remainingKmFinal = Math.round(Math.max(0, remainingKm));

  // Estimated replacement date
  let replacementDays = null;
  const dailyKm = 300; // fleet average km/day
  if (remainingKmFinal > 0) {
    replacementDays = Math.round(remainingKmFinal / dailyKm);
  }

  return {
    shelfLifePct,
    remainingKm:       remainingKmFinal,
    wearRate:          Math.round(wearRate * 1000) / 1000,
    replacementDays,
    replacementDate:   replacementDays ? daysFromNow(replacementDays) : null,
    treadRemaining_mm: Math.round(remainingDepth * 10) / 10,
  };
}

function daysFromNow(days) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

/**
 * Estimate tread depth from mileage and position (for display before inspection)
 */
export function estimateTreadFromMileage(mileageKm, position = 'drive') {
  const pos = POSITION_BASELINES[position] || POSITION_BASELINES.drive;
  const worn = (mileageKm / 1000) * pos.wearRate;
  return Math.max(0.5, pos.newDepth - worn);
}

/**
 * Shelf life percentage bar color
 */
export function shelfLifeColor(pct) {
  if (pct >= 60) return '#00e676';
  if (pct >= 35) return '#ffcc00';
  if (pct >= 15) return '#ff6b00';
  return '#ff1744';
}

/**
 * Reliability score color
 */
export function reliabilityColor(score) {
  if (score >= 80) return '#00e676';
  if (score >= 60) return '#00d4ff';
  if (score >= 40) return '#ffcc00';
  if (score >= 20) return '#ff6b00';
  return '#ff1744';
}
