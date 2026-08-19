/**
 * risk-engine.js — Hard-rule Risk Flag Classifier
 * Tire Vision — Fleet Tire Intelligence System
 *
 * Implements the 4-class risk classification:
 * DO-NOT-OPERATE → CRITICAL → MONITOR → SAFE
 */

export const RISK_FLAGS = {
  DNO:      'DO-NOT-OPERATE',
  CRITICAL: 'CRITICAL',
  MONITOR:  'MONITOR',
  SAFE:     'SAFE',
};

export const RISK_COLORS = {
  'DO-NOT-OPERATE': '#ff1744',
  'CRITICAL':       '#ff6b00',
  'MONITOR':        '#ffcc00',
  'SAFE':           '#00e676',
};

export const RISK_CSS_CLASS = {
  'DO-NOT-OPERATE': 'dno',
  'CRITICAL':       'critical',
  'MONITOR':        'monitor',
  'SAFE':           'safe',
};

export const RISK_ICONS = {
  'DO-NOT-OPERATE': '🚫',
  'CRITICAL':       '⚠️',
  'MONITOR':        '👁️',
  'SAFE':           '✅',
};

export const RISK_REQUIRED_ACTION = {
  'DO-NOT-OPERATE': 'IMMEDIATE REPLACEMENT — DO NOT OPERATE THIS VEHICLE',
  'CRITICAL':       'Replace tire within 48 hours or 100 miles',
  'MONITOR':        'Schedule follow-up inspection within 2–4 weeks',
  'SAFE':           'Continue normal operation — no immediate action required',
};

/**
 * Classify risk from feature vector.
 * @param {Object} features - the 16-feature vector
 * @returns {{ flag: string, cssClass: string, triggers: string[], action: string, color: string }}
 */
export function classifyRisk(features) {
  const f = features || {};
  const triggers = [];

  // ── DO-NOT-OPERATE ──────────────────────────────────────
  const dnoTriggers = [];

  if ((f.bulge_confidence ?? 0) > 0.85) {
    dnoTriggers.push(`Bulge detected (confidence: ${((f.bulge_confidence ?? 0) * 100).toFixed(0)}%)`);
  }
  if ((f.tread_depth_mm ?? 10) < 2.0) {
    dnoTriggers.push(`Tread depth critically low: ${(f.tread_depth_mm ?? 0).toFixed(1)} mm`);
  }
  if (f.exposed_cords) {
    dnoTriggers.push('Exposed cords detected — structural failure imminent');
  }
  if ((f.sidewall_crack_mm ?? 0) > 25) {
    dnoTriggers.push(`Large sidewall crack: ${(f.sidewall_crack_mm ?? 0).toFixed(0)} mm`);
  }

  if (dnoTriggers.length > 0) {
    return {
      flag:     RISK_FLAGS.DNO,
      cssClass: RISK_CSS_CLASS[RISK_FLAGS.DNO],
      triggers: dnoTriggers,
      action:   RISK_REQUIRED_ACTION[RISK_FLAGS.DNO],
      color:    RISK_COLORS[RISK_FLAGS.DNO],
      icon:     RISK_ICONS[RISK_FLAGS.DNO],
    };
  }

  // ── CRITICAL ─────────────────────────────────────────────
  const criticalTriggers = [];

  if ((f.tread_depth_mm ?? 10) < 4.0) {
    criticalTriggers.push(`Tread depth below safe threshold: ${(f.tread_depth_mm ?? 0).toFixed(1)} mm (min 4 mm)`);
  }
  if ((f.cut_puncture_count ?? 0) >= 2) {
    criticalTriggers.push(`Multiple deep cuts / punctures: ${f.cut_puncture_count}`);
  }
  if ((f.heat_damage_idx ?? 0) > 0.8) {
    criticalTriggers.push(`Severe heat damage index: ${((f.heat_damage_idx ?? 0) * 100).toFixed(0)}%`);
  }
  if ((f.ozone_aging_score ?? 0) > 0.85) {
    criticalTriggers.push(`Severe ozone aging / dry rot: ${((f.ozone_aging_score ?? 0) * 100).toFixed(0)}%`);
  }

  if (criticalTriggers.length > 0) {
    return {
      flag:     RISK_FLAGS.CRITICAL,
      cssClass: RISK_CSS_CLASS[RISK_FLAGS.CRITICAL],
      triggers: criticalTriggers,
      action:   RISK_REQUIRED_ACTION[RISK_FLAGS.CRITICAL],
      color:    RISK_COLORS[RISK_FLAGS.CRITICAL],
      icon:     RISK_ICONS[RISK_FLAGS.CRITICAL],
    };
  }

  // ── MONITOR ──────────────────────────────────────────────
  const monitorTriggers = [];

  if ((f.tread_wear_variance ?? 0) > 0.3) {
    monitorTriggers.push(`Uneven wear variance: ${((f.tread_wear_variance ?? 0)).toFixed(2)} (threshold 0.3)`);
  }
  if ((f.sidewall_crack_density ?? 0) > 0.3) {
    monitorTriggers.push(`Moderate sidewall cracking density: ${((f.sidewall_crack_density ?? 0) * 100).toFixed(0)}%`);
  }
  if ((f.tire_age_months ?? 0) > 72) {
    monitorTriggers.push(`Tire age exceeds 6 years: ${Math.floor((f.tire_age_months ?? 0) / 12)} years ${Math.floor((f.tire_age_months ?? 0) % 12)} months`);
  }
  if ((f.shoulder_wear_idx ?? 0) > 0.6) {
    monitorTriggers.push(`Elevated shoulder wear: ${((f.shoulder_wear_idx ?? 0) * 100).toFixed(0)}%`);
  }
  if ((f.rim_corrosion_lvl ?? 0) > 0.5) {
    monitorTriggers.push(`Significant rim corrosion: ${((f.rim_corrosion_lvl ?? 0) * 100).toFixed(0)}%`);
  }

  if (monitorTriggers.length > 0) {
    return {
      flag:     RISK_FLAGS.MONITOR,
      cssClass: RISK_CSS_CLASS[RISK_FLAGS.MONITOR],
      triggers: monitorTriggers,
      action:   RISK_REQUIRED_ACTION[RISK_FLAGS.MONITOR],
      color:    RISK_COLORS[RISK_FLAGS.MONITOR],
      icon:     RISK_ICONS[RISK_FLAGS.MONITOR],
    };
  }

  // ── SAFE ─────────────────────────────────────────────────
  return {
    flag:     RISK_FLAGS.SAFE,
    cssClass: RISK_CSS_CLASS[RISK_FLAGS.SAFE],
    triggers: ['No critical thresholds exceeded'],
    action:   RISK_REQUIRED_ACTION[RISK_FLAGS.SAFE],
    color:    RISK_COLORS[RISK_FLAGS.SAFE],
    icon:     RISK_ICONS[RISK_FLAGS.SAFE],
  };
}

/**
 * Get urgency level (0=safe, 1=monitor, 2=critical, 3=dno) for sorting
 */
export function riskUrgency(flag) {
  const map = { 'SAFE': 0, 'MONITOR': 1, 'CRITICAL': 2, 'DO-NOT-OPERATE': 3 };
  return map[flag] ?? 0;
}

/**
 * Derive risk from mileage alone (for vehicles with no recent inspection)
 */
export function riskFromMileage(mileageKm, position = 'drive') {
  const lifespan = { steer: 150000, drive: 250000, trailer: 180000 };
  const pct = mileageKm / (lifespan[position] || 250000);
  if (pct >= 0.95) return RISK_FLAGS.DNO;
  if (pct >= 0.80) return RISK_FLAGS.CRITICAL;
  if (pct >= 0.65) return RISK_FLAGS.MONITOR;
  return RISK_FLAGS.SAFE;
}
