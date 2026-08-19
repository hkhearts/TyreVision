/**
 * fleet.js — Fleet Intelligence & Management Engine
 * Tire Vision — Fleet Tire Intelligence System
 */

import { getFleetVehicles, saveFleet, seedFleetIfEmpty, getAllInspections } from './db.js';
import { riskFromMileage, classifyRisk, RISK_CSS_CLASS } from './risk-engine.js';
import { calcShelfLife, estimateTreadFromMileage, POSITION_BASELINES } from './reliability.js';

export class FleetEngine {
  constructor() {
    this.vehicles    = [];
    this.fleetData   = null;
    this.inspections = [];
    this._listeners  = new Map();
    this._serverBase = window.TV_API_BASE || 'http://localhost:5000/api';
  }

  // ─── Init ──────────────────────────────────────────────────
  async init() {
    await seedFleetIfEmpty();
    await this.refresh();
    return this;
  }

  async refresh() {
    // Try server, fallback to IndexedDB
    try {
      const resp = await fetch(`${this._serverBase}/fleet`, { signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        this.fleetData = await resp.json();
        if (this.fleetData?.vehicles) {
          await saveFleet(this.fleetData);
        }
      }
    } catch(e) {
      console.warn('[Fleet] Server unavailable, using local data');
    }

    this.vehicles    = await getFleetVehicles();
    this.inspections = await getAllInspections();
    this._enrichVehicles();
    this._emit('updated', this.vehicles);
    return this;
  }

  // ─── Enrich vehicle data with derived metrics ──────────────
  _enrichVehicles() {
    const now = Date.now();
    this.vehicles = this.vehicles.map(v => {
      const tires  = v.tires || [];
      const inspByTire = {};
      this.inspections.forEach(insp => {
        if (!inspByTire[insp.tireId] || insp.timestamp > inspByTire[insp.tireId].timestamp) {
          inspByTire[insp.tireId] = insp;
        }
      });

      const enrichedTires = tires.map(t => {
        const pos  = this._positionKey(t.position);
        const insp = inspByTire[t.id];
        let riskFlag, reliability, shelfLife;

        if (insp?.result) {
          riskFlag    = insp.result.risk_flag || riskFromMileage(t.mileageOnTire, pos);
          reliability = insp.result.reliability_score;
          shelfLife   = insp.result.shelf_life_pct;
        } else {
          riskFlag    = riskFromMileage(t.mileageOnTire, pos);
          const depth = estimateTreadFromMileage(t.mileageOnTire, pos);
          const feats = { tread_depth_mm: depth, tire_age_months: this._dotToAgeMonths(t.dotCode) };
          reliability = null;
          const sl = calcShelfLife(feats, t.mileageOnTire, pos);
          shelfLife = sl.shelfLifePct;
        }

        return { ...t, riskFlag, reliability, shelfLife, lastInspection: insp };
      });

      // Vehicle-level aggregate risk
      const worstRisk = enrichedTires.reduce((worst, tire) => {
        const order = { 'SAFE': 0, 'MONITOR': 1, 'CRITICAL': 2, 'DO-NOT-OPERATE': 3 };
        return order[tire.riskFlag] > order[worst] ? tire.riskFlag : worst;
      }, 'SAFE');

      const avgShelfLife = enrichedTires.length
        ? Math.round(enrichedTires.reduce((s, t) => s + (t.shelfLife || 0), 0) / enrichedTires.length)
        : null;

      // Driving behavior score (0–100)
      const telem = v.telematics || {};
      const drivingScore = this._calcDrivingScore(telem);

      return { ...v, tires: enrichedTires, worstRisk, avgShelfLife, drivingScore };
    });
  }

  _positionKey(posLabel = '') {
    const l = posLabel.toLowerCase();
    if (l.includes('steer') || l.includes('front')) return 'steer';
    if (l.includes('trailer'))                        return 'trailer';
    return 'drive';
  }

  _dotToAgeMonths(dotCode = '') {
    const m = (dotCode || '').match(/(\d{2})(\d{2})/);
    if (!m) return 36;
    const ww = parseInt(m[1], 10);
    const yy = parseInt(m[2], 10);
    const year = yy <= 30 ? 2000 + yy : 1900 + yy;
    const mfg  = new Date(year, 0, 1);
    mfg.setDate(mfg.getDate() + (ww - 1) * 7);
    return Math.max(0, (Date.now() - mfg.getTime()) / (1000 * 60 * 60 * 24 * 30.44));
  }

  // ─── Driving Behavior Score ────────────────────────────────
  _calcDrivingScore(telem) {
    if (!telem) return 100;
    const harshBrake  = Math.min(1, (telem.harshBrakeEvents  || 0) / 30) * 25;
    const harshAccel  = Math.min(1, (telem.harshAccelEvents  || 0) / 25) * 20;
    const cornering   = Math.min(1, (telem.corneringEvents   || 0) / 20) * 15;
    const idle        = Math.min(1, (telem.idleTimePct       || 0) / 30) * 10;
    const fuelDemerit = Math.max(0, (10 - (telem.fuelEfficiency || 10)) / 10) * 10;
    const engineTemp  = Math.min(1, Math.max(0, ((telem.engineTempC || 90) - 95) / 15)) * 20;
    const demerits    = harshBrake + harshAccel + cornering + idle + fuelDemerit + engineTemp;
    return Math.round(Math.max(0, 100 - demerits));
  }

  // ─── Fleet KPIs ────────────────────────────────────────────
  getFleetKPIs() {
    const counts = { SAFE: 0, MONITOR: 0, CRITICAL: 0, 'DO-NOT-OPERATE': 0 };
    let totalTires = 0;

    this.vehicles.forEach(v => {
      (v.tires || []).forEach(t => {
        counts[t.riskFlag] = (counts[t.riskFlag] || 0) + 1;
        totalTires++;
      });
    });

    const criticalVehicles = this.vehicles.filter(v => v.worstRisk === 'CRITICAL' || v.worstRisk === 'DO-NOT-OPERATE');
    const avgDrivingScore  = this.vehicles.length
      ? Math.round(this.vehicles.reduce((s, v) => s + (v.drivingScore || 100), 0) / this.vehicles.length)
      : 100;

    return {
      totalVehicles:    this.vehicles.length,
      totalTires,
      riskCounts:       counts,
      criticalVehicles: criticalVehicles.length,
      avgDrivingScore,
      safetyRate:       totalTires ? Math.round((counts.SAFE / totalTires) * 100) : 100,
    };
  }

  // ─── Maintenance Schedule Generation ──────────────────────
  getMaintenanceSchedule() {
    const schedule = [];
    this.vehicles.forEach(v => {
      (v.tires || []).forEach(t => {
        if (t.riskFlag === 'DO-NOT-OPERATE') {
          schedule.push({
            vehicleId:   v.id,
            vehicleName: `${v.make} ${v.model} — ${v.regNo}`,
            tireId:      t.id,
            tireLabel:   t.label,
            type:        'replacement',
            priority:    'do-not-operate',
            dueDate:     'IMMEDIATE',
            notes:       `Risk: DO-NOT-OPERATE — ${t.riskFlag}`,
          });
        } else if (t.riskFlag === 'CRITICAL') {
          const due = new Date();
          due.setDate(due.getDate() + 2);
          schedule.push({
            vehicleId:   v.id,
            vehicleName: `${v.make} ${v.model} — ${v.regNo}`,
            tireId:      t.id,
            tireLabel:   t.label,
            type:        'replacement',
            priority:    'critical',
            dueDate:     due.toISOString().slice(0, 10),
            notes:       'Critical wear/damage — replace within 48 hours',
          });
        } else if (t.riskFlag === 'MONITOR') {
          const due = new Date();
          due.setDate(due.getDate() + 14);
          schedule.push({
            vehicleId:   v.id,
            vehicleName: `${v.make} ${v.model} — ${v.regNo}`,
            tireId:      t.id,
            tireLabel:   t.label,
            type:        'inspection',
            priority:    'monitor',
            dueDate:     due.toISOString().slice(0, 10),
            notes:       'Monitor condition — re-inspect in 2 weeks',
          });
        }
      });
    });

    return schedule.sort((a, b) => {
      const order = { 'do-not-operate': 0, 'critical': 1, 'monitor': 2 };
      return (order[a.priority] || 9) - (order[b.priority] || 9);
    });
  }

  // ─── Active Alerts ─────────────────────────────────────────
  getActiveAlerts() {
    const alerts = [];
    this.vehicles.forEach(v => {
      (v.tires || []).forEach(t => {
        if (t.riskFlag === 'DO-NOT-OPERATE') {
          alerts.push({
            id:       `${v.id}-${t.id}-dno`,
            type:     'DO-NOT-OPERATE',
            severity: 'dno',
            icon:     '🚫',
            title:    `DO NOT OPERATE — ${v.make} ${v.model}`,
            desc:     `Tire ${t.label} requires immediate replacement. Vehicle must not be operated.`,
            vehicle:  v,
            tire:     t,
            time:     'NOW',
          });
        } else if (t.riskFlag === 'CRITICAL') {
          alerts.push({
            id:       `${v.id}-${t.id}-crit`,
            type:     'CRITICAL',
            severity: 'critical',
            icon:     '⚠️',
            title:    `Critical Tire — ${v.make} ${v.model}`,
            desc:     `Tire ${t.label} is in critical condition. Replace within 48 hours.`,
            vehicle:  v,
            tire:     t,
            time:     '<48h',
          });
        } else if (t.riskFlag === 'MONITOR') {
          alerts.push({
            id:       `${v.id}-${t.id}-mon`,
            type:     'MONITOR',
            severity: 'monitor',
            icon:     '👁️',
            title:    `Monitor — ${v.make} ${v.model}`,
            desc:     `Tire ${t.label} requires monitoring. Schedule re-inspection.`,
            vehicle:  v,
            tire:     t,
            time:     '2 weeks',
          });
        }
      });

      // Driving behavior alerts
      if ((v.drivingScore || 100) < 60) {
        alerts.push({
          id:       `${v.id}-driving`,
          type:     'DRIVING',
          severity: 'monitor',
          icon:     '🚗',
          title:    `Aggressive Driving — ${v.driverName || v.id}`,
          desc:     `Driving score: ${v.drivingScore}/100. Harsh braking/acceleration detected.`,
          vehicle:  v,
          time:     'Ongoing',
        });
      }
    });

    return alerts.sort((a, b) => {
      const order = { dno: 0, critical: 1, monitor: 2 };
      return (order[a.severity] || 9) - (order[b.severity] || 9);
    });
  }

  // ─── Telematics summary ────────────────────────────────────
  getTelematicsSummary() {
    if (!this.vehicles.length) return null;
    const all = this.vehicles.map(v => v.telematics || {});
    const avg = (key) => Math.round(all.reduce((s, t) => s + (t[key] || 0), 0) / all.length * 10) / 10;
    return {
      avgSpeed:         avg('avgSpeed'),
      totalHarshBrakes: all.reduce((s, t) => s + (t.harshBrakeEvents || 0), 0),
      totalHarshAccels: all.reduce((s, t) => s + (t.harshAccelEvents  || 0), 0),
      avgFuelEfficiency: avg('fuelEfficiency'),
      avgEngineTemp:    avg('engineTempC'),
      avgIdleTime:      avg('idleTimePct'),
    };
  }

  // ─── Wear Rate Trends (for analytics chart) ────────────────
  getWearRateTrends() {
    const mileagePoints = [0, 10000, 20000, 30000, 40000, 50000, 60000, 75000, 100000];
    const positions     = ['steer', 'drive', 'trailer'];
    return positions.reduce((acc, pos) => {
      const bl = POSITION_BASELINES[pos];
      acc[pos] = mileagePoints.map(km => {
        const worn  = (km / 1000) * bl.wearRate;
        const depth = Math.max(0, bl.newDepth - worn);
        return { km, depth: Math.round(depth * 100) / 100 };
      });
      return acc;
    }, {});
  }

  // ─── Get vehicle by ID ─────────────────────────────────────
  getVehicle(id) { return this.vehicles.find(v => v.id === id); }

  // ─── Event emitter ─────────────────────────────────────────
  on(event, cb)  { this._listeners.set(event, [...(this._listeners.get(event) || []), cb]); }
  _emit(event, data) { (this._listeners.get(event) || []).forEach(cb => cb(data)); }
}

// Singleton
export const fleet = new FleetEngine();
