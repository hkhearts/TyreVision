/**
 * db.js — IndexedDB wrapper for Tire Vision
 * Stores: inspections, fleet data, pending sync queue
 */

const DB_NAME = 'TireVisionDB';
const DB_VERSION = 2;

const STORES = {
  INSPECTIONS: 'inspections',
  FLEET:       'fleet',
  ALERTS:      'alerts',
  SYNC_QUEUE:  'syncQueue',
};

let _db = null;

async function openDB() {
  if (_db) return _db;
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = e => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORES.INSPECTIONS)) {
        const store = db.createObjectStore(STORES.INSPECTIONS, { keyPath: 'id', autoIncrement: true });
        store.createIndex('vehicleId', 'vehicleId', { unique: false });
        store.createIndex('tireId',    'tireId',    { unique: false });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.FLEET)) {
        db.createObjectStore(STORES.FLEET, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORES.ALERTS)) {
        const a = db.createObjectStore(STORES.ALERTS, { keyPath: 'id', autoIncrement: true });
        a.createIndex('vehicleId', 'vehicleId', { unique: false });
        a.createIndex('severity',  'severity',  { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.SYNC_QUEUE)) {
        db.createObjectStore(STORES.SYNC_QUEUE, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = e => { _db = e.target.result; resolve(_db); };
    req.onerror   = e => reject(e.target.error);
  });
}

function tx(storeName, mode = 'readonly') {
  return _db.transaction(storeName, mode).objectStore(storeName);
}

// ─── Generic CRUD ───────────────────────────────────────────
export async function put(storeName, record) {
  await openDB();
  return new Promise((res, rej) => {
    const req = tx(storeName, 'readwrite').put(record);
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}

export async function get(storeName, key) {
  await openDB();
  return new Promise((res, rej) => {
    const req = tx(storeName).get(key);
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}

export async function getAll(storeName) {
  await openDB();
  return new Promise((res, rej) => {
    const req = tx(storeName).getAll();
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}

export async function del(storeName, key) {
  await openDB();
  return new Promise((res, rej) => {
    const req = tx(storeName, 'readwrite').delete(key);
    req.onsuccess = () => res();
    req.onerror   = () => rej(req.error);
  });
}

export async function getByIndex(storeName, indexName, value) {
  await openDB();
  return new Promise((res, rej) => {
    const req = tx(storeName).index(indexName).getAll(value);
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}

// ─── Inspection-specific helpers ────────────────────────────
export async function saveInspection(inspection) {
  const record = { ...inspection, timestamp: Date.now(), synced: false };
  const id = await put(STORES.INSPECTIONS, record);
  // Add to sync queue
  await put(STORES.SYNC_QUEUE, { type: 'inspection', inspectionId: id, timestamp: Date.now() });
  return id;
}

export async function getInspectionsByVehicle(vehicleId) {
  return getByIndex(STORES.INSPECTIONS, 'vehicleId', vehicleId);
}

export async function getInspectionsByTire(tireId) {
  return getByIndex(STORES.INSPECTIONS, 'tireId', tireId);
}

export async function getAllInspections() {
  return getAll(STORES.INSPECTIONS);
}

// ─── Fleet data ──────────────────────────────────────────────
export async function saveFleet(fleetData) {
  await openDB();
  const vehicles = fleetData.vehicles || [];
  for (const v of vehicles) {
    await put(STORES.FLEET, v);
  }
}

export async function getFleetVehicles() {
  return getAll(STORES.FLEET);
}

// ─── Alerts ──────────────────────────────────────────────────
export async function saveAlert(alert) {
  return put(STORES.ALERTS, { ...alert, timestamp: Date.now(), resolved: false });
}

export async function getAllAlerts() {
  return getAll(STORES.ALERTS);
}

export async function resolveAlert(id) {
  const alert = await get(STORES.ALERTS, id);
  if (alert) await put(STORES.ALERTS, { ...alert, resolved: true, resolvedAt: Date.now() });
}

// ─── Seed fleet data from JSON ───────────────────────────────
export async function seedFleetIfEmpty() {
  await openDB();
  const existing = await getAll(STORES.FLEET);
  if (existing.length > 0) return;
  try {
    const res  = await fetch('/data/fleet-seed.json');
    const data = await res.json();
    await saveFleet(data);
    // Seed initial alerts
    const alerts = data.maintenanceSchedule || [];
    for (const m of alerts) {
      await saveAlert({
        vehicleId:  m.vehicleId,
        tireId:     m.tireId,
        type:       m.type,
        severity:   m.priority,
        title:      `${m.type.toUpperCase()} — ${m.tireId}`,
        description: m.notes,
        dueDate:    m.dueDate,
      });
    }
    console.log('[DB] Fleet data seeded');
  } catch (e) {
    console.warn('[DB] Could not seed fleet:', e);
  }
}

export { STORES };
