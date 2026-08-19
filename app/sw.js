/**
 * Tire Vision Service Worker
 * Handles: caching, offline fallback, background sync
 */

const CACHE_VERSION = 'tv-v1.2';
const STATIC_CACHE  = `${CACHE_VERSION}-static`;
const DATA_CACHE    = `${CACHE_VERSION}-data`;

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/css/main.css',
  '/js/app.js',
  '/js/db.js',
  '/js/cv-engine.js',
  '/js/risk-engine.js',
  '/js/reliability.js',
  '/js/dot-ocr.js',
  '/js/ar-overlay.js',
  '/js/digital-twin.js',
  '/js/fleet.js',
  '/js/alerts.js',
  '/js/charts.js',
  '/pages/dashboard.html',
  '/pages/inspect.html',
  '/pages/ar-guide.html',
  '/pages/results.html',
  '/pages/fleet.html',
  '/pages/vehicle-detail.html',
  '/pages/analytics.html',
  '/pages/alerts.html',
  '/pages/settings.html',
  '/data/fleet-seed.json',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js',
];

// ─── Install ───────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then(cache => {
      // Cache critical assets, skip failures gracefully
      return Promise.allSettled(
        STATIC_ASSETS.map(url => cache.add(url).catch(() => console.warn('[SW] Failed to cache:', url)))
      );
    }).then(() => self.skipWaiting())
  );
});

// ─── Activate ──────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== STATIC_CACHE && k !== DATA_CACHE).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// ─── Fetch Strategy ────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // API calls: network-first, fallback to cached
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(DATA_CACHE).then(cache => cache.put(request, clone));
          }
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Static assets: cache-first
  event.respondWith(
    caches.match(request).then(cached => {
      if (cached) return cached;
      return fetch(request).then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(STATIC_CACHE).then(cache => cache.put(request, clone));
        }
        return response;
      }).catch(() => {
        // Offline fallback for navigation requests
        if (request.mode === 'navigate') {
          return caches.match('/index.html');
        }
      });
    })
  );
});

// ─── Background Sync ───────────────────────────────────────
self.addEventListener('sync', event => {
  if (event.tag === 'sync-inspections') {
    event.waitUntil(syncPendingInspections());
  }
});

async function syncPendingInspections() {
  // Would retrieve from IndexedDB and POST to server
  console.log('[SW] Syncing pending inspections...');
  const clients = await self.clients.matchAll();
  clients.forEach(client => client.postMessage({ type: 'SYNC_COMPLETE' }));
}

// ─── Push Notifications ─────────────────────────────────────
self.addEventListener('push', event => {
  const data = event.data?.json() || { title: 'Tire Vision Alert', body: 'Fleet alert requires attention' };
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/assets/icons/icon-192.png',
      badge: '/assets/icons/icon-192.png',
      tag: 'tv-alert',
      data: data.url,
    })
  );
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(clients.openWindow(event.notification.data || '/'));
});

// Message handler
self.addEventListener('message', event => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting();
});
