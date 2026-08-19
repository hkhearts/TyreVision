/**
 * app.js — Main Application Router & Shell Controller
 * Tire Vision — Fleet Tire Intelligence System
 */

import { fleet }       from './fleet.js';
import { seedFleetIfEmpty } from './db.js';

window.TV_API_BASE = 'http://localhost:5000/api';

class TireVisionApp {
  constructor() {
    this.currentPage = null;
    this.serverOnline = false;
    this._toastContainer = null;
  }

  async init() {
    // Register Service Worker
    if ('serviceWorker' in navigator) {
      try {
        const reg = await navigator.serviceWorker.register('/sw.js');
        console.log('[SW] Registered:', reg.scope);
      } catch(e) {
        console.warn('[SW] Registration failed:', e);
      }
    }

    // Init DB
    await seedFleetIfEmpty();

    // Check server
    await this._checkServer();
    setInterval(() => this._checkServer(), 30000);

    // Init fleet
    await fleet.init();

    // Setup navigation
    this._setupNavigation();
    this._setupMobileMenu();
    this._setupToasts();
    this._updateNavBadges();

    // Route to initial page
    this._route();
    window.addEventListener('hashchange', () => this._route());

    console.log('[App] Tire Vision initialized');
  }

  async _checkServer() {
    try {
      const resp = await fetch(`${window.TV_API_BASE}/health`, {
        signal: AbortSignal.timeout(2000)
      });
      const online = resp.ok;
      if (online !== this.serverOnline) {
        this.serverOnline = online;
        this._updateServerStatus(online);
      }
    } catch {
      if (this.serverOnline) {
        this.serverOnline = false;
        this._updateServerStatus(false);
      }
    }
  }

  _updateServerStatus(online) {
    const dot   = document.querySelector('.status-dot');
    const label = document.querySelector('.server-status-label');
    if (dot)   dot.className = `status-dot${online ? '' : ' offline'}`;
    if (label) label.textContent = online ? 'AI Server Online' : 'Offline Mode';
  }

  _route() {
    const hash = window.location.hash.replace('#', '') || 'dashboard';
    const routes = {
      'dashboard':      '/pages/dashboard.html',
      'inspect':        '/pages/inspect.html',
      'ar':             '/pages/ar-guide.html',
      'results':        '/pages/results.html',
      'fleet':          '/pages/fleet.html',
      'vehicle':        '/pages/vehicle-detail.html',
      'analytics':      '/pages/analytics.html',
      'alerts':         '/pages/alerts.html',
      'settings':       '/pages/settings.html',
    };
    const page = routes[hash.split('/')[0]] || routes['dashboard'];
    if (page !== this.currentPage) {
      this.currentPage = page;
      this._loadPage(page, hash);
    }
    this._updateActiveNav(hash.split('/')[0]);
  }

  async _loadPage(url, hash) {
    const main = document.getElementById('page-frame');
    if (!main) return;

    main.style.opacity = '0';
    main.style.transform = 'translateY(8px)';

    try {
      const resp = await fetch(url);
      const html = await resp.text();
      const parser = new DOMParser();
      const doc    = parser.parseFromString(html, 'text/html');

      // Extract body content and scripts
      const content = doc.querySelector('#page-content') || doc.body;
      main.innerHTML = content?.innerHTML || html;

      // Execute inline scripts
      main.querySelectorAll('script').forEach(s => {
        const ns = document.createElement('script');
        if (s.src) ns.src = s.src;
        else ns.textContent = s.textContent;
        ns.type = s.type || 'module';
        document.body.appendChild(ns);
      });

    } catch(e) {
      main.innerHTML = `<div style="padding:2rem;text-align:center;color:var(--text-muted)">
        <p style="font-size:3rem">🔧</p>
        <h3>Page loading...</h3>
        <p>Open <a href="${url}" style="color:var(--cyan)">${url}</a> directly</p>
      </div>`;
    }

    requestAnimationFrame(() => {
      main.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
      main.style.opacity    = '1';
      main.style.transform  = 'translateY(0)';
    });
  }

  _setupNavigation() {
    document.querySelectorAll('[data-nav]').forEach(el => {
      el.addEventListener('click', () => {
        const target = el.dataset.nav;
        window.location.hash = target;
      });
    });
  }

  _updateActiveNav(page) {
    document.querySelectorAll('.nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.nav === page || el.dataset.nav?.startsWith(page));
    });
  }

  _setupMobileMenu() {
    const btn     = document.querySelector('.mobile-menu-btn');
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');

    if (btn && sidebar) {
      btn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        overlay?.classList.toggle('open');
      });
      overlay?.addEventListener('click', () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('open');
      });
    }
  }

  _setupToasts() {
    if (!document.getElementById('toast-container')) {
      const c = document.createElement('div');
      c.id = 'toast-container';
      document.body.appendChild(c);
      this._toastContainer = c;
    } else {
      this._toastContainer = document.getElementById('toast-container');
    }
  }

  _updateNavBadges() {
    fleet.on('updated', (vehicles) => {
      const kpis = fleet.getFleetKPIs();
      const alertCount = kpis.riskCounts['DO-NOT-OPERATE'] + kpis.riskCounts['CRITICAL'];
      const badge = document.querySelector('[data-nav="alerts"] .nav-badge');
      if (badge) {
        badge.textContent = alertCount > 0 ? alertCount : '';
        badge.style.display = alertCount > 0 ? '' : 'none';
      }
    });
  }

  // ─── Global toast notification ─────────────────────────────
  static toast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const icons = { success: '✅', error: '🚫', warning: '⚠️', info: 'ℹ️' };
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
    container.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(100%)';
      el.style.transition = 'all 0.3s ease';
      setTimeout(() => el.remove(), 300);
    }, duration);
  }
}

// ─── Export global ──────────────────────────────────────────
window.TVApp = TireVisionApp;
window.toast = TireVisionApp.toast;

// Auto-init when DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new TireVisionApp().init());
} else {
  new TireVisionApp().init();
}
