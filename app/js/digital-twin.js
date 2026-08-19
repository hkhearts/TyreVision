/**
 * digital-twin.js — Three.js 3D Tire Digital Twin
 * Tire Vision — Fleet Tire Intelligence System
 *
 * Creates a procedural 3D tire model with:
 * - PBR rubber material
 * - Zone-based heat map overlays
 * - Animated damage markers
 * - Touch/mouse orbit controls
 */

export class TireDigitalTwin {
  constructor(canvasEl, options = {}) {
    this.canvas   = canvasEl;
    this.options  = options;
    this.THREE    = null;
    this.scene    = null;
    this.camera   = null;
    this.renderer = null;
    this.tireMesh = null;
    this.treadMesh= null;
    this.sidewallMeshes = [];
    this.markers  = [];
    this._raf     = null;
    this._orbitStart = null;
    this._features = null;
    this._riskFlag = 'SAFE';
  }

  async init() {
    // Load Three.js from CDN
    if (!window.THREE) {
      await this._loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js');
    }
    this.THREE = window.THREE;
    this._setup();
    return this;
  }

  _loadScript(src) {
    return new Promise((res, rej) => {
      if (document.querySelector(`script[src="${src}"]`)) { res(); return; }
      const s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  _setup() {
    const T = this.THREE;
    const W = this.canvas.clientWidth  || 500;
    const H = this.canvas.clientHeight || 400;

    // Scene
    this.scene = new T.Scene();
    this.scene.background = new T.Color(0x0a0c14);
    this.scene.fog = new T.FogExp2(0x0a0c14, 0.08);

    // Camera
    this.camera = new T.PerspectiveCamera(45, W / H, 0.1, 100);
    this.camera.position.set(3.5, 1.5, 3.5);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new T.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setSize(W, H);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = T.PCFSoftShadowMap;

    // Lighting
    const ambient = new T.AmbientLight(0x334466, 0.8);
    this.scene.add(ambient);

    const keyLight = new T.DirectionalLight(0x00d4ff, 1.2);
    keyLight.position.set(5, 8, 5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width  = 1024;
    keyLight.shadow.mapSize.height = 1024;
    this.scene.add(keyLight);

    const fillLight = new T.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-3, 2, -3);
    this.scene.add(fillLight);

    const rimLight = new T.PointLight(0x00d4ff, 0.6, 10);
    rimLight.position.set(-2, -1, 2);
    this.scene.add(rimLight);

    // Build tire
    this._buildTire();

    // Ground reflection plane
    const planeMat = new T.MeshStandardMaterial({
      color: 0x111420,
      metalness: 0.1,
      roughness: 0.9,
      transparent: true,
      opacity: 0.4,
    });
    const plane = new T.Mesh(new T.PlaneGeometry(12, 12), planeMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -1.6;
    plane.receiveShadow = true;
    this.scene.add(plane);

    // Grid
    const grid = new T.GridHelper(10, 20, 0x1a2040, 0x111a30);
    grid.position.y = -1.6;
    this.scene.add(grid);

    // Setup controls
    this._setupControls();

    // Resize observer
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(this.canvas.parentElement || document.body);

    // Start loop
    this._animate();
  }

  _buildTire() {
    const T = this.THREE;
    const tireGroup = new T.Group();

    // ── Tire body (torus) ──────────────────────────────────
    const torusGeo = new T.TorusGeometry(1.0, 0.38, 48, 120);

    // PBR rubber material (dark, matte)
    const rubberMat = new T.MeshStandardMaterial({
      color:     new T.Color(0x1a1a1a),
      roughness: 0.92,
      metalness: 0.01,
      bumpScale: 0.012,
    });
    this.tireMesh = new T.Mesh(torusGeo, rubberMat);
    this.tireMesh.castShadow    = true;
    this.tireMesh.receiveShadow = true;
    this.tireMesh.rotation.x    = Math.PI / 2;
    tireGroup.add(this.tireMesh);

    // ── Tread pattern overlay (slightly larger torus) ──────
    const treadGeo = new T.TorusGeometry(1.0, 0.395, 48, 120);
    const treadCanvas = this._generateTreadTexture();
    const treadTex    = new T.CanvasTexture(treadCanvas);
    treadTex.wrapS = treadTex.wrapT = T.RepeatWrapping;
    treadTex.repeat.set(8, 1);

    const treadMat = new T.MeshStandardMaterial({
      map:         treadTex,
      roughness:   0.95,
      metalness:   0.0,
      transparent: true,
      opacity:     0.9,
    });
    this.treadMesh = new T.Mesh(treadGeo, treadMat);
    this.treadMesh.rotation.x = Math.PI / 2;
    tireGroup.add(this.treadMesh);

    // ── Rim (cylinder) ────────────────────────────────────
    const rimGeo = new T.CylinderGeometry(0.62, 0.62, 0.36, 48, 1, false);
    const rimMat = new T.MeshStandardMaterial({
      color:     new T.Color(0x888fa0),
      roughness: 0.3,
      metalness: 0.8,
    });
    const rim = new T.Mesh(rimGeo, rimMat);
    rim.castShadow = true;
    tireGroup.add(rim);

    // Rim spokes
    for (let i = 0; i < 5; i++) {
      const spokeGeo = new T.BoxGeometry(0.08, 0.34, 0.55);
      const spoke    = new T.Mesh(spokeGeo, rimMat);
      spoke.rotation.y = (i / 5) * Math.PI * 2;
      tireGroup.add(spoke);
    }

    // ── Valve stem ────────────────────────────────────────
    const valveGeo = new T.CylinderGeometry(0.015, 0.015, 0.12, 8);
    const valveMat = new T.MeshStandardMaterial({ color: 0x444, metalness: 0.9, roughness: 0.2 });
    const valve    = new T.Mesh(valveGeo, valveMat);
    valve.position.set(0.63, 0, 0);
    tireGroup.add(valve);

    // ── Zone highlight meshes (initially hidden) ──────────
    const zoneConfigs = [
      { name: 'tread',    r: 1.01, tub: 0.41, color: 0x00e676 },
      { name: 'sidewall', r: 0.90, tub: 0.30, color: 0xffcc00 },
      { name: 'shoulder', r: 0.96, tub: 0.36, color: 0xff6b00 },
    ];
    this._zoneMeshes = {};
    zoneConfigs.forEach(({ name, r, tub, color }) => {
      const geo = new T.TorusGeometry(r, tub, 24, 80);
      const mat = new T.MeshStandardMaterial({
        color: new T.Color(color),
        transparent: true,
        opacity: 0,
        emissive: new T.Color(color),
        emissiveIntensity: 0,
      });
      const mesh = new T.Mesh(geo, mat);
      mesh.rotation.x = Math.PI / 2;
      mesh.name = `zone-${name}`;
      tireGroup.add(mesh);
      this._zoneMeshes[name] = { mesh, mat, color };
    });

    this.tireGroup = tireGroup;
    this.scene.add(tireGroup);
  }

  _generateTreadTexture() {
    const size = 512;
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');

    // Base rubber color
    ctx.fillStyle = '#1c1c1c';
    ctx.fillRect(0, 0, size, size);

    // Tread grooves (longitudinal)
    const grooveColor = '#0a0a0a';
    ctx.fillStyle = grooveColor;
    const grooves = [0.18, 0.36, 0.64, 0.82];
    grooves.forEach(pct => {
      ctx.fillRect(0, pct * size - 6, size, 12);
    });

    // Block sipes
    ctx.fillStyle = grooveColor;
    for (let x = 0; x < size; x += 48) {
      ctx.fillRect(x, 0, 4, size);
    }

    // Wear indicators (red stripe simulation)
    ctx.fillStyle = 'rgba(180, 30, 30, 0.6)';
    ctx.fillRect(0, 0.5 * size - 2, size, 4);

    return c;
  }

  // ─── Apply heat map based on features ─────────────────────
  applyHeatMap(features, riskFlag = 'SAFE') {
    this._features = features;
    this._riskFlag = riskFlag;

    const zones = {
      tread:    1.0 - (features.tread_depth_mm || 8) / 24,
      sidewall: (features.sidewall_crack_density || 0) * 0.8 + (features.bulge_confidence || 0) * 0.2,
      shoulder: (features.shoulder_wear_idx || 0) * 0.7 + (features.heat_damage_idx || 0) * 0.3,
    };

    Object.entries(zones).forEach(([name, severity]) => {
      const zone = this._zoneMeshes[name];
      if (!zone) return;
      const { mat } = zone;
      const color = this._severityToColor(severity, riskFlag);
      mat.color.set(color);
      mat.emissive.set(color);
      mat.opacity = Math.max(0, Math.min(0.55, severity * 0.7));
      mat.emissiveIntensity = severity * 0.8;
    });

    // Update tread mesh color
    if (this.tireMesh) {
      const wearSeverity = zones.tread;
      const treadColor = this._severityToColor(wearSeverity, riskFlag);
      this.tireMesh.material.color.lerp(new this.THREE.Color(treadColor), 0.3);
    }

    // Add damage markers
    this._clearMarkers();
    if (features.bulge_detected) this._addMarker([0, 1.05, 0.38], '#ff1744', '⚠ BULGE');
    if ((features.cut_puncture_count || 0) > 0) this._addMarker([1.05, 0, 0.38], '#ff6b00', `✂ ${features.cut_puncture_count} CUT`);
    if ((features.sidewall_crack_density || 0) > 0.4) this._addMarker([-1.0, 0, 0.3], '#ffcc00', '〰 CRACK');
  }

  _severityToColor(severity, riskFlag) {
    if (riskFlag === 'DO-NOT-OPERATE') return '#ff1744';
    if (riskFlag === 'CRITICAL')       return '#ff6b00';
    if (severity > 0.7) return '#ff6b00';
    if (severity > 0.4) return '#ffcc00';
    return '#00e676';
  }

  _addMarker(position, color, label) {
    const T = this.THREE;
    const geo  = new T.SphereGeometry(0.04, 8, 8);
    const mat  = new T.MeshStandardMaterial({
      color:    new T.Color(color),
      emissive: new T.Color(color),
      emissiveIntensity: 1.5,
      transparent: true,
      opacity: 0.9,
    });
    const mesh = new T.Mesh(geo, mat);
    mesh.position.set(...position);
    mesh._label = label;
    mesh._color = color;
    mesh._baseY = position[1];
    this.tireGroup.add(mesh);
    this.markers.push(mesh);
  }

  _clearMarkers() {
    this.markers.forEach(m => this.tireGroup.remove(m));
    this.markers = [];
  }

  // ─── Animate ──────────────────────────────────────────────
  _animate() {
    this._raf = requestAnimationFrame(() => this._animate());
    const t = Date.now() / 1000;

    // Slow tire rotation
    if (this.tireGroup) {
      this.tireGroup.rotation.y += 0.003;
    }

    // Pulsing markers
    this.markers.forEach((m, i) => {
      m.scale.setScalar(1 + 0.2 * Math.sin(t * 3 + i));
      m.material.emissiveIntensity = 0.8 + 0.7 * Math.sin(t * 2 + i);
    });

    // Pulsing zone highlights
    if (this._features && this._riskFlag === 'DO-NOT-OPERATE') {
      Object.values(this._zoneMeshes).forEach(z => {
        if (z.mat.opacity > 0) {
          z.mat.emissiveIntensity = 0.4 + 0.6 * Math.sin(t * 3);
        }
      });
    }

    this.renderer.render(this.scene, this.camera);
  }

  // ─── Controls (touch + mouse orbit) ───────────────────────
  _setupControls() {
    const el = this.canvas;
    let lastX = 0, lastY = 0, isDragging = false;
    let dist = 3.5;

    const onStart = (x, y) => { lastX = x; lastY = y; isDragging = true; };
    const onMove  = (x, y) => {
      if (!isDragging) return;
      const dx = (x - lastX) * 0.01;
      const dy = (y - lastY) * 0.01;
      const { position } = this.camera;
      const radius = position.length();
      const theta  = Math.atan2(position.x, position.z) + dx;
      const phi    = Math.max(0.1, Math.min(Math.PI - 0.1, Math.acos(position.y / radius) + dy));
      position.x = radius * Math.sin(phi) * Math.sin(theta);
      position.y = radius * Math.cos(phi);
      position.z = radius * Math.sin(phi) * Math.cos(theta);
      this.camera.lookAt(0, 0, 0);
      lastX = x; lastY = y;
    };
    const onEnd = () => { isDragging = false; };

    el.addEventListener('mousedown',  e => onStart(e.clientX, e.clientY));
    el.addEventListener('mousemove',  e => onMove(e.clientX, e.clientY));
    el.addEventListener('mouseup',    onEnd);
    el.addEventListener('mouseleave', onEnd);

    el.addEventListener('touchstart', e => { const t = e.touches[0]; onStart(t.clientX, t.clientY); }, { passive: true });
    el.addEventListener('touchmove',  e => { const t = e.touches[0]; onMove(t.clientX, t.clientY); },  { passive: true });
    el.addEventListener('touchend',   onEnd, { passive: true });

    el.addEventListener('wheel', e => {
      dist = Math.max(2, Math.min(8, dist + e.deltaY * 0.005));
      const { position } = this.camera;
      const norm = position.clone().normalize().multiplyScalar(dist);
      position.copy(norm);
      e.preventDefault();
    }, { passive: false });
  }

  _resize() {
    if (!this.renderer) return;
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    this.camera.aspect = W / H;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(W, H);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this.renderer) this.renderer.dispose();
  }

  // ─── Animate to specific view ──────────────────────────────
  focusZone(zoneName) {
    const positions = {
      tread:    [0, 2.5, 3],
      sidewall: [3, 0.5, 2],
      shoulder: [2, 1.5, 2.5],
      valve:    [2.5, 0, 1],
    };
    const pos = positions[zoneName] || [3.5, 1.5, 3.5];
    this.camera.position.set(...pos);
    this.camera.lookAt(0, 0, 0);
  }

  // Reset to default view
  resetView() {
    this.camera.position.set(3.5, 1.5, 3.5);
    this.camera.lookAt(0, 0, 0);
    if (this.tireGroup) this.tireGroup.rotation.y = 0;
  }
}
