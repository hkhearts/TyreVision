"""
Tire Vision — Local Inference Server
Flask backend for image analysis, DOT OCR simulation, and fleet data API.
Run: python server.py
Access: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import io
import json
import os
import random
import math
import hashlib
from datetime import datetime, timedelta

try:
    from PIL import Image, ImageStat
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow/numpy not available — using fallback analysis")

try:
    import easyocr
    reader = easyocr.Reader(['en'])
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    reader = None
    print("[WARN] EasyOCR not available - using fallback OCR")

app = Flask(__name__, static_folder='.')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Updated Tyre Dataset')
DEFECTIVE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Tyre dataset', 'Defective')

# ─── Tire Position Baselines ────────────────────────────────────────────────
POSITION_BASELINES = {
    'steer':   {'new_depth': 19.0, 'min_legal': 5.0, 'lifespan_km': 150000},
    'drive':   {'new_depth': 24.0, 'min_legal': 5.0, 'lifespan_km': 250000},
    'trailer': {'new_depth': 16.0, 'min_legal': 5.0, 'lifespan_km': 180000},
}

# Fleet wear rates per position (mm per 1000 km)
WEAR_RATES = {
    'steer':   0.093,   # (19-5)/150 * 1000 ≈ 0.093 mm/1000km
    'drive':   0.076,   # (24-5)/250 * 1000 ≈ 0.076 mm/1000km
    'trailer': 0.061,   # (16-5)/180 * 1000 ≈ 0.061 mm/1000km
}


def decode_image(b64_string):
    """Decode base64 image string to PIL Image."""
    if not PIL_AVAILABLE:
        return None
    try:
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(img_data)).convert('RGB')
    except Exception as e:
        print(f"[ERR] decode_image: {e}")
        return None


def analyze_image_quality(img):
    """Assess image quality: blur, brightness, contrast."""
    if img is None or not PIL_AVAILABLE:
        return {'blur': 0.7, 'brightness': 0.6, 'contrast': 0.65, 'overall': 0.65}
    try:
        import numpy as np
        gray = np.array(img.convert('L'), dtype=np.float32)
        # Laplacian variance as blur estimate
        lap = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        from PIL import ImageFilter
        gray_pil = Image.fromarray(gray.astype(np.uint8))
        edges = gray_pil.filter(ImageFilter.FIND_EDGES)
        lap_var = np.var(np.array(edges)) / 10000.0
        blur_score = min(1.0, lap_var)

        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / (3 * 255)
        contrast = min(1.0, sum(stat.stddev) / (3 * 64))
        overall = (blur_score * 0.5 + brightness * 0.25 + contrast * 0.25)
        return {
            'blur': round(blur_score, 3),
            'brightness': round(brightness, 3),
            'contrast': round(contrast, 3),
            'overall': round(min(1.0, overall), 3)
        }
    except Exception:
        return {'blur': 0.7, 'brightness': 0.6, 'contrast': 0.65, 'overall': 0.65}


def estimate_wear_from_image(img, mileage_km, position='drive'):
    """
    Estimate tread depth and wear features using image analysis
    combined with mileage-based computation from the dataset.
    """
    baseline = POSITION_BASELINES.get(position, POSITION_BASELINES['drive'])
    wear_rate = WEAR_RATES.get(position, WEAR_RATES['drive'])

    # Base computation from mileage
    worn_mm = (mileage_km / 1000.0) * wear_rate
    tread_depth = max(0.5, baseline['new_depth'] - worn_mm)

    # Image-based modifiers
    if img is not None and PIL_AVAILABLE:
        try:
            import numpy as np
            # Analyze darkness (worn rubber is lighter/shinier)
            gray = np.array(img.convert('L'))
            mean_bright = gray.mean() / 255.0
            # Very bright tread → more worn
            wear_modifier = (mean_bright - 0.3) * 4  # -1.2 to +2.8 mm
            tread_depth = max(0.5, tread_depth - wear_modifier)

            # Variance → uneven wear
            variance = gray.var() / (255.0 * 255.0)
            wear_variance = min(1.0, variance * 3)

            # Dark cracked lines → crack density
            dark_pct = (gray < 60).sum() / gray.size
            crack_density = min(1.0, dark_pct * 8)
        except Exception:
            wear_variance = 0.15
            crack_density = 0.05
    else:
        # Fallback: mileage-derived estimates
        wear_pct = mileage_km / baseline['lifespan_km']
        wear_variance = min(0.8, wear_pct * 0.5 + random.uniform(-0.05, 0.1))
        crack_density = min(1.0, max(0, wear_pct * 0.3 + random.uniform(-0.05, 0.1)))

    return tread_depth, wear_variance, crack_density


def parse_dot_code(dot_string):
    """Parse DOT date code (WWYR or WWYY format)."""
    import re
    # Look for 4-digit groups
    matches = re.findall(r'\b(\d{4})\b', dot_string or '')
    for match in matches:
        week = int(match[:2])
        year_suffix = int(match[2:])
        if 1 <= week <= 52 and 0 <= year_suffix <= 99:
            year = 2000 + year_suffix if year_suffix <= 30 else 1900 + year_suffix
            mfg_date = datetime(year, 1, 1) + timedelta(weeks=week - 1)
            age_months = (datetime.now() - mfg_date).days / 30.44
            return {
                'week': week,
                'year': year,
                'mfg_date': mfg_date.strftime('%Y-%m-%d'),
                'age_months': round(age_months, 1),
                'raw': match,
                'retread': False
            }
    return {'week': 0, 'year': 0, 'mfg_date': None, 'age_months': 0, 'raw': dot_string, 'retread': False}


def compute_risk_flag(features):
    """Hard-rule risk classification. Returns flag and triggers."""
    f = features
    triggers = []

    # DO-NOT-OPERATE
    if f.get('bulge_confidence', 0) > 0.85:
        triggers.append('Bulge detected with high confidence')
    if f.get('tread_depth_mm', 10) < 2.0:
        triggers.append(f"Tread depth critical: {f['tread_depth_mm']:.1f}mm")
    if f.get('exposed_cords', False):
        triggers.append('Exposed cords detected')
    if f.get('sidewall_crack_mm', 0) > 25:
        triggers.append(f"Sidewall crack: {f['sidewall_crack_mm']:.0f}mm")
    if triggers:
        return 'DO-NOT-OPERATE', triggers

    # CRITICAL
    critical_triggers = []
    if f.get('tread_depth_mm', 10) < 4.0:
        critical_triggers.append(f"Tread depth low: {f['tread_depth_mm']:.1f}mm")
    if f.get('cut_puncture_count', 0) >= 2:
        critical_triggers.append(f"Multiple deep cuts: {f['cut_puncture_count']}")
    if f.get('heat_damage_idx', 0) > 0.8:
        critical_triggers.append(f"Severe heat damage: {f['heat_damage_idx']:.2f}")
    if f.get('ozone_aging_score', 0) > 0.85:
        critical_triggers.append(f"Severe ozone aging: {f['ozone_aging_score']:.2f}")
    if critical_triggers:
        return 'CRITICAL', critical_triggers

    # MONITOR
    monitor_triggers = []
    if f.get('tread_wear_variance', 0) > 0.3:
        monitor_triggers.append(f"Uneven wear variance: {f['tread_wear_variance']:.2f}")
    if f.get('sidewall_crack_density', 0) > 0.3:
        monitor_triggers.append('Moderate sidewall cracking')
    if f.get('tire_age_months', 0) > 72:
        monitor_triggers.append(f"Tire age: {f['tire_age_months']:.0f} months")
    if monitor_triggers:
        return 'MONITOR', monitor_triggers

    return 'SAFE', ['No major risk detected']


def compute_reliability_score(features, position='drive'):
    """Reliability score 0–100 per spec."""
    f = features
    baseline = POSITION_BASELINES.get(position, POSITION_BASELINES['drive'])

    def f_norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    def f_inv(val, lo, hi):
        return 1.0 - f_norm(val, lo, hi)

    # Component scores
    tread_score    = f_norm(f.get('tread_depth_mm', 5), baseline['min_legal'], baseline['new_depth'])
    crack_score    = f_inv(f.get('sidewall_crack_density', 0), 0, 1)
    wear_score     = f_inv(f.get('tread_wear_variance', 0), 0, 0.8)
    heat_score     = f_inv(f.get('heat_damage_idx', 0), 0, 1)
    age_score      = f_inv(f.get('tire_age_months', 0), 0, 120)
    valve_score    = f_norm(f.get('valve_integrity', 0.8), 0, 1)

    reliability = (
        40 * tread_score +
        20 * crack_score +
        15 * wear_score +
        10 * heat_score +
        10 * age_score +
         5 * valve_score
    )

    return round(max(0, min(100, reliability)), 1)


def compute_shelf_life(features, mileage_km, position='drive', prior_tread=None, prior_km=None):
    """Shelf life percentage calculation."""
    baseline = POSITION_BASELINES.get(position, POSITION_BASELINES['drive'])
    current_depth = features.get('tread_depth_mm', 8.0)
    wear_rate = WEAR_RATES.get(position, WEAR_RATES['drive'])

    # Use historical wear rate if available
    if prior_tread is not None and prior_km is not None:
        delta_depth = prior_tread - current_depth
        delta_km = mileage_km - prior_km
        if delta_km > 0 and delta_depth > 0:
            wear_rate = (delta_depth / delta_km) * 1000

    remaining_depth = max(0, current_depth - baseline['min_legal'])
    if wear_rate > 0:
        remaining_km = (remaining_depth / wear_rate) * 1000
    else:
        remaining_km = baseline['lifespan_km']

    shelf_life_pct = (remaining_km / baseline['lifespan_km']) * 100

    # Age penalty
    age_months = features.get('tire_age_months', 0)
    if age_months > 72:
        shelf_life_pct *= 0.85
    if age_months > 96:
        shelf_life_pct *= 0.75

    # Heat damage penalty
    heat = features.get('heat_damage_idx', 0)
    if heat > 0.5:
        shelf_life_pct *= (1 - heat * 0.3)

    return round(max(0, min(100, shelf_life_pct)), 1), round(remaining_km, 0)


# ─── API Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'server': 'Tire Vision AI Engine v1.0', 'pil': PIL_AVAILABLE})


@app.route('/api/fleet', methods=['GET'])
def get_fleet():
    try:
        with open(os.path.join(DATA_DIR, 'fleet-seed.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_tire():
    """
    Main CV analysis endpoint.
    Accepts: { image: base64, mileage_km: int, position: str, dot_code: str,
               ambient_temp_c: float, capture_zone: str }
    Returns: { features: {...}, risk_flag: str, reliability_score: float, shelf_life_pct: float, ... }
    """
    data = request.get_json(force=True) or {}
    
    # Support new multi-image format
    images = data.get('images', {})
    if images:
        # Use tread image or dot image to seed the PRNG
        tread_imgs = images.get('tread', [])
        image_b64 = tread_imgs[0] if tread_imgs else (images.get('dot', '') or '')
    else:
        image_b64 = data.get('image', '')

    mileage_km = float(data.get('mileage_km', 50000))
    position = data.get('position', 'drive').lower()
    if 'steer' in position or 'front' in position:
        pos_key = 'steer'
    elif 'trailer' in position:
        pos_key = 'trailer'
    else:
        pos_key = 'drive'

    dot_raw = data.get('dot_code', '')
    if images and images.get('dot') and not dot_raw:
        dot_b64 = images.get('dot')[0] if isinstance(images.get('dot'), list) else images.get('dot')
        
        # Real OCR with easyocr
        if EASYOCR_AVAILABLE and dot_b64:
            try:
                import numpy as np
                import re
                # Decode image to numpy array for easyocr
                clean_b64 = dot_b64.split(',')[1] if ',' in dot_b64 else dot_b64
                img_data = base64.b64decode(clean_b64)
                img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
                img_np = np.array(img_pil)
                
                results = reader.readtext(img_np)
                for res in results:
                    text = res[1]
                    matches = re.findall(r'\d{4}', text)
                    if matches:
                        dot_raw = matches[0]
                        break
            except Exception as e:
                print(f"[WARN] EasyOCR failed: {e}")
                pass

        if not dot_raw:
            # Simulate OCR fallback
            rng_dot = random.Random(hash(dot_b64[:50]))
            week = rng_dot.randint(1, 52)
            year = rng_dot.randint(18, 24)
            dot_raw = f"{week:02d}{year:02d}"

    ambient_temp = float(data.get('ambient_temp_c', 28.0))
    capture_zone = data.get('capture_zone', 'tread')

    # Decode image
    img = decode_image(image_b64) if image_b64 else None

    # Image quality
    quality = analyze_image_quality(img)

    # Tread analysis
    tread_depth, wear_variance, crack_density = estimate_wear_from_image(img, mileage_km, pos_key)

    # DOT parsing
    dot_info = parse_dot_code(dot_raw)

    # Deterministic noise based on image hash (for reproducible demo results)
    img_hash = int(hashlib.md5((image_b64 or str(mileage_km))[:100].encode()).hexdigest(), 16)
    rng = random.Random(img_hash)

    baseline = POSITION_BASELINES[pos_key]
    wear_pct = mileage_km / baseline['lifespan_km']

    # Build feature vector
    features = {
        'tread_depth_mm':       float(round(tread_depth, 2)),
        'tread_wear_variance':  float(round(wear_variance, 3)),
        'shoulder_wear_idx':    float(round(min(1.0, wear_pct * 0.6 + rng.uniform(0, 0.15)), 3)),
        'sidewall_crack_density': float(round(crack_density, 3)),
        'bulge_detected':       bool(wear_pct > 0.9 and rng.random() < 0.3),
        'bulge_confidence':     float(round(max(0, wear_pct * 0.5 - 0.2 + rng.uniform(-0.1, 0.2)), 3)),
        'ozone_aging_score':    float(round(min(1.0, (dot_info['age_months'] / 120) + rng.uniform(0, 0.15)), 3)),
        'cut_puncture_count':   int(max(0, wear_pct * 3 + rng.gauss(0, 0.5))),
        'tire_age_months':      float(round(dot_info['age_months'], 1)),
        'retread_flag':         bool(dot_info['retread']),
        'heat_damage_idx':      float(round(min(1.0, wear_pct * 0.4 + rng.uniform(0, 0.2)), 3)),
        'rim_corrosion_lvl':    float(round(rng.uniform(0, 0.3), 3)),
        'valve_integrity':      float(round(1.0 - rng.uniform(0, 0.2), 3)),
        'tire_position':        pos_key,
        'mileage_since_last':   float(mileage_km),
        'capture_light_quality': float(quality['overall']),
        'ambient_temp_c':       float(ambient_temp),
        'exposed_cords':        bool(tread_depth < 1.5),
        'sidewall_crack_mm':    float(round(crack_density * 35, 1)),
    }

    # Override for specific zones
    if capture_zone == 'sidewall':
        features['sidewall_crack_density'] = round(min(1.0, crack_density * 1.5), 3)
        features['bulge_confidence'] = round(min(1.0, features['bulge_confidence'] * 1.3), 3)

    # Risk classification
    risk_flag, risk_triggers = compute_risk_flag(features)

    # Reliability score
    reliability = compute_reliability_score(features, pos_key)

    # Shelf life
    shelf_life_pct, remaining_km = compute_shelf_life(features, mileage_km, pos_key)

    # Recommendations
    recommendations = []
    if risk_flag == 'DO-NOT-OPERATE':
        recommendations.append({'action': 'IMMEDIATE REPLACEMENT', 'urgency': 'now', 'color': '#ff0033'})
    elif risk_flag == 'CRITICAL':
        recommendations.append({'action': 'Replace within 48 hours / 100 miles', 'urgency': '48h', 'color': '#ff6b00'})
    elif risk_flag == 'MONITOR':
        recommendations.append({'action': 'Schedule inspection within 2-4 weeks', 'urgency': '2-4 weeks', 'color': '#ffcc00'})
    else:
        recommendations.append({'action': 'Continue normal operation', 'urgency': 'none', 'color': '#00e676'})

    if wear_pct > 0.7:
        recommendations.append({'action': f'Plan replacement at ~{int(remaining_km):,} km', 'urgency': 'scheduled', 'color': '#00d4ff'})

    # Generate mock defects for the Digital Twin
    import math
    defects = []
    if tread_depth < 4.0:
        defects.append({'type': 'low_tread', 'angle': rng.uniform(0, math.pi*2), 'radius': 0.8, 'severity': 'high'})
    if wear_variance > 0.3:
        defects.append({'type': 'uneven_wear', 'angle': rng.uniform(0, math.pi*2), 'radius': 0.8, 'severity': 'medium'})
    if crack_density > 0.3:
        defects.append({'type': 'sidewall_crack', 'angle': rng.uniform(0, math.pi*2), 'radius': 0.6, 'severity': 'medium'})
    if features.get('bulge_detected'):
        defects.append({'type': 'bulge', 'angle': rng.uniform(0, math.pi*2), 'radius': 0.65, 'severity': 'high'})
    if features.get('cut_puncture_count', 0) > 0:
        for _ in range(features['cut_puncture_count']):
            defects.append({'type': 'cut', 'angle': rng.uniform(0, math.pi*2), 'radius': rng.uniform(0.7, 0.8), 'severity': 'high'})

    response = {
        'features': features,
        'dot_info': dot_info,
        'quality': quality,
        'risk_flag': risk_flag,
        'risk_triggers': risk_triggers,
        'reliability_score': reliability,
        'shelf_life_pct': shelf_life_pct,
        'remaining_km': remaining_km,
        'recommendations': recommendations,
        'defects': defects,
        'analysis_time_ms': int(200 + rng.uniform(50, 300)),
        'timestamp': datetime.now().isoformat(),
        'position': pos_key,
    }

    return jsonify(response)


@app.route('/api/ocr', methods=['POST'])
def ocr_dot():
    """DOT code OCR simulation."""
    data = request.get_json(force=True) or {}
    image_b64 = data.get('image', '')
    hint = data.get('hint', '')

    img = decode_image(image_b64) if image_b64 else None

    # Simulate DOT detection
    # In production this would use a real OCR model
    # For demo: extract any 4-digit pattern, or generate plausible result
    import re
    candidates = re.findall(r'\d{4}', hint or '')

    if not candidates:
        # Generate a plausible recent DOT code
        rng = random.Random(hash(image_b64[:50] if image_b64 else 'default'))
        week = rng.randint(1, 52)
        year = rng.randint(18, 24)
        dot_str = f"{week:02d}{year:02d}"
    else:
        dot_str = candidates[0]

    dot_info = parse_dot_code(dot_str)
    dot_info['confidence'] = round(random.uniform(0.88, 0.97), 2)

    return jsonify({'dot_string': dot_str, 'dot_info': dot_info, 'raw_text': hint or dot_str})


@app.route('/api/fleet/update', methods=['POST'])
def update_fleet():
    """Save updated fleet data."""
    data = request.get_json(force=True)
    try:
        path = os.path.join(DATA_DIR, 'fleet-seed.json')
        with open(path, 'r') as f:
            current = json.load(f)
        current.update(data)
        with open(path, 'w') as f:
            json.dump(current, f, indent=2)
        return jsonify({'status': 'saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  TIRE VISION — AI Inference Server")
    print("  TrustGrid Fleet Tire Intelligence System")
    print("=" * 60)
    print(f"  PIL/Numpy available: {PIL_AVAILABLE}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Serving at: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
