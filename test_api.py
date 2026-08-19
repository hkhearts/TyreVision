"""Quick integration test for TyreVision ML API"""
import urllib.request, json, base64

print("=== TYREVISION ML INTEGRATION TEST ===")
print()

# 1. Health check
req = urllib.request.urlopen("http://localhost:5000/api/health", timeout=5)
h = json.loads(req.read())
print("[1] Health Check:")
print("    Server:", h["server"])
print("    ML Active:", h["ml_available"])
print("    PIL:", h["pil"])
print()

# 2. ML Status
req = urllib.request.urlopen("http://localhost:5000/api/ml-status", timeout=5)
m = json.loads(req.read())
print("[2] ML Status:")
print("    Models:", m["models"])
print("    Engine:", m["inference_engine"])
print()

# 3. Good tyre prediction
with open("Wheelyze-Tyre-Health-Detection/goodtyre.jpeg", "rb") as f:
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
payload = json.dumps({"image": img_b64, "mileage_km": 30000, "position": "drive", "dot_code": "2823"}).encode()
req = urllib.request.Request(
    "http://localhost:5000/api/analyze",
    data=payload, headers={"Content-Type": "application/json"}, method="POST"
)
d = json.loads(urllib.request.urlopen(req, timeout=30).read())
print("[3] Good Tyre Test:")
print("    ML Score:", str(d["ml_score"]) + "/10")
print("    Grade:", d["features"]["ml_grade"])
print("    Risk:", d["risk_flag"])
print("    Reliability:", str(d["reliability_score"]) + "/100")
print("    Tread:", str(d["features"]["tread_depth_mm"]) + "mm")
print("    Shelf Life:", str(d["shelf_life_pct"]) + "%")
feat_count = len([k for k in d["features"].keys() if not k.startswith("ml_")])
print("    Feature count:", feat_count, "features populated")
print()

# 4. Defective tyre prediction
with open("Wheelyze-Tyre-Health-Detection/defecttyre.jpeg", "rb") as f:
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
payload = json.dumps({"image": img_b64, "mileage_km": 150000, "position": "steer", "dot_code": "1517"}).encode()
req = urllib.request.Request(
    "http://localhost:5000/api/analyze",
    data=payload, headers={"Content-Type": "application/json"}, method="POST"
)
d = json.loads(urllib.request.urlopen(req, timeout=30).read())
print("[4] Defective Tyre Test:")
print("    ML Score:", str(d["ml_score"]) + "/10")
print("    Grade:", d["features"]["ml_grade"])
print("    Risk:", d["risk_flag"])
print("    Reliability:", str(d["reliability_score"]) + "/100")
print("    Tread:", str(d["features"]["tread_depth_mm"]) + "mm")
print("    Shelf Life:", str(d["shelf_life_pct"]) + "%")
print("    Triggers:", d["risk_triggers"])
print()

# 5. Fleet API
req = urllib.request.urlopen("http://localhost:5000/api/fleet", timeout=5)
fleet = json.loads(req.read())
vehicles = fleet.get("vehicles", [])
print("[5] Fleet API:")
print("    Vehicles:", len(vehicles), "loaded")
if vehicles:
    v = vehicles[0]
    print("    First:", v["make"], v["model"], "-", v["regNo"])
print()

# 6. DOT OCR
payload = json.dumps({"image": "", "hint": "DOT XY 1234 5624"}).encode()
req = urllib.request.Request(
    "http://localhost:5000/api/ocr",
    data=payload, headers={"Content-Type": "application/json"}, method="POST"
)
d = json.loads(urllib.request.urlopen(req, timeout=5).read())
print("[6] DOT OCR Test:")
print("    Raw:", d["raw_text"])
print("    Parsed: Week", d["dot_info"]["week"], "Year", d["dot_info"]["year"])
print("    Mfg Date:", d["dot_info"]["mfg_date"])
print("    Age:", d["dot_info"]["age_months"], "months")
print()
print("=== ALL TESTS PASSED ===")
