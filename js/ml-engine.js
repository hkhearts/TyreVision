/**
 * ml-engine.js — Browser-side ML Inference Engine
 * Tire Vision — Fleet Tire Intelligence System
 *
 * Loads ONNX models via onnxruntime-web and runs the same
 * 4-model ensemble as the Wheelyze backend:
 *   1. binary_model.onnx   — EfficientNet-B3: Good (1) vs Defective (0)
 *   2. defective_model.onnx — EfficientNet-B3: Defective sub-grade 1–5
 *   3. good_model.onnx      — EfficientNet-B3: Good sub-grade 6–10
 *   4. resnet_model.onnx    — ResNet18: Full 1–10 grade
 *
 * The ensemble logic matches final_combine.py exactly.
 */

const ML_ENGINE = (() => {

  // ImageNet normalization constants (same as Wheelyze val_transform)
  const MEAN = [0.485, 0.456, 0.406];
  const STD  = [0.229, 0.224, 0.225];
  const INPUT_SIZE = 300;

  let sessions = null;
  let loading   = false;
  let loadError = null;

  /**
   * Load all 4 ONNX model sessions. Returns true on success.
   * Models are loaded from /models/ on the same origin.
   */
  async function loadModels() {
    if (sessions) return true;
    if (loading)  return false;
    if (loadError) return false;

    loading = true;
    try {
      // Ensure ort is available (loaded via CDN in index.html)
      if (typeof ort === 'undefined') {
        throw new Error('onnxruntime-web not loaded');
      }

      // Use WASM backend for broad compatibility
      ort.env.wasm.numThreads = 1;

      console.log('[ML] Loading ONNX models...');
      const [binary, defective, good, resnet] = await Promise.all([
        ort.InferenceSession.create('/models/binary_model.onnx',   { executionProviders: ['wasm'] }),
        ort.InferenceSession.create('/models/defective_model.onnx', { executionProviders: ['wasm'] }),
        ort.InferenceSession.create('/models/good_model.onnx',      { executionProviders: ['wasm'] }),
        ort.InferenceSession.create('/models/resnet_model.onnx',    { executionProviders: ['wasm'] }),
      ]);

      sessions = { binary, defective, good, resnet };
      console.log('[ML] All 4 models loaded successfully ✓');
      loading = false;
      return true;
    } catch (e) {
      console.warn('[ML] Model load failed:', e.message);
      loadError = e.message;
      loading = false;
      return false;
    }
  }

  /**
   * Preprocess an HTMLImageElement or HTMLCanvasElement into a Float32 tensor.
   * Applies: Resize → ToTensor → Normalize (ImageNet stats).
   */
  function preprocessImage(imageSource) {
    const canvas = document.createElement('canvas');
    canvas.width  = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageSource, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = imageData;  // RGBA interleaved
    const n = INPUT_SIZE * INPUT_SIZE;

    // NCHW float32 tensor: [1, 3, 300, 300]
    const tensor = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < n; i++) {
      const r = data[i * 4]     / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      tensor[0 * n + i] = (r - MEAN[0]) / STD[0];  // R channel
      tensor[1 * n + i] = (g - MEAN[1]) / STD[1];  // G channel
      tensor[2 * n + i] = (b - MEAN[2]) / STD[2];  // B channel
    }

    return new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  }

  /**
   * Softmax utility.
   */
  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  /**
   * Argmax utility.
   */
  function argmax(arr) {
    let maxIdx = 0;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > arr[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  /**
   * Run inference on a single image source (HTMLImageElement or canvas).
   * Returns { efficientnet_pred, resnet_pred, ensemble_pred, ml_grade, probabilities }
   */
  async function runInference(imageSource) {
    if (!sessions) {
      const ok = await loadModels();
      if (!ok) return null;
    }

    const inputTensor = preprocessImage(imageSource);
    const feeds = { input: inputTensor };

    // Stage 1: Binary EfficientNet (Good=1 vs Defective=0)
    const binaryOut  = await sessions.binary.run(feeds);
    const binaryLogits = Array.from(binaryOut.output.data);
    const binaryProbs  = softmax(binaryLogits);
    const predBinary   = argmax(binaryProbs);  // 0=Defective, 1=Good

    let effPred;
    let effProbs;
    let validRange;

    if (predBinary === 0) {
      // Defective: use defective model → grades 1–5
      const out = await sessions.defective.run(feeds);
      const logits = Array.from(out.output.data);
      effProbs = softmax(logits);
      effPred  = argmax(effProbs) + 1;  // 0-indexed → 1-5
      validRange = [1, 2, 3, 4, 5];
    } else {
      // Good: use good model → grades 6–10
      const out = await sessions.good.run(feeds);
      const logits = Array.from(out.output.data);
      effProbs = softmax(logits);
      effPred  = argmax(effProbs) + 6;  // 0-indexed → 6-10
      validRange = [6, 7, 8, 9, 10];
    }

    // Stage 2: ResNet18 full prediction (1–10)
    const resnetOut    = await sessions.resnet.run(feeds);
    const resnetLogits = Array.from(resnetOut.output.data);
    const resnetProbs  = softmax(resnetLogits);
    const resPred      = argmax(resnetProbs) + 1;  // 0-indexed → 1-10

    // Ensemble logic (matches final_combine.py exactly)
    let ensemblePred;
    const inRange = validRange.includes(resPred);

    if (inRange) {
      const avg = (effPred + resPred) / 2;
      ensemblePred = (avg - Math.floor(avg)) >= 0.5
        ? Math.floor(avg) + 1
        : Math.floor(avg);
    } else {
      // ResNet disagrees → EfficientNet only (starred)
      ensemblePred = `${effPred}*`;
    }

    return {
      efficientnet_pred: effPred,
      resnet_pred:       resPred,
      ensemble_pred:     ensemblePred,
      binary_pred:       predBinary,
      ml_grade:          predBinary === 0 ? 'Defective' : 'Good',
      binary_confidence: Math.round(binaryProbs[predBinary] * 100),
      model_agreed:      inRange,
    };
  }

  /**
   * Build the full 16-feature vector from ML ensemble score.
   * Score 1 = worst, 10 = best.
   */
  function buildFeaturesFromScore(mlResult, inspectionData, dotInfo) {
    const { mileageKm = 50000, position = 'drive', ambientTempC = 28 } = inspectionData;

    const BASELINES = {
      steer:   { newDepth: 19, minLegal: 5, lifespan: 150000, wearRate: 0.093 },
      drive:   { newDepth: 24, minLegal: 5, lifespan: 250000, wearRate: 0.076 },
      trailer: { newDepth: 16, minLegal: 5, lifespan: 180000, wearRate: 0.061 },
    };

    const pos = (position.includes('steer') || position.includes('front')) ? 'steer'
              : position.includes('trailer') ? 'trailer' : 'drive';
    const bl  = BASELINES[pos];

    // Parse ensemble score
    const scoreRaw = String(mlResult.ensemble_pred);
    const scoreInt = Math.max(1, Math.min(10, parseInt(scoreRaw.replace('*', ''), 10)));

    // Score → feature percentages (calibrated)
    const SCORE_MAP = {
      1:  { treadPct:0.04, crack:0.90, heat:0.92, ozone:0.88, cuts:4, bulgeConf:0.92, wearVar:0.85, shoulder:0.88 },
      2:  { treadPct:0.09, crack:0.78, heat:0.80, ozone:0.75, cuts:3, bulgeConf:0.75, wearVar:0.72, shoulder:0.76 },
      3:  { treadPct:0.17, crack:0.65, heat:0.65, ozone:0.60, cuts:2, bulgeConf:0.52, wearVar:0.58, shoulder:0.60 },
      4:  { treadPct:0.28, crack:0.50, heat:0.52, ozone:0.48, cuts:2, bulgeConf:0.38, wearVar:0.45, shoulder:0.46 },
      5:  { treadPct:0.40, crack:0.38, heat:0.40, ozone:0.36, cuts:1, bulgeConf:0.22, wearVar:0.34, shoulder:0.35 },
      6:  { treadPct:0.53, crack:0.26, heat:0.28, ozone:0.25, cuts:0, bulgeConf:0.10, wearVar:0.22, shoulder:0.24 },
      7:  { treadPct:0.64, crack:0.16, heat:0.18, ozone:0.16, cuts:0, bulgeConf:0.05, wearVar:0.14, shoulder:0.15 },
      8:  { treadPct:0.75, crack:0.09, heat:0.10, ozone:0.10, cuts:0, bulgeConf:0.02, wearVar:0.08, shoulder:0.08 },
      9:  { treadPct:0.86, crack:0.04, heat:0.05, ozone:0.05, cuts:0, bulgeConf:0.01, wearVar:0.04, shoulder:0.04 },
      10: { treadPct:0.95, crack:0.01, heat:0.02, ozone:0.02, cuts:0, bulgeConf:0.00, wearVar:0.02, shoulder:0.02 },
    };

    const sm = SCORE_MAP[scoreInt];
    const treadDepth = Math.round(bl.newDepth * sm.treadPct * 100) / 100;
    const ageMonths  = dotInfo?.ageMonths || dotInfo?.age_months || 0;

    // Small deterministic noise from mileage
    const seed = mileageKm % 997;
    const noise = (k, scale=0.04) => ((seed * k * 2654435761 % 65536) / 65536 - 0.5) * scale;

    return {
      tread_depth_mm:         treadDepth,
      tread_wear_variance:    Math.max(0, Math.min(1, sm.wearVar + noise(1))),
      shoulder_wear_idx:      Math.max(0, Math.min(1, sm.shoulder + noise(2))),
      sidewall_crack_density: Math.max(0, Math.min(1, sm.crack + noise(4))),
      bulge_detected:         sm.bulgeConf > 0.85,
      bulge_confidence:       Math.max(0, Math.min(1, sm.bulgeConf + noise(3))),
      ozone_aging_score:      Math.max(0, Math.min(1, sm.ozone + ageMonths/240 + noise(5))),
      cut_puncture_count:     Math.max(0, sm.cuts + (sm.cuts > 0 ? Math.round(noise(6, 1)) : 0)),
      tire_age_months:        ageMonths,
      retread_flag:           dotInfo?.retread || false,
      heat_damage_idx:        Math.max(0, Math.min(1, sm.heat + noise(7))),
      rim_corrosion_lvl:      Math.max(0, Math.min(0.6, 0.15 * (1 - sm.treadPct) + noise(8, 0.05))),
      valve_integrity:        Math.max(0.3, Math.min(1, 0.5 + sm.treadPct * 0.5 + noise(9))),
      tire_position:          pos,
      mileage_since_last:     mileageKm,
      capture_light_quality:  0.78,
      ambient_temp_c:         ambientTempC,
      exposed_cords:          treadDepth < 1.5,
      sidewall_crack_mm:      Math.round(sm.crack * 35 * 10) / 10,
      // ML metadata
      ml_score:               scoreInt,
      ml_grade:               mlResult.ml_grade,
      ml_model_agreed:        mlResult.model_agreed,
    };
  }

  return {
    loadModels,
    runInference,
    buildFeaturesFromScore,
    isAvailable: () => sessions !== null,
    isLoading:   () => loading,
    getError:    () => loadError,
  };
})();

export default ML_ENGINE;
