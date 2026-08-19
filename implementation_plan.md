# Goal Description
Update the TireVision V2 app to support a multi-step inspection flow. The flow will guide the user through capturing images/video of the tread, sidewall, shoulder, valve, and DOT code with AR overlays. After capture, 16 features will be calculated on the backend, and a neatly formatted, center-aligned report will be generated. The 3D Digital Twin will be updated to display specific defect locations.

## Proposed Changes

### Backend (`app/server.py`)
- **[MODIFY] `server.py`**:
  - Update `/api/analyze` endpoint to accept a complex payload containing multiple images (tread, sidewall, shoulder, valve, dot) or video data instead of a single image.
  - Return the 16 backend features explicitly in the response payload.
  - Return a list of `defects` with coordinates/angles to render on the Digital Twin (e.g., `{ type: 'cut', angle: 45, radius: 0.8 }`).
  - Integrate DOT OCR directly from the DOT image passed in the payload and calculate tire age.

### Frontend (`app/index.html`)
- **[MODIFY] `index.html`**:
  - **Camera View**: Implement a multi-step wizard state machine:
    1. **Tread**: Capture 3 photos or record a 10-second video.
    2. **Sidewall**: Capture 2 photos.
    3. **Shoulder**: Capture 1 photo.
    4. **Valve**: Capture 1 photo.
    5. **DOT**: Capture 1 photo.
  - **AR Augmentations**: Dynamically update the canvas overlay (`drawAROverlay`) based on the current step (e.g., tread rectangle, sidewall circle, DOT box) with instructional text like "Place the DOT code inside the box."
  - **Results / Report View**:
    - Build a neatly formatted, center-aligned table/list displaying all 16 calculated features.
    - Improve the Print / PDF download styling to ensure the report looks clean and professional.
  - **Digital Twin**:
    - Update the Three.js implementation to parse the `defects` from the backend response.
    - Render red markers/decals on the 3D torus corresponding to the locations of the defects.

## User Review Required
> [!IMPORTANT]  
> The 10-second video capture for the tread will use the standard HTML5 `MediaRecorder` API. Do you want the video to be sent to the backend for processing, or should the frontend extract frames from it to send as images? (For simplicity and performance, sending extracted frames is often better).

> [!WARNING]
> Creating accurate markings on the 3D Digital Twin will rely on simulated defect coordinates from the backend, as mapping 2D photos to exact 3D coordinates requires complex 3D reconstruction. We will simulate realistic defect locations on the 3D model.

## Open Questions
- What Tailwind/CSS aesthetic preferences do you have for the report layout? It will be center-aligned and neat as requested, using the existing color scheme.
- Should the frontend automatically move to the next step after a photo is captured, or require a "Next" button?

## Verification Plan
### Manual Verification
- Open `http://localhost:5000` in the browser.
- Start the camera and verify the UI guides through Tread -> Sidewall -> Shoulder -> Valve -> DOT.
- Verify AR overlays change per step.
- Verify the final report displays the 16 features cleanly and centered.
- Verify the 3D Digital Twin displays markers for defects.
- Download the PDF report and verify its formatting.
