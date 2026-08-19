# TireVision 🛞

TireVision is an AI-powered fleet tire intelligence system designed to capture, analyze, and report on tire health and wear in real-time. It provides a guided web interface for capturing comprehensive tire images and uses a local Python backend to estimate tread depth, check for defects, and calculate reliability scores and shelf life.

## 🚀 Key Features

### Frontend UI & Experience
- **Guided 5-Step Inspection Wizard:** Easy-to-follow process capturing crucial angles of the tire (Tread, Sidewall, Shoulder, Valve, and DOT code).
- **Web Camera Integration:** Real-time access to the device camera directly from the browser for taking inspection photos.
- **Detailed Diagnostic Reports:** Displays computed metrics in a beautiful dashboard, categorizing tire risk as SAFE, MONITOR, CRITICAL, or DO-NOT-OPERATE.
- **3D Digital Twin Viewer:** Renders a 3D model of the tire to visualize wear zones and structural anomalies.

### AI & Backend Capabilities
- **Tread Wear Estimation:** Uses image processing techniques (brightness, contrast, edge variances) combined with historical fleet mileage to estimate tread depth accurately.
- **Defect Detection Pipeline:** Identifies potential issues like uneven wear variances, high crack densities, heat damage, and exposed cords.
- **DOT Code OCR Analysis:** Extracts 4-digit Date of Manufacture (DOT) codes from images to accurately determine the tire's age in months and applies age-based penalties to its shelf life.
- **Reliability & Shelf-Life Algorithms:** Computes an aggregated 0–100 reliability score and estimates the remaining shelf life based on multiple weighted variables (tread depth, cracks, heat index, and age).

## 🛠️ Technology Stack

**Frontend:**
- HTML5, CSS3, JavaScript (Vanilla)
- WebRTC (for camera stream access)

**Backend / AI Inference Server:**
- **Python 3.9+**
- **Flask:** Serves the frontend static files and hosts the RESTful JSON API endpoints (`/api/analyze`).
- **Flask-CORS:** Manages cross-origin requests.
- **Pillow (PIL) & NumPy:** For decoding base64 image streams and extracting mathematical image statistics (laplacian variance for blur, brightness, contrast) for quality checks and wear modifiers.
- **EasyOCR:** For reading and extracting text patterns (like the DOT date code) from captured tire sidewall images.

## ⚙️ Setup and Installation

1. **Prerequisites:**
   - Install [Python 3.9+](https://www.python.org/downloads/) on your system.

2. **Open the Repository:**
   Navigate to the root directory of the `TyreVision` project in your terminal.

3. **Install Dependencies:**
   Run the following command to install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages: `flask`, `flask-cors`, `Pillow`, `numpy`, `easyocr`*

4. **Start the Server:**
   You can either run the provided batch script or start the python server manually.
   ```bash
   # Option 1: Using the batch script (Windows)
   start.bat

   # Option 2: Running manually
   python server.py
   ```
   
5. **Access the Application:**
   Once the server starts (it may take a moment to download the OCR models on the first run), open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```

## 📝 Usage
Click on **Start Inspection** to launch the camera wizard. Take the required snapshots (Tread, Sidewall, Shoulder, Valve, and DOT code) and let the engine analyze them. The system will process the images locally and return a comprehensive analysis of your tire's health and structural integrity.
