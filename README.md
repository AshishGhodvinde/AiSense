# AiSense: Premium AI Image Detection

A highly accurate, modern web application that classifies whether an image is AI-generated or a real, authentic photograph using a hybrid ensemble approach of **Metadata/Provenance Heuristics (EXIF, Noise, ELA)** and a custom **PyTorch Convolutional Neural Network (CNN)**.

This project was built to deliver rapid local execution without compromising on a beautifully sleek, Vanta.js animated glassmorphic interface.

## ✨ Features
*   **Hybrid Decision Engine:** Combines deep spatial feature analysis (CNN) with hard mathematical optical heuristics (Error Level Analysis, Laplacian Variance) and contextual metadata detection (EXIF, C2PA markers).
*   **Rapid Local Machine Learning:** Includes a custom PyTorch model training script that automatically downloads a lightweight dataset (CIFAKE) and trains locally in just a few minutes.
*   **Modern UI/UX:** React + Vite frontend featuring a sleek Glassmorphism interface, interactive drag-and-drop mechanics, real-time topological background animations (`Vanta.js`), and pristine typography.
*   **Fully Extensible Explainability:** Provides a transparent breakdown of component predictions, technical score metrics, and readable rationale for users.

---

## 🚀 Getting Started

Follow these step-by-step instructions to get the AI Image Detector running natively on your machine after downloading the `.zip` or cloning the repository.

### Prerequisites
*   [Node.js](https://nodejs.org/) (For compiling the Frontend UI)
*   [Python 3.8+](https://www.python.org/) (For serving the PyTorch Backend)

### 1. Setup the React Frontend
Open a terminal and navigate to the extracted project directory, then into the `frontend` folder:
```bash
cd AiSense/frontend

# Install the Node dependencies
npm install

# Start the frontend development server
npm run dev
```
*The frontend interface will launch at `http://localhost:5173`.*

### 2. Setup the Python Backend
Open a **second, new terminal** and navigate to the `backend` folder:
```bash
cd AiSense/backend

# Create an isolated python virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install required Machine Learning and API libraries
pip install -r requirements.txt
```

### 3. Train the Local CNN Model
Before you can analyze your first image, you must train the local PyTorch model.
With your virtual environment still activated in the backend folder, run the training script:
```bash
python train.py
```
*What this does:* This script will automatically use `kagglehub` to download the lightweight `CIFAKE` image dataset (105MB) and rapidly train a custom Convolutional Neural Network. It takes roughly 2-4 minutes depending on your hardware, and immediately saves the resulting `best_model.pth` into your local `weights/` directory for the backend to use.

### 4. Boot the FastAPI Server
Once the model finishes training successfully, you can launch the backend inference engine:
```bash
uvicorn app:app --reload
```
*The backend API will run statelessly on `http://127.0.0.1:8000`.*

---

## 🎯 Usage
1. Ensure both your Backend (`uvicorn`) and Frontend (`npm run dev`) terminals are running and active simultaneously.
2. Open your web browser to `http://localhost:5173`.
3. Drag and drop any image (PNG, JPG, JPEG) onto the glowing glass panel.
4. Review the AI prediction, dynamic confidence percentage, generated ELA optical heatmap, and full metadata explanation!
