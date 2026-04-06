# EmpathyCo System

EmpathyCo (formerly EmotiDesk) is a multimodal emotion recognition SaaS Dashboard, integrating speech and text transcription analysis.

## Project Structure
- `frontend/` - React, TypeScript, and Vite-powered UI configured with Tailwind CSS v4.
- `backend/` - FastAPI backend application for model inference (entry point: `main.py`).
- `src/` - Reusable modules for audio processing, text analysis, and fusion.
- `models/` - Trained models (`.h5`) for English, Hindi, and Marathi analysis.
- `data/` - Raw and processed datasets for training phases.

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+

### Setup

1. **Backend Environment**
```bash
# Create a virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r backend/requirements.txt
```

2. **Frontend Environment**
```bash
cd frontend
npm install
```

### Running the Application

You can start both the backend (Port 8001) and the frontend (Port 5173) simultaneously using the provided startup script at the project root:

```bash
# From the project root
.\start.bat
```

Alternatively, you can run from the frontend folder:

```bash
cd frontend
npm run dev
```

### Environment Variables
Ensure the local configuration points to the correct backend port (8001). The Vite proxy in `vite.config.ts` is already configured for `http://localhost:8001`.
