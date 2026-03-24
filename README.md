# EmpathyCo System

EmpathyCo (formerly EmotiDesk) is a multimodal emotion recognition SaaS Dashboard, integrating speech and text transcription analysis.

## Project Structure
- `frontend/` - React, TypeScript, and Vite-powered UI configured with Tailwind CSS v4.
- `main.py` - FastAPI backend application for model inference.
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
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

2. **Frontend Environment**
```bash
cd frontend
npm install
```

### Running the Application

The frontend `package.json` has been configured with `concurrently` to run both the Vite UI and the FastAPI backend simultaneously.

```bash
cd frontend
npm run dev
```

This ensures:
- The React application is running at `http://localhost:5173`
- The Python FastAPI backend acts on port `8000`.

### Environment Variables
Ensure the `.env` file exists in `frontend/`:
```env
VITE_API_URL=http://localhost:8000
```
*(The Vite proxy acts identically behind the scenes).*
