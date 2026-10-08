# Hospital RAG Assistant

A full-stack AI application using React, Express, Python (LangChain), and Google Gemini to answer questions about hospital reviews via Retrieval-Augmented Generation (RAG).

## Tech Stack
- **Frontend:** React.js
- **Backend Middleware:** Express.js (Node.js) with `python-shell`
- **AI / RAG Pipeline:** Python, LangChain, Chroma DB, Google Gemini API

## Setup Instructions

### 1. Backend Setup
```bash
cd backend
npm install
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt