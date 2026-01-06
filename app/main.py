import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import Config
from crew_runner import run_crew

import io

# # Force UTF-8 stdout/stderr (Windows safe)
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

app = FastAPI(
    title="🧠 Agentic ML Bug Hunter",
    description="AI-powered bug hunting system for ML projects",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_path = os.path.join(os.path.dirname(__file__), "..", "static")
templates_path = os.path.join(os.path.dirname(__file__), "..", "templates")

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": Config.OLLAMA_MODEL,
        "temperature": Config.TEMPERATURE,
        "max_tokens": Config.MAX_TOKENS
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, code: str = Form(...)):
    """Analyze ML code with AI agents"""
    try:
        if not code or len(code.strip()) < 10:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": "❌ Please provide meaningful code or error logs (at least 10 characters)."
                }
            )
        
        # Run the crew analysis
        result = run_crew(code)
        
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result, "code": code}
        )
    
    except Exception as e:
        error_message = f"""
SERVER ERROR
══════════════════════════════════════════════════════════════

An unexpected error occurred:
{str(e)}

Please try again or contact support if the issue persists.
══════════════════════════════════════════════════════════════
"""
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": error_message}
        )

if __name__ == "__main__":
    import sys
    import io
    # Fix encoding for Windows console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("="*70)
    print("AGENTIC ML BUG HUNTER")
    print("="*70)
    print("Starting FastAPI server...")
    print("Server will be available at: http://127.0.0.1:8000")
    print("Powered by: CrewAI + Ollama (llama2:7b)")
    print("="*70)

    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
