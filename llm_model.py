# llm_model.py
import sys
import io
from crewai import LLM
from config import Config


# ───────────────────────────────────────────────────────────────
# LLM initialization function
def llm_model_fn():
    """
    Initialize CrewAI LLM for DeepSeek-Coder model safely on Windows.
    Returns:
        LLM instance
    """
    try:
        llm = LLM(
            model=Config.OLLAMA_MODEL,  # deepseek-coder:1.3b
            base_url=Config.OLLAMA_BASE_URL,  # http://localhost:11434
            temperature=Config.TEMPERATURE,  # 0.4
            max_tokens=Config.MAX_TOKENS,    # 512
            timeout=1000,
            do_sample=True  # optional: enables better code suggestion diversity
        )
        # Safe ASCII-only logging
        print("OLLAMA_MODEL LLM initialized successfully!\n")
        return llm

    except Exception as e:
        print("OLLAMA_MODEL LLM initialization failed.\n")
        print("Error:", str(e))
        return None

# ───────────────────────────────────────────────────────────────
# Initialize globally if needed
# llm = llm_model_fn()
