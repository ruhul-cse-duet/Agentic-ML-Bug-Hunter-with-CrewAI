import os
from crewai import Agent
from llm_model import llm_model_fn

llm = llm_model_fn()

# Load performance prompt
prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "performance.txt")
with open(prompt_path, 'r', encoding='utf-8') as f:
    performance_prompt = f.read()

performance_agent = Agent(
    role="Performance Optimizer",
    goal="Optimize training speed and memory usage for ML models",
    backstory=performance_prompt,
    llm=llm,
    verbose=True,
    allow_delegation=False
)
