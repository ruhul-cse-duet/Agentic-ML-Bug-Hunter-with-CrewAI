import os
from crewai import Agent
from llm_model import llm_model_fn

llm = llm_model_fn()

# Load runtime prompt
prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "runtime.txt")
with open(prompt_path, 'r', encoding='utf-8') as f:
    runtime_prompt = f.read()

runtime_agent = Agent(
    role="Runtime Error Debugger",
    goal="Analyze ML runtime errors and tracebacks with precision",
    backstory=runtime_prompt,
    llm=llm,
    verbose=True,
    allow_delegation=False
)
