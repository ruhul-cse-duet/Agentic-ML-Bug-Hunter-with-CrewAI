import os
from crewai import Agent, LLM
from llm_model import llm_model_fn

llm = llm_model_fn()

# Load fix prompt
prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "fix.txt")
with open(prompt_path, 'r', encoding='utf-8') as f:
    fix_prompt = f.read()

fix_agent = Agent(
    role="Code Fix Generator",
    goal="Generate clean and correct code patches for identified bugs",
    backstory=fix_prompt,
    llm=llm,
    verbose=True,
    allow_delegation=False
)
