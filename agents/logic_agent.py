import os
from crewai import Agent
from llm_model import llm_model_fn

llm = llm_model_fn()


# Load logic prompt
prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "logic.txt")
with open(prompt_path, 'r', encoding='utf-8') as f:
    logic_prompt = f.read()

logic_agent = Agent(
    role="ML Logic Reviewer",
    goal="Detect silent bugs in model design and training logic",
    backstory=logic_prompt,
    llm=llm,
    verbose=True,
    allow_delegation=False
)
