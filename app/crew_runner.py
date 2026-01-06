import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from crewai import Crew, Task, Process
from agents.runtime_agent import runtime_agent
from agents.logic_agent import logic_agent
from agents.fix_agent import fix_agent
from agents.performance_agent import performance_agent
import requests
import traceback

def run_crew(user_input):
    """
    Run the CrewAI bug hunting crew to analyze ML code.
    
    Args:
        user_input: ML code or error logs to analyze
        
    Returns:
        Comprehensive bug report with code-only fixes and suggestions.
    """
    
    try:
        # Task 1: Runtime Error Analysis
        t1 = Task(
            description=f"""
            Analyze the following ML code or error logs for runtime errors:
            
            {user_input}
            
            Your analysis should include:
            1. Identify any runtime errors or exceptions
            2. Detect tensor shape mismatches
            3. Find CUDA/GPU-related issues
            4. Check for memory problems (OOM, leaks)
            5. Identify data type incompatibilities
            6. Verify device placements (CPU vs GPU)
            7. Look for DataLoader issues
            
            Provide detailed findings with line numbers if applicable.
            """,
            agent=runtime_agent,
            expected_output="Detailed runtime error analysis with identified issues"
        )

        # Task 2: ML Logic Review
        t2 = Task(
            description=f"""
            Review the ML logic in the following code:
            
            {user_input}
            
            Check for:
            1. Model architecture issues
            2. Loss function correctness
            3. Optimizer configuration problems
            4. Learning rate scheduling issues
            5. Data preprocessing logic
            6. Layer dimension mismatches
            7. Activation function choices
            8. Gradient flow problems
            9. Training loop logic errors
            10. Validation/test split issues
            
            Focus on silent bugs that don't throw errors but produce incorrect results.
            """,
            agent=logic_agent,
            expected_output="Comprehensive ML logic review with identified logical issues",
            context=[t1]
        )

        # Task 3: Code Fix Generation
        t3 = Task(
            description="""
            Based on the runtime errors and logic issues identified by the previous agents,
            generate clean, production-ready code fixes.
            
            Your fixes should:
            1. Address all identified issues
            2. Include necessary imports
            3. Have clear comments explaining changes
            4. Follow Python/PyTorch best practices
            5. Be copy-paste ready
            6. Include error handling where needed
            7. Maintain code readability
            
            Provide complete, working code solutions with explanations.
            """,
            agent=fix_agent,
            expected_output="Complete code fixes with explanations for all identified issues",
            context=[t1, t2]
        )

        # Task 4: Performance Optimization
        t4 = Task(
            description="""
            Analyze the code for performance optimization opportunities.
            
            Suggest optimizations for:
            1. Training speed improvements
            2. Memory usage reduction
            3. Efficient data loading
            4. GPU utilization
            5. Batch size optimization
            6. Mixed precision training (AMP)
            7. Gradient accumulation
            8. Model parallelization
            9. Code generate
            
            IMPORTANT: Only provide code or code examples, only bullet point explanations..
            """,
            agent=performance_agent,
            expected_output="performance optimization suggestions with code examples",
            context=[t1, t2, t3]
        )

        # Create and run the crew
        crew = Crew(
            agents=[runtime_agent, logic_agent, fix_agent, performance_agent],
            tasks=[t1, t2, t3, t4],
            verbose=True,
            process=Process.sequential  # Tasks run in order
        )
        
        # Run the crew and return results
        print("🚀 Starting AI agents analysis...")
        result = crew.kickoff()
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     🎯 ML BUG HUNTER ANALYSIS REPORT                     ║
╚══════════════════════════════════════════════════════════════════════════╝

{result}

╔══════════════════════════════════════════════════════════════════════════╗
║                         ✨ ANALYSIS COMPLETE ✨                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return f"""
🔴 ANALYSIS ERROR
══════════════════════════════════════════════════════════════

An error occurred during analysis:

{str(e)}

FULL TRACEBACK:
{error_trace}

TROUBLESHOOTING:
1. Verify Ollama is running: ollama list
2. Check model availability: ollama pull
3. Ensure all dependencies are installed: pip install -r requirements.txt
4. Check if ports 8000 and 11434 are available

If the problem persists, please check the terminal output for more details.
══════════════════════════════════════════════════════════════
"""
