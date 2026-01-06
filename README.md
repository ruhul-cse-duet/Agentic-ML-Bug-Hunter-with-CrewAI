# 🧠 Agentic ML Bug Hunter with CrewAI

An intelligent bug hunting system for Machine Learning projects powered by CrewAI and Local LLM (Ollama). This system uses multiple specialized AI agents to analyze, debug, and optimize your ML code.

## ✨ Features

- 🔍 **Runtime Error Analysis**: Detect CUDA, tensor, and memory errors
- 🧩 **ML Logic Review**: Find silent bugs in model architecture
- 🔧 **Automatic Code Fixes**: Generate production-ready patches
- ⚡ **Performance Optimization**: Improve training speed and memory usage
- 💻 **Beautiful UI**: Modern, responsive web interface
- 🤖 **Local LLM**: Runs entirely on your machine using Ollama
- 🎯 **Multi-Agent System**: 4 specialized AI agents working together

## 🏗️ Architecture

The system uses 4 specialized AI agents working together:

1. **Runtime Error Debugger**: Analyzes runtime errors and tracebacks
2. **ML Logic Reviewer**: Detects logical bugs in model design
3. **Code Fix Generator**: Creates clean code patches
4. **Performance Optimizer**: Suggests speed and memory optimizations

## 📋 Prerequisites

- Python 3.11.14 or higher
- Ollama installed and running
- At least one Ollama model downloaded (llama2:7b recommended)
- 8GB RAM minimum (16GB recommended)
- Windows 10/11, Linux, or macOS

## 🚀 Installation

### Step 1: Clone/Navigate to Project
```bash
cd "E:\Data Science\ML_and_DL_project\NLP Project\Agentic Crewai Bug Hunter for ML Projects"
```

### Step 2: Run Setup (One-time)
```bash
# Double-click or run:
setup.bat
```

This will:
- ✅ Create virtual environment
- ✅ Install all Python dependencies
- ✅ Prepare the project

### Step 3: Install & Setup Ollama

1. **Download Ollama**: https://ollama.ai/download

2. **Install and start Ollama**:
```bash
# In a new terminal:
ollama serve
```

3. **Pull a model** (choose one):
```bash
# Recommended (3.8 GB) - Best quality
ollama pull llama2:7b

# OR Faster option (1.6 GB) - Good for testing
ollama pull gemma2:2b
```

4. **Verify installation**:
```bash
ollama list
```

## 🎮 Usage

### Quick Start (Recommended)

Simply double-click:
```
run_local.bat
```

This will:
- ✅ Check all prerequisites
- ✅ Verify Ollama is running
- ✅ Check model availability
- ✅ Start the application
- ✅ Open at http://localhost:8000

### Manual Start

```bash
# Activate virtual environment
venv\Scripts\activate

# Ensure Ollama is running (in separate terminal)
ollama serve

# Start application
cd app
python main.py
```

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### Analyze your code

1. Paste your ML code or error logs into the text area
2. Click "Analyze with AI"
3. Wait 30-120 seconds for AI agents to process
4. Review the comprehensive debug report
5. Copy fixes and apply them to your code

## 📁 Project Structure

```
Agentic Crewai Bug Hunter for ML Projects/
│
├── agents/                    # AI Agent definitions
│   ├── runtime_agent.py      # Runtime error debugger
│   ├── logic_agent.py         # ML logic reviewer
│   ├── fix_agent.py           # Code fix generator
│   └── performance_agent.py   # Performance optimizer
│
├── app/                       # Application code
│   ├── main.py               # FastAPI application
│   └── crew_runner.py        # CrewAI orchestration
│
├── prompts/                   # Agent system prompts
│   ├── runtime.txt           # Runtime analysis prompt
│   ├── logic.txt             # Logic review prompt
│   ├── fix.txt               # Fix generation prompt
│   └── performance.txt        # Performance prompt
│
├── static/                    # Frontend assets
│   ├── style.css             # Responsive CSS
│   └── javascript.js         # Interactive JS
│
├── templates/                 # HTML templates
│   └── index.html            # Main UI
│
├── config.py                 # Configuration management
├── llm_model.py              # LLM initialization
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── setup.bat                 # Setup script
├── run_local.bat             # Local run script
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
│
└── Documentation/
    ├── README.md             # This file
    ├── QUICKSTART.md         # Quick start guide
    ├── TESTING.md            # Testing guide
    └── COMPLETION.md         # Project summary
```

## 🛠️ Configuration

### Change Model

Edit `.env` file:
```ini
# Use llama2 (recommended)
OLLAMA_MODEL=ollama/llama2:7b

# OR use gemma2 (faster)
OLLAMA_MODEL=ollama/gemma2:2b
```

### Adjust Model Parameters

```ini
TEMPERATURE=0.4        # 0.0 = deterministic, 1.0 = creative
MAX_TOKENS=512         # Maximum response length
```

### Customize Agent Behavior

Modify prompt files in the `prompts/` directory to customize agent behavior.

## 💡 Example Use Cases

### 1. Debug PyTorch Runtime Errors
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)  # Wrong dimension!
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)  # Shape mismatch!
        return x
```

### 2. Optimize Training Performance
```python
# Slow training loop
for epoch in range(epochs):
    for data in dataloader:
        output = model(data)
        loss.backward()
        optimizer.step()
```

### 3. Fix Logic Errors
```python
# Wrong loss for classification
criterion = nn.MSELoss()  # Should be CrossEntropyLoss!
```

## 🎯 Tips for Best Results

1. **Include Context**: Provide complete code snippets with imports
2. **Add Error Logs**: Paste full tracebacks for runtime errors
3. **Specify Framework**: Mention if using PyTorch, TensorFlow, etc.
4. **Model Details**: Include model architecture and training setup
5. **Be Specific**: Describe what's not working as expected

## 🔧 Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model Not Found
```bash
# Pull the model again
ollama pull llama2:7b

# List available models
ollama list
```

### Port Already in Use
```bash
# Use a different port
# Edit app/main.py and change port=8000 to port=8001
uvicorn main:app --port 8001
```

### Slow Analysis
- Ensure Ollama is using GPU acceleration if available
- Try a smaller/faster model (gemma2:2b)
- Check system resources (RAM, GPU memory)
- Reduce MAX_TOKENS in .env file

## 📊 Performance

- **Analysis Time**: 30-120 seconds (depends on code complexity)
- **Memory Usage**: 2-4GB (model dependent)
- **GPU**: Recommended for faster inference
- **CPU**: Works but slower

## 🐳 Docker Support

### Build and run with Docker Compose

```bash
docker-compose up --build
```

### Or without Docker Compose:
```bash
docker build -t ml-bug-hunter .
docker run -p 8000:8000 ml-bug-hunter
```

**Note**: Docker setup requires Ollama to be running on host machine.

## 🧪 Testing

See [TESTING.md](TESTING.md) for comprehensive testing guide.

Quick test:
```bash
# Verify system
verify.bat

# Run application
run_local.bat
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **CrewAI** for the multi-agent framework
- **Ollama** for local LLM inference
- **FastAPI** for the web framework
- **LiteLLM** for unified LLM interface

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check [TESTING.md](TESTING.md) for troubleshooting
- Review [QUICKSTART.md](QUICKSTART.md) for quick help

## 🌟 Features

- ✅ **Complete Privacy**: Everything runs locally
- ✅ **No API Keys Required**: Uses local Ollama models
- ✅ **Fast Analysis**: Multi-agent parallel processing
- ✅ **Production Ready**: Clean, documented code
- ✅ **Easy Setup**: One-click installation and run
- ✅ **Docker Support**: Containerized deployment option

---

**Made by ❤️[Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/)❤️ ML Engineers and Researchers**

**Happy Debugging! 🧠✨🔍**
