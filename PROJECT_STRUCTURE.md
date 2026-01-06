# 📁 Project Structure Documentation

Complete documentation of the Agentic ML Bug Hunter project structure.

---

## 🌳 Directory Tree

```
Agentic Crewai Bug Hunter for ML Projects/
│
├── 📁 agents/                      # AI Agent Definitions
│   ├── 📄 runtime_agent.py        # Runtime error debugger agent
│   ├── 📄 logic_agent.py          # ML logic reviewer agent
│   ├── 📄 fix_agent.py            # Code fix generator agent
│   ├── 📄 performance_agent.py    # Performance optimizer agent
│   └── 📁 __pycache__/            # Python cache (auto-generated)
│
├── 📁 app/                         # Application Entry Point
│   ├── 📄 main.py                 # FastAPI application server
│   ├── 📄 crew_runner.py          # CrewAI orchestration logic
│   └── 📁 __pycache__/            # Python cache (auto-generated)
│
├── 📁 prompts/                     # Agent System Prompts
│   ├── 📄 runtime.txt             # Runtime analysis instructions
│   ├── 📄 logic.txt               # Logic review instructions
│   ├── 📄 fix.txt                 # Fix generation instructions
│   └── 📄 performance.txt         # Performance optimization instructions
│
├── 📁 static/                      # Frontend Assets
│   ├── 📄 style.css               # Responsive UI styling
│   └── 📄 javascript.js           # Interactive UI logic
│
├── 📁 templates/                   # HTML Templates
│   └── 📄 index.html              # Main web interface
│
├── 📁 .idea/                       # IDE Configuration (PyCharm/IntelliJ)
│   ├── 📄 .gitignore
│   ├── 📄 misc.xml
│   ├── 📄 modules.xml
│   ├── 📄 workspace.xml
│   └── 📁 inspectionProfiles/
│
├── 📁 __pycache__/                 # Root-level Python cache
│
├── 📄 config.py                    # Configuration management
├── 📄 llm_model.py                 # LLM initialization & setup
├── 📄 .env                         # Environment variables (API keys, model config)
├── 📄 requirements.txt             # Python dependencies
│
├── 📄 setup.bat                    # Environment setup script (Windows)
├── 📄 run_local.bat               # Local launch script (Windows)
├── 📄 verify.bat                  # System verification script
│
├── 📄 Dockerfile                   # Docker container configuration
├── 📄 docker-compose.yml          # Docker Compose orchestration
│
├── 📄 README.md                    # Main documentation
├── 📄 README_BN.md                 # Bangla documentation
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 TESTING.md                   # Testing procedures
├── 📄 COMPLETION.md                # Project completion summary
├── 📄 PROJECT_STRUCTURE.md         # This file
│
├── 📄 Demo_image.png               # Screenshot/demo image
└── 📄 demo_img_2.png               # Additional screenshot
```

---

## 📋 File Descriptions

### 🤖 Agents (`agents/`)

#### `runtime_agent.py`
- **Purpose**: Analyzes runtime errors in ML code
- **Responsibilities**:
  - Detect CUDA/GPU errors
  - Identify tensor shape mismatches
  - Find memory issues (OOM, leaks)
  - Verify device placements
  - Check DataLoader configuration
- **LLM**: Uses Ollama model configured in .env
- **Prompt Source**: `prompts/runtime.txt`

#### `logic_agent.py`
- **Purpose**: Reviews ML model logic and architecture
- **Responsibilities**:
  - Validate model architecture
  - Check loss function correctness
  - Review optimizer configuration
  - Identify silent bugs (no errors but wrong results)
  - Verify data preprocessing logic
- **LLM**: Uses Ollama model configured in .env
- **Prompt Source**: `prompts/logic.txt`

#### `fix_agent.py`
- **Purpose**: Generates code fixes for identified issues
- **Responsibilities**:
  - Create production-ready patches
  - Include necessary imports
  - Add clear comments
  - Follow Python/PyTorch best practices
  - Maintain code readability
- **LLM**: Uses Ollama model configured in .env
- **Prompt Source**: `prompts/fix.txt`

#### `performance_agent.py`
- **Purpose**: Suggests performance optimizations
- **Responsibilities**:
  - Improve training speed
  - Reduce memory usage
  - Optimize GPU utilization
  - Suggest batch size improvements
  - Recommend AMP (Automatic Mixed Precision)
- **LLM**: Uses Ollama model configured in .env
- **Prompt Source**: `prompts/performance.txt`

---

### 🌐 Application (`app/`)

#### `main.py`
- **Purpose**: FastAPI web server
- **Key Features**:
  - Serves web interface at http://localhost:8000
  - Handles code analysis requests
  - Provides health check endpoint
  - CORS enabled for API access
  - Static file serving
- **Endpoints**:
  - `GET /` - Main web interface
  - `POST /analyze` - Code analysis
  - `GET /health` - Server health check
- **Dependencies**: FastAPI, Uvicorn, Jinja2

#### `crew_runner.py`
- **Purpose**: CrewAI orchestration
- **Key Features**:
  - Manages 4 AI agents
  - Coordinates sequential task execution
  - Aggregates agent outputs
  - Handles errors gracefully
- **Process Flow**:
  1. Runtime Error Analysis
  2. ML Logic Review (uses context from step 1)
  3. Code Fix Generation (uses context from steps 1-2)
  4. Performance Optimization (uses context from steps 1-3)

---

### 📝 Prompts (`prompts/`)

#### `runtime.txt`
- **Content**: System prompt for runtime error debugger
- **Focus**: Technical expertise in PyTorch, CUDA, tensor operations
- **Format**: Plain text instruction set

#### `logic.txt`
- **Content**: System prompt for ML logic reviewer
- **Focus**: Deep learning architecture, loss functions, optimizers
- **Format**: Plain text instruction set

#### `fix.txt`
- **Content**: System prompt for code fix generator
- **Focus**: Production-grade coding, best practices
- **Format**: Plain text instruction set

#### `performance.txt`
- **Content**: System prompt for performance optimizer
- **Focus**: GPU optimization, AMP, memory management
- **Format**: Plain text instruction set

---

### 🎨 Frontend (`static/` & `templates/`)

#### `static/style.css`
- **Purpose**: Responsive UI styling
- **Features**:
  - Dark theme with gradients
  - Mobile-responsive design
  - Smooth animations
  - Modern card-based layout
  - Loading indicators
- **Lines**: ~800 lines of CSS
- **Framework**: Pure CSS (no preprocessor)

#### `static/javascript.js`
- **Purpose**: Interactive UI functionality
- **Features**:
  - Form submission handling
  - Loading animation
  - Copy to clipboard
  - Download results
  - Ollama status checking
  - Character counter
  - Keyboard shortcuts (Ctrl+Enter, Escape)
- **Lines**: ~200 lines of JavaScript
- **Dependencies**: None (vanilla JS)

#### `templates/index.html`
- **Purpose**: Main web interface
- **Features**:
  - Code input textarea
  - Real-time status indicator
  - 4-step loading animation
  - Results display
  - Copy/Download buttons
  - Example loader
- **Template Engine**: Jinja2
- **Responsive**: Mobile, Tablet, Desktop

---

### ⚙️ Configuration

#### `config.py`
- **Purpose**: Centralized configuration management
- **Settings**:
  - HuggingFace API configuration (optional)
  - Ollama model selection
  - Model parameters (temperature, max_tokens)
  - File upload settings
  - Validation logic
- **Features**: Environment variable loading, config validation

#### `llm_model.py`
- **Purpose**: LLM initialization
- **Functionality**:
  - Creates LLM instance from config
  - Sets up Ollama connection
  - Handles initialization errors
  - Returns configured LLM for agents
- **Dependencies**: crewai.LLM, config.py

#### `.env`
- **Purpose**: Environment variables
- **Contains**:
  - OLLAMA_MODEL (primary model to use)
  - HUGGINGFACE_API_KEY (optional, for fallback)
  - HUGGINGFACE_MODEL (optional)
  - TEMPERATURE (model creativity)
  - MAX_TOKENS (response length)
  - CREWAI_TRACING_ENABLED (debugging)
- **Security**: Should NOT be committed to Git (contains API keys)

---

### 📦 Dependencies

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Key Packages**:
  - `fastapi==0.124.0` - Web framework
  - `uvicorn==0.35.0` - ASGI server
  - `crewai==1.7.2` - Multi-agent framework
  - `crewai-tools==1.7.2` - Agent tools
  - `langchain>=1.1.1` - LLM framework
  - `langchain-community>=0.4.1` - Community integrations
  - `ollama>=0.6.1` - Ollama client
  - `litellm` - Unified LLM interface
  - `pydantic` - Data validation
  - `jinja2` - Template engine
  - `python-dotenv` - Environment variable loading
- **Total Packages**: ~70+ (with dependencies)

---

### 🚀 Scripts

#### `setup.bat` (Windows)
- **Purpose**: One-time project setup
- **Actions**:
  1. Check Python installation
  2. Create virtual environment
  3. Upgrade pip
  4. Install dependencies from requirements.txt
  5. Display next steps
- **Usage**: Double-click or run from terminal

#### `run_local.bat` (Windows)
- **Purpose**: Start the application
- **Checks**:
  1. Virtual environment exists
  2. Ollama service is running
  3. Model is available
  4. Configuration file exists
  5. Required files are present
- **Actions**:
  - Activates virtual environment
  - Starts FastAPI server
  - Opens at http://localhost:8000
- **Usage**: Double-click or run from terminal

#### `verify.bat` (Windows)
- **Purpose**: System verification
- **Checks**:
  1. Python installation
  2. Virtual environment
  3. Ollama installation
  4. Ollama service status
  5. Model availability
  6. Dependencies installed
  7. Port availability
  8. Required files
- **Output**: Pass/Fail report with troubleshooting

---

### 🐳 Docker

#### `Dockerfile`
- **Purpose**: Container image definition
- **Base Image**: python:3.11-slim
- **Setup**:
  1. Install dependencies
  2. Copy application code
  3. Expose port 8000
  4. Start FastAPI server

#### `docker-compose.yml`
- **Purpose**: Multi-container orchestration
- **Services**:
  - `ollama`: Ollama service container
  - `app`: FastAPI application
- **Volumes**: Persistent Ollama model storage
- **Networks**: Internal communication

---

### 📚 Documentation

#### `README.md`
- **Content**: Main project documentation
- **Sections**:
  - Features
  - Installation
  - Usage
  - Configuration
  - Troubleshooting
  - Examples

#### `README_BN.md`
- **Content**: Bangla language documentation
- **Purpose**: Accessibility for Bangla speakers

#### `QUICKSTART.md`
- **Content**: 5-minute quick start guide
- **Purpose**: Get users running quickly

#### `TESTING.md`
- **Content**: Comprehensive testing procedures
- **Includes**: Test cases, expected results, troubleshooting

#### `COMPLETION.md`
- **Content**: Project completion summary
- **Includes**: What was completed, statistics, next steps

---

## 🔄 Data Flow

### Analysis Request Flow

```
1. User Input
   └─> Browser (templates/index.html)
       └─> JavaScript (static/javascript.js)
           └─> POST /analyze

2. Server Processing
   └─> FastAPI (app/main.py)
       └─> crew_runner.py
           ├─> runtime_agent.py (Step 1)
           ├─> logic_agent.py (Step 2)
           ├─> fix_agent.py (Step 3)
           └─> performance_agent.py (Step 4)

3. LLM Interaction
   └─> llm_model.py
       └─> config.py (.env settings)
           └─> Ollama (localhost:11434)
               └─> llama2:7b model

4. Response Generation
   └─> Aggregated Results
       └─> HTML Template Rendering
           └─> Browser Display
```

---

## 🎯 Key Design Patterns

### 1. Multi-Agent Architecture
- **Pattern**: Specialized agents with specific roles
- **Benefits**: Focused expertise, parallel processing
- **Implementation**: CrewAI framework

### 2. Configuration Management
- **Pattern**: Environment-based configuration
- **Benefits**: Easy customization, secure API key storage
- **Implementation**: python-dotenv + config.py

### 3. Template-Based Prompts
- **Pattern**: Externalized prompt templates
- **Benefits**: Easy customization, version control
- **Implementation**: Text files in prompts/

### 4. REST API Design
- **Pattern**: RESTful endpoints
- **Benefits**: Standard interface, easy integration
- **Implementation**: FastAPI

---

## 🔐 Security Considerations

### API Keys
- Stored in `.env` (not committed to Git)
- Loaded via python-dotenv
- Validated in config.py

### User Input
- Sanitized before processing
- No direct code execution
- LLM analysis only

### Local Processing
- Everything runs locally
- No data sent to external servers (except chosen LLM provider)
- Full user control

---

## 📊 Performance Metrics

### File Sizes
- Total Project: ~5 MB (excluding venv)
- Python Code: ~50 KB
- Frontend: ~30 KB
- Documentation: ~200 KB

### Dependencies
- Production: ~70 packages
- Development: Add testing, linting tools

### Execution
- Startup Time: ~3-5 seconds
- Analysis Time: 30-120 seconds (model dependent)
- Memory Usage: 2-4 GB (model dependent)

---

## 🛠️ Development

### Adding New Agents
1. Create agent file in `agents/`
2. Create prompt file in `prompts/`
3. Import in `crew_runner.py`
4. Add to agent list and task sequence

### Modifying UI
- Edit `templates/index.html` for structure
- Edit `static/style.css` for styling
- Edit `static/javascript.js` for functionality

### Changing Models
- Update `OLLAMA_MODEL` in `.env`
- Pull new model with Ollama
- Restart application

---

## 📝 Notes

- **Python Version**: 3.11.14 (tested)
- **OS**: Windows (scripts), Linux/Mac (compatible)
- **Browser**: Modern browsers (Chrome, Firefox, Edge, Safari)
- **GPU**: Optional (speeds up inference)

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
