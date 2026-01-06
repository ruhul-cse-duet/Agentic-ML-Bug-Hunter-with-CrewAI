# 🧪 Testing Guide - Local Setup & Verification

## Quick Start (3 Steps)

### 1️⃣ Setup Environment
```bash
# Run the setup script
setup.bat

# This will:
# - Create virtual environment
# - Install all dependencies
# - Upgrade pip
```

### 2️⃣ Start Ollama
Open a **new terminal** and run:
```bash
# Start Ollama server
ollama serve

# In another terminal, pull the model (first time only)
ollama pull unsloth/deepseek-r1-0528-qwen3-gguf

# Verify it's working
ollama list
```

### 3️⃣ Run the Application
```bash
# Run the app
run_local.bat

# Or manually:
cd app
python main.py
```

Open browser: **http://localhost:8000**

---

## ✅ Verification Checklist

### Before Running
- [ ] Python 3.11.14 installed
- [ ] Virtual environment created (`venv` folder exists)
- [ ] Dependencies installed (check with `pip list`)
- [ ] Ollama installed and running (`ollama list` works)
- [ ] DeepSeek-R1 model downloaded

### During Testing
- [ ] Web page loads at http://localhost:8000
- [ ] Ollama status shows "Connected" (top right)
- [ ] Can paste code in textarea
- [ ] "Analyze with AI" button works
- [ ] Loading animation shows 4 steps
- [ ] Results appear after analysis
- [ ] Can copy report to clipboard
- [ ] Can download report as .txt file

### UI/UX Testing
- [ ] Responsive on mobile (test with browser dev tools)
- [ ] All animations work smoothly
- [ ] Character counter updates
- [ ] "Load Example" button works
- [ ] "Clear" button works
- [ ] Keyboard shortcuts work (Ctrl+Enter to submit)

---

## 🧪 Test Cases

### Test Case 1: Simple PyTorch Bug
```python
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

# Create model and test
model = TestModel()
x = torch.randn(32, 20)  # Wrong input size!
output = model(x)
```

**Expected**: Should detect dimension mismatch (expects 10, got 20)

---

### Test Case 2: Wrong Loss Function
```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
criterion = nn.MSELoss()  # Wrong for classification!

# Simulating classification task
logits = model(torch.randn(32, 10))
labels = torch.randint(0, 5, (32,))
loss = criterion(logits, labels)  # This will fail!
```

**Expected**: Should suggest CrossEntropyLoss for classification

---

### Test Case 3: Memory Issue (Gradient Accumulation)
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1000, 5000),
    nn.ReLU(),
    nn.Linear(5000, 10000),
    nn.ReLU(),
    nn.Linear(10000, 1000)
)

# Large batch might cause OOM
batch_size = 1024
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    data = torch.randn(batch_size, 1000)
    output = model(data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
```

**Expected**: Should suggest gradient accumulation or smaller batch size

---

## 🐛 Common Issues & Solutions

### Issue 1: "Cannot connect to Ollama"
**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Check if model exists
ollama list
```

---

### Issue 2: "Model not found"
**Solution**:
```bash
# Pull the model
ollama pull unsloth/deepseek-r1-0528-qwen3-gguf

# Or use alternative model
ollama pull deepseek-r1

# Update config.py with the model name you have
```

---

### Issue 3: Port 8000 already in use
**Solution**:
```bash
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process or use different port
python main.py --port 8001

# Or in code, change uvicorn.run() port parameter
```

---

### Issue 4: Dependencies not installing
**Solution**:
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install one by one
pip install fastapi
pip install uvicorn
pip install crewai
# ... etc

# Or with verbose output
pip install -r requirements.txt -v
```

---

### Issue 5: Virtual environment activation fails
**Solution**:
```bash
# On Windows PowerShell, you might need to enable scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
venv\Scripts\activate

# Or use cmd.exe instead of PowerShell
```

---

## 📊 Performance Testing

### Expected Performance Metrics
- **Initial Load**: < 2 seconds
- **Ollama Connection Check**: < 1 second
- **Simple Analysis**: 30-60 seconds
- **Complex Analysis**: 60-120 seconds
- **Memory Usage**: 2-4 GB (depends on model)

### Load Testing (Optional)
```python
# Test multiple concurrent requests
import requests
import concurrent.futures

def test_analyze():
    data = {"code": "import torch\nx = torch.randn(10, 10)"}
    response = requests.post("http://localhost:8000/analyze", data=data)
    return response.status_code

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(test_analyze) for _ in range(5)]
    results = [f.result() for f in futures]
    print(f"Success rate: {results.count(200)}/5")
```

---

## 🔍 Debugging Tips

### Enable Verbose Logging
Edit `app/main.py`:
```python
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    log_level="debug"  # Changed from "info"
)
```

### Check Ollama Logs
```bash
# On Windows
Get-EventLog -LogName Application -Source Ollama

# Or check Ollama terminal output
```

### Test LLM Connection Directly
```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="unsloth/deepseek-r1-0528-qwen3-gguf",
    base_url="http://localhost:11434",
    temperature=0.2
)

response = llm.invoke("Hello, are you working?")
print(response)
```

---

## ✅ Final Checklist Before Docker Build

- [ ] All tests pass locally
- [ ] No errors in console/terminal
- [ ] UI looks good on desktop & mobile
- [ ] All features work as expected
- [ ] Ollama integration stable
- [ ] Error handling works properly
- [ ] Performance is acceptable

---

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** in terminal
2. **Verify Ollama** is running: `ollama list`
3. **Test health endpoint**: http://localhost:8000/health
4. **Check browser console** for JavaScript errors (F12)
5. **Review error messages** carefully

---

**Ready to proceed?** Once local testing is complete, you can build Docker image! 🐳
