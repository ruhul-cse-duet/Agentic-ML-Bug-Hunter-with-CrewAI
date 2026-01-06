# 🚀 QUICK START GUIDE

Get up and running in **5 minutes**!

---

## ✅ Prerequisites (Check First)

- [ ] **Python 3.11.14+** installed
- [ ] **Ollama** installed
- [ ] At least **8GB RAM**
- [ ] **5GB disk space** available

---

## 🎯 3-Step Setup

### Step 1️⃣: Setup Environment (One-time)

**Option A - Double Click:**
```
Double-click: setup.bat
```

**Option B - Terminal:**
```bash
# Navigate to project
cd "E:\Data Science\ML_and_DL_project\NLP Project\Agentic Crewai Bug Hunter for ML Projects"

# Run setup
setup.bat
```

**What it does:**
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Prepares the project

---

### Step 2️⃣: Setup Ollama

**Install Ollama:**
- Download from: https://ollama.ai/download
- Install and run

**Start Ollama** (in a NEW terminal):
```bash
ollama serve
```

**Pull a Model** (choose one):
```bash
# Recommended (3.8 GB) - Best quality
ollama pull llama2:7b

# OR Faster (1.6 GB) - Good for testing
ollama pull gemma2:2b
```

**Verify:**
```bash
ollama list
```

You should see your downloaded model listed.

---

### Step 3️⃣: Run the Application

**Option A - Double Click (Easiest):**
```
Double-click: run_local.bat
```

**Option B - Terminal:**
```bash
run_local.bat
```

**What happens:**
- ✅ Checks all prerequisites
- ✅ Verifies Ollama is running  
- ✅ Confirms model available
- ✅ Starts the server
- ✅ Opens at http://localhost:8000

---

## 🌐 Access the Application

Open your browser:
```
http://localhost:8000
```

---

## 🎮 How to Use

### 1. Test with Example

Click **"Load Example"** button in the UI

### 2. Analyze Code

1. **Paste** your ML code in the textarea
2. **Click** "Analyze with AI"
3. **Wait** 30-120 seconds
4. **Review** the results

### 3. Get Results

You'll receive:
- 🔍 **Runtime Error Analysis**
- 🧩 **Logic Review** 
- 🔧 **Auto-Generated Fixes**
- ⚡ **Performance Tips**

---

## 💡 Example Test Code

Try this code (has intentional bugs):

```python
import torch
import torch.nn as nn

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)  # Wrong dimension!
    
    def forward(self, x):
        # Input: (batch, 3, 224, 224)
        x = self.conv(x)  # Output: (batch, 64, 224, 224)
        x = self.fc(x)    # Will crash! 💥
        return x

# This will fail
model = BuggyModel()
x = torch.randn(32, 3, 224, 224)
output = model(x)  # Dimension mismatch error
```

**Expected Output:**
- Identifies the dimension mismatch
- Explains why it fails
- Provides working fix
- Suggests performance improvements

---

## ⚙️ Configuration (Optional)

Edit `.env` file to change model:

```ini
# Use llama2 (recommended - better quality)
OLLAMA_MODEL=ollama/llama2:7b

# OR use gemma2 (faster - good for testing)
OLLAMA_MODEL=ollama/gemma2:2b

# Adjust parameters
TEMPERATURE=0.4        # 0.0 = strict, 1.0 = creative
MAX_TOKENS=512         # Response length
```

---

## 🔧 Troubleshooting

### ❌ "Ollama not running"

**Solution:**
```bash
# Start Ollama in a new terminal
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

---

### ❌ "Model not found"

**Solution:**
```bash
# Pull the model
ollama pull llama2:7b

# Check it's there
ollama list
```

---

### ❌ "Port 8000 in use"

**Solution:**

**Option 1 - Kill existing process:**
```bash
# Find process
netstat -ano | findstr :8000

# Kill it (replace PID with actual number)
taskkill /F /PID <PID>
```

**Option 2 - Use different port:**
Edit `app/main.py`, change:
```python
uvicorn.run("main:app", port=8001)  # Changed from 8000
```

---

### ❌ "Dependencies error"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## ⌨️ Keyboard Shortcuts

- **Ctrl + Enter** - Submit analysis
- **Escape** - Clear form (when in textarea)
- **F5** - Refresh page

---

## 📊 What to Expect

| Metric | Value |
|--------|-------|
| **Page Load** | < 2 seconds |
| **Simple Analysis** | 30-60 seconds |
| **Complex Analysis** | 60-120 seconds |
| **Memory Usage** | 2-4 GB |

---

## ✅ Success Indicators

You'll know it's working when:

1. ✅ Status shows "Ollama Connected" (top right)
2. ✅ Loading animation shows 4 processing steps
3. ✅ Detailed analysis appears
4. ✅ Can copy/download results

---

## 🎓 Tips for Best Results

1. **Complete Code**: Include all imports
2. **Error Logs**: Paste full tracebacks
3. **Framework**: Specify PyTorch/TensorFlow
4. **Context**: Add comments explaining issue
5. **Size**: Keep under 200 lines for faster analysis

---

## 🛑 How to Stop

**Method 1 - Terminal:**
```
Press CTRL + C
```

**Method 2 - Task Manager:**
```
Find Python process and end it
```

---

## 📚 Next Steps

Once running successfully:

1. ✅ Test with your real ML code
2. ✅ Try different types of bugs
3. ✅ Explore all features
4. ✅ Customize agents (prompts folder)
5. 🐳 Build Docker image (optional)

---

## 🆘 Still Having Issues?

1. **Run verification:**
   ```bash
   verify.bat
   ```

2. **Check documentation:**
   - `README.md` - Full documentation
   - `TESTING.md` - Detailed testing guide
   - `PROJECT_STRUCTURE.md` - Structure details

3. **Check logs:**
   - Terminal output for errors
   - Browser console (F12) for UI errors

---

## 🎉 Success!

If you see the web interface and can analyze code:

**Congratulations! 🎊**

You're ready to debug ML code with AI! 🧠✨

---

## 📞 Need Help?

- Review error messages carefully
- Check Ollama is running: `ollama list`
- Verify port 8000 is free
- Ensure all dependencies installed

---

**Happy Debugging! 🚀**

**Powered by CrewAI + Ollama + Local LLM** 🤖
