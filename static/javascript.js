// Check Ollama status on page load
async function checkOllamaStatus() {
    const statusIndicator = document.getElementById('ollamaStatus');
    const statusText = statusIndicator.querySelector('.status-text');
    
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.ollama) {
            statusIndicator.classList.add('connected');
            statusIndicator.classList.remove('error');
            statusText.textContent = 'Ollama Connected';
        } else {
            statusIndicator.classList.add('error');
            statusIndicator.classList.remove('connected');
            statusText.textContent = 'Ollama Not Connected';
        }
    } catch (error) {
        statusIndicator.classList.add('error');
        statusIndicator.classList.remove('connected');
        statusText.textContent = 'Connection Error';
    }
}

// Character counter for textarea
const textarea = document.getElementById('code');
const charCount = document.querySelector('.char-count');

if (textarea && charCount) {
    textarea.addEventListener('input', function() {
        const count = this.value.length;
        charCount.textContent = `${count.toLocaleString()} characters`;
        
        if (count > 10000) {
            charCount.style.color = 'var(--warning-color)';
        } else {
            charCount.style.color = 'var(--text-muted)';
        }
    });
    
    // Update count on page load
    charCount.textContent = `${textarea.value.length.toLocaleString()} characters`;
}

// Form submission with enhanced loading animation
const analyzeForm = document.getElementById('analyzeForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');

if (analyzeForm) {
    analyzeForm.addEventListener('submit', function(e) {
        const code = textarea.value.trim();
        
        if (!code || code.length < 10) {
            e.preventDefault();
            showNotification('❌ Please enter at least 10 characters of code!', 'error');
            return;
        }
        
        // Show loading animation
        loading.style.display = 'block';
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;
        
        // Hide any previous results
        const results = document.getElementById('results');
        if (results) {
            results.style.display = 'none';
        }
        
        // Animate loading steps
        animateLoadingSteps();
        
        // Scroll to loading section
        setTimeout(() => {
            loading.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    });
}

// Enhanced step animation
function animateLoadingSteps() {
    const steps = document.querySelectorAll('.step');
    let currentStep = 0;
    
    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            // Activate current step
            steps[currentStep].classList.add('active');
            steps[currentStep].querySelector('.step-status').textContent = 'Processing...';
            
            // Complete previous step
            if (currentStep > 0) {
                steps[currentStep - 1].classList.remove('active');
                steps[currentStep - 1].classList.add('complete');
                steps[currentStep - 1].querySelector('.step-status').textContent = 'Complete';
            }
            
            currentStep++;
        } else {
            clearInterval(interval);
            // Mark last step as complete
            if (steps.length > 0) {
                steps[steps.length - 1].classList.remove('active');
                steps[steps.length - 1].classList.add('complete');
                steps[steps.length - 1].querySelector('.step-status').textContent = 'Complete';
            }
        }
    }, 2000);
}

// Clear form function
function clearForm() {
    const confirmed = confirm('Are you sure you want to clear the code input?');
    if (confirmed) {
        textarea.value = '';
        textarea.focus();
        if (charCount) {
            charCount.textContent = '0 characters';
        }
        showNotification('✅ Input cleared', 'success');
    }
}

// Load example code
function loadExample() {
    const exampleCode = `import torch
import torch.nn as nn
import torch.optim as optim

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # Wrong dimensions!
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Input: (batch, 3, 224, 224)
        x = torch.relu(self.conv1(x))  # (batch, 64, 224, 224)
        x = torch.relu(self.conv2(x))  # (batch, 128, 224, 224)
        x = x.view(x.size(0), -1)      # Flatten - what's the actual size?
        x = torch.relu(self.fc1(x))    # This will crash!
        x = self.fc2(x)
        return x

# Training code with issues
model = BuggyModel()
criterion = nn.MSELoss()  # Wrong loss for classification!
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Learning rate too high!

# This will produce errors...
input_tensor = torch.randn(32, 3, 224, 224)
output = model(input_tensor)`;

    textarea.value = exampleCode;
    textarea.focus();
    
    if (charCount) {
        charCount.textContent = `${exampleCode.length.toLocaleString()} characters`;
    }
    
    showNotification('💡 Example loaded successfully!', 'success');
}

// Copy results to clipboard
function copyResults() {
    const resultContent = document.getElementById('resultContent');
    if (!resultContent) return;
    
    const text = resultContent.textContent;
    
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('✅ Report copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

// Fallback copy method
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showNotification('✅ Report copied to clipboard!', 'success');
    } catch (err) {
        console.error('Failed to copy:', err);
        showNotification('❌ Failed to copy. Please select and copy manually.', 'error');
    }
    
    document.body.removeChild(textArea);
}

// Download results as text file
function downloadResults() {
    const resultContent = document.getElementById('resultContent');
    if (!resultContent) return;
    
    const text = resultContent.textContent;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const filename = `ml-bug-report-${timestamp}.txt`;
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    showNotification('💾 Report downloaded successfully!', 'success');
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.textContent = message;
    
    const colors = {
        success: 'var(--success-color)',
        error: 'var(--danger-color)',
        warning: 'var(--warning-color)',
        info: 'var(--info-color)'
    };
    
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        animation: slideInRight 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        max-width: 320px;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize on page load
window.addEventListener('load', function() {
    // Focus textarea
    if (textarea) {
        textarea.focus();
    }
    
    // Check Ollama status
    checkOllamaStatus();
    
    // Recheck status every 30 seconds
    setInterval(checkOllamaStatus, 30000);
    
    // Add timestamp to results
    const resultsHeader = document.querySelector('.results-header .timestamp');
    if (resultsHeader) {
        const now = new Date();
        resultsHeader.textContent = now.toLocaleString();
    }
});

// Handle page visibility - pause/resume status checks
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        checkOllamaStatus();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (analyzeForm && !analyzeBtn.disabled) {
            analyzeForm.requestSubmit();
        }
    }
    
    // Escape to clear form (with confirmation)
    if (e.key === 'Escape' && textarea === document.activeElement) {
        e.preventDefault();
        clearForm();
    }
});

console.log('🧠 Agentic ML Bug Hunter - UI Loaded Successfully!');
