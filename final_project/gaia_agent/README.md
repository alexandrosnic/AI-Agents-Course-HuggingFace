# GAIA Agent - Advanced Research Assistant

A comprehensive AI agent optimized for the GAIA benchmark, featuring multi-modal capabilities including web search, mathematical computation, document processing, code execution, and image analysis.

## Features

### 🔍 Research Capabilities
- **Web Search**: DuckDuckGo, Wikipedia, ArXiv integration
- **Document Processing**: PDF, DOCX, CSV, Excel with OCR support
- **Webpage Analysis**: Content extraction and analysis

### 🧮 Mathematical Tools
- Basic arithmetic operations (add, subtract, multiply, divide)
- Advanced functions (factorial, combinations, permutations)
- Trigonometric functions (sin, cos, tan)
- Statistical analysis (mean, median, variance, standard deviation)
- Linear regression and data analysis

### 💻 Code Execution
- **Multi-language support**: Python, Bash, SQL, JavaScript, R
- **Sandboxed execution** with timeout protection
- **Data visualization** with matplotlib integration
- **DataFrame processing** with pandas

### 🖼️ Image Processing
- Image loading, analysis, and transformation
- OCR text extraction from images
- Image generation and manipulation
- Histogram analysis and visual processing

### 🏗️ Architecture
- **Token-efficient design**: Single reasoning chain with strategic tool usage
- **LangGraph integration**: Optimized workflow orchestration
- **Vector retrieval**: Optional context enhancement with Supabase
- **Graceful degradation**: Robust error handling and fallbacks

## Installation

### Prerequisites
- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd final_project/gaia_agent
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Additional System Dependencies

#### For OCR Support (Tesseract)
```bash
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS with Homebrew:
brew install tesseract

# Windows:
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### For JavaScript/Node.js Support (Optional)
```bash
# Ubuntu/Debian:
sudo apt-get install nodejs npm

# macOS with Homebrew:
brew install node

# Windows:
# Download from: https://nodejs.org/
```

#### For R Support (Optional)
```bash
# Ubuntu/Debian:
sudo apt-get install r-base

# macOS with Homebrew:
brew install r

# Windows:
# Download from: https://cran.r-project.org/
```

### 5. Environment Configuration

Create a `.env` file in the project root:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for enhanced retrieval)
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key

# For HuggingFace Space deployment
SPACE_ID=your_space_id
```

#### Getting API Keys

1. **Groq API Key** (Required):
   - Visit [Groq Console](https://console.groq.com/)
   - Sign up/login and create an API key
   - Add to `.env` file

2. **Supabase** (Optional):
   - Visit [Supabase](https://supabase.com/)
   - Create a project and get URL + service role key
   - Enables vector similarity search for enhanced retrieval

## Usage

### Local Development

#### 1. Interactive Mode
```bash
python app.py
```
This launches a Gradio interface at `http://localhost:7860`

#### 2. Command Line Testing
```python
from gaia_orchestrator import GaiaAgent

# Initialize agent
agent = GaiaAgent(provider="groq")

# Test with a question
result = agent("What is the capital of France?")
print(result)
```

#### 3. Direct Graph Usage
```python
from gaia_orchestrator import build_graph
from langchain_core.messages import HumanMessage

# Build graph
graph = build_graph(provider="groq")

# Process question
messages = [HumanMessage(content="Your question here")]
result = graph.invoke({"messages": messages})
print(result["messages"][-1].content)
```

### GAIA Benchmark Evaluation

1. **Launch the interface**:
   ```bash
   python app.py
   ```

2. **Open browser** to `http://localhost:7860`

3. **Login** with your HuggingFace account

4. **Click "Run Evaluation & Submit All Answers"**

The agent will:
- Fetch all GAIA benchmark questions
- Process each question using appropriate tools
- Submit answers to the evaluation server
- Display results and scores

### HuggingFace Spaces Deployment

1. **Create a new HuggingFace Space**:
   - Go to [HuggingFace Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the SDK

2. **Upload files**:
   ```
   gaia_agent/
   ├── app.py
   ├── gaia_orchestrator.py
   ├── requirements.txt
   ├── browser_tools.py
   ├── math_tools.py
   ├── document_processing_tools.py
   ├── code_interpreter_tools.py
   ├── image_processing_tools.py
   └── README.md
   ```

3. **Set environment variables** in Space settings:
   - `GROQ_API_KEY`: Your Groq API key
   - `SUPABASE_URL`: (Optional) Your Supabase URL
   - `SUPABASE_SERVICE_ROLE_KEY`: (Optional) Your Supabase key

4. **The Space will automatically deploy** and be available at:
   `https://huggingface.co/spaces/your-username/your-space-name`

## Configuration

### Tool Selection
Tools are automatically selected based on question type:

- **Factual questions**: Web search, Wikipedia, ArXiv
- **Mathematical problems**: Math tools, code execution
- **Data analysis**: CSV/Excel tools, Python execution
- **Image tasks**: Image processing tools
- **Document analysis**: PDF/DOCX extraction, OCR

### Model Configuration
Currently supports Groq models. To change model:

```python
# In gaia_orchestrator.py
def _setup_llm(self):
    if self.provider == "groq":
        return ChatGroq(
            model="qwen-qwq-32b",  # Change model here
            temperature=0
        )
```

Available Groq models:
- `qwen-qwq-32b` (default, good reasoning)
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. OCR Not Working
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS

# Verify installation
tesseract --version
```

#### 3. API Key Issues
- Verify `.env` file exists and contains correct keys
- Check Groq console for API key validity
- Ensure no extra spaces in environment variables

#### 4. Memory Issues
```bash
# For large datasets, increase system memory or reduce batch size
# Monitor memory usage with:
htop  # Linux/Mac
# Task Manager on Windows
```

#### 5. Tool Execution Timeouts
```python
# Increase timeout in code_interpreter_tools.py
class SafeCodeInterpreter:
    def __init__(self, max_execution_time=60):  # Increase from 30
```

### Debugging

#### Enable Verbose Logging
```python
# In gaia_orchestrator.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test Individual Tools
```python
from browser_tools import wiki_search
result = wiki_search("Python programming")
print(result)
```

## Performance Optimization

### Token Efficiency
- Uses single reasoning chain (not multi-agent)
- Strategic tool selection based on question type
- Minimal LLM calls with maximum tool utilization

### Speed Optimization
- Parallel tool execution where possible
- Efficient caching with vector store
- Optimized prompts for direct answers

### Memory Management
- Automatic cleanup of temporary files
- Limited output sizes for large datasets
- Graceful degradation for resource constraints

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Make changes** and test thoroughly
4. **Commit changes**: `git commit -m "Add feature"`
5. **Push branch**: `git push origin feature-name`
6. **Create Pull Request**

### Development Guidelines
- Follow existing code structure
- Add proper error handling
- Include type hints
- Test all tool integrations
- Update documentation

## License

[Specify your license here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create new issue with detailed description
4. Include error logs and environment details

## Changelog

### v1.0.0
- Initial release with full GAIA benchmark support
- Multi-modal tool integration
- LangGraph workflow optimization
- Gradio evaluation interface