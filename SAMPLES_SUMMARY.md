# LangChain Learning Samples

This repository now includes 30 complete, hands-on LangChain samples organized into three difficulty levels.

## 📂 Folder Structure

```
pages/
├── beginner/       # 10 foundational samples
├── advanced/       # 10 intermediate samples
└── expert/         # 10 advanced samples
```

## 🎓 Beginner Level (pages/beginner/)

Perfect for those new to LangChain. Each sample teaches one core concept.

1. **01_Simple_Chat.py** - Your first chat with Ollama using LangChain
2. **02_Prompt_Template.py** - Create reusable prompt templates
3. **03_Output_Parser.py** - Parse AI responses into structured formats
4. **04_Simple_Chain.py** - Chain together prompts, LLMs, and parsers
5. **05_Chat_History.py** - Maintain conversation context
6. **06_Text_Embeddings.py** - Generate and compare text embeddings
7. **07_Document_Loader.py** - Load various document formats
8. **08_Text_Splitter.py** - Split documents into chunks
9. **09_Vector_Store.py** - Store and search document embeddings
10. **10_Simple_RAG.py** - Complete RAG system from scratch

**Learning Path:** Start at 01 and work through sequentially. Each builds on previous concepts.

## 🚀 Advanced Level (pages/advanced/)

For developers comfortable with basics, ready for production patterns.

1. **01_Custom_Chain.py** - Build sophisticated custom chains
2. **02_Conversation_Buffer.py** - Advanced memory management strategies
3. **03_MultiQuery_RAG.py** - Improve RAG with multiple query variations
4. **04_Streaming_Chat.py** - Stream responses token-by-token
5. **05_Function_Calling.py** - LLMs calling external functions
6. **06_Agent_Basics.py** - Autonomous agents with tools
7. **07_PDF_RAG_Advanced.py** - RAG with citations and metadata
8. **08_Web_Scraping_QA.py** - Q&A from web content
9. **09_Structured_Output.py** - Pydantic models for validation
10. **10_Routing_Chain.py** - Dynamic chain selection

**Learning Path:** Focus on areas relevant to your project. All samples are independent.

## 🔬 Expert Level (pages/expert/)

For experienced developers building production systems.

1. **01_LangGraph_Basics.py** - Introduction to graph-based workflows
2. **02_Multi_Agent_System.py** - Coordinate multiple specialized agents
3. **03_Custom_Tools.py** - Build custom tools for agents
4. **04_Agentic_RAG.py** - RAG where agents decide when to retrieve
5. **05_Conditional_Graph.py** - Dynamic branching in workflows
6. **06_Human_In_Loop.py** - Interactive approval workflows
7. **07_Multimodal_Agent.py** - Work with text and images
8. **08_Advanced_Memory.py** - Custom multi-layer memory systems
9. **09_Self_RAG.py** - Self-correcting RAG with evaluation
10. **10_Production_Pipeline.py** - Complete production system

**Learning Path:** These samples showcase architectural patterns for complex applications.

## 🛠️ Technical Details

### Prerequisites
- Python 3.11+
- Ollama installed and running
- Required packages (see `pyproject.toml`):
  - langchain
  - langchain-ollama
  - langchain-community
  - langgraph
  - streamlit
  - And others...

### Running Samples

```bash
# Install dependencies
uv sync

# Run any sample with Streamlit
streamlit run pages/beginner/01_Simple_Chat.py
```

### Key Features
- ✅ **Self-contained**: Each sample runs independently
- ✅ **Educational**: Inline explanations and learning sections
- ✅ **Local-first**: Uses Ollama for privacy and control
- ✅ **Interactive**: Built with Streamlit for hands-on learning
- ✅ **Production-ready patterns**: Real-world architectures

## 📚 Learning Resources

Each sample includes:
- **What you'll learn** section explaining concepts
- **How it works** code patterns and explanations
- **Use cases** real-world applications
- **Next steps** suggestions for further learning
- **Best practices** tips and common pitfalls

## 🎯 Recommended Learning Paths

### Path 1: RAG Developer
1. Beginner: 01, 02, 06, 07, 08, 09, 10
2. Advanced: 03, 07, 08, 09
3. Expert: 04, 09

### Path 2: Agent Developer
1. Beginner: 01, 02, 03, 04, 05
2. Advanced: 05, 06, 10
3. Expert: 01, 02, 03, 04, 05

### Path 3: Production Engineer
1. Advanced: 01, 02, 04, 09, 10
2. Expert: All samples

## 🤝 Contributing

These samples are designed to be:
- Clear and well-documented
- Following project conventions
- Using existing helper libraries
- Tested and working with Ollama

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Note**: All samples use local Ollama models. No API keys required!
