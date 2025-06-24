import os
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client, Client
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Import your existing tools
from .browser_tools import duckduckgo_search, wiki_search, fetch_webpage_content, arxiv_search
from .math_tools import (
    add, subtract, multiply, divide, modulus, power, square_root, 
    factorial, combination, permutation, log, sin, cos, tan,
    mean, median, variance, standard_deviation, total, linear_regression
)
from .document_processing_tools import (
    save_and_read_file, read_text_file, download_file_from_url,
    extract_text_from_image, extract_text_from_pdf, extract_text_from_docx,
    analyze_csv_file, analyze_excel_file, list_excel_sheets,
    extract_archive, get_file_info, count_words_in_text
)
from .code_interpreter_tools import (
    execute_python_code, execute_bash_command, execute_sql_query,
    execute_code_multilang, create_sql_table, list_available_libraries
)
from .image_processing_tools import (
    load_image_from_file, save_image_to_file, analyze_image,
    transform_image, draw_on_image, generate_simple_image,
    combine_images, get_image_histogram
)

class GaiaAgent:
    def __init__(self, provider: str = "groq"):
        self.provider = provider
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.vector_store = self._setup_vector_store()
        self.graph = self._build_graph()
        
    def _setup_llm(self):
        """Setup LLM based on provider"""
        if self.provider == "groq":
            return ChatGroq(model="qwen-qwq-32b", temperature=0)
        else:
            raise ValueError("Only 'groq' provider supported")
    
    def _setup_tools(self):
        """Setup all available tools using your existing implementations"""
        return [
            # Web search and research tools
            duckduckgo_search,
            wiki_search,
            fetch_webpage_content,
            arxiv_search,
            
            # Mathematical tools
            add, subtract, multiply, divide, modulus, power, square_root,
            factorial, combination, permutation, log, sin, cos, tan,
            mean, median, variance, standard_deviation, total, linear_regression,
            
            # Document processing tools
            save_and_read_file, read_text_file, download_file_from_url,
            extract_text_from_image, extract_text_from_pdf, extract_text_from_docx,
            analyze_csv_file, analyze_excel_file, list_excel_sheets,
            extract_archive, get_file_info, count_words_in_text,
            
            # Code execution tools
            execute_python_code, execute_bash_command, execute_sql_query,
            execute_code_multilang, create_sql_table, list_available_libraries,
            
            # Image processing tools
            load_image_from_file, save_image_to_file, analyze_image,
            transform_image, draw_on_image, generate_simple_image,
            combine_images, get_image_histogram,
        ]
    
    def _setup_vector_store(self):
        """Setup vector store for retrieval (optional)"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            supabase: Client = create_client(
                os.environ.get("SUPABASE_URL"), 
                os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            )
            return SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents2",
                query_name="match_documents_2",
            )
        except Exception:
            return None  # Graceful degradation
    
    def _build_graph(self):
        """Build the execution graph optimized for GAIA benchmark"""
        
        # System prompt optimized for GAIA
        system_prompt = """You are an expert research assistant optimized for the GAIA benchmark.

CORE CAPABILITIES:
- Web search and information retrieval (duckduckgo_search, wiki_search, arxiv_search)
- Mathematical computation and analysis (comprehensive math tools available)
- Document processing (PDF, DOCX, CSV, Excel, images with OCR)
- Code execution in multiple languages (Python, Bash, SQL, JavaScript, R)
- Image analysis and processing (load, analyze, transform, generate)
- File system operations and data analysis

RESEARCH STRATEGY:
1. **Analyze the question type**:
   - Factual: Use wiki_search, duckduckgo_search, fetch_webpage_content
   - Mathematical: Use mathematical tools and execute_python_code for complex calculations
   - Data analysis: Use analyze_csv_file, analyze_excel_file, execute_python_code
   - Image tasks: Use image processing tools
   - Document analysis: Use appropriate extraction tools + analysis

2. **Tool selection priority**:
   - Start with the most specific tool for the task
   - Use multiple sources to cross-verify information
   - Prefer authoritative sources (Wikipedia, ArXiv) for facts
   - Use code execution for complex calculations and data analysis
   - Always verify numerical results with multiple approaches

3. **Quality assurance**:
   - Cross-reference information from multiple sources
   - Verify calculations with code execution
   - Check units and formats carefully
   - Provide precise, exact answers as requested

ANSWER FORMAT:
Think through the problem step by step, use appropriate tools, then provide your final answer in the exact format requested. End with: FINAL ANSWER: [precise answer]

For numbers: no commas, no units unless specified
For strings: no articles, write numbers as digits, exact spelling
For lists: comma-separated with exactly one space after each comma
"""

        llm_with_tools = self.llm.bind_tools(self.tools)

        def assistant(state: MessagesState):
            """Main assistant node"""
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def retriever(state: MessagesState):
            """Optional retrieval-augmented node"""
            if self.vector_store:
                try:
                    query = state["messages"][-1].content
                    similar_docs = self.vector_store.similarity_search(query, k=2)
                    
                    if similar_docs:
                        context = "\n\n".join([doc.page_content for doc in similar_docs])
                        context_msg = HumanMessage(
                            content=f"REFERENCE EXAMPLES:\n{context}\n\nORIGINAL QUERY: {query}"
                        )
                        return {"messages": state["messages"] + [context_msg]}
                except Exception:
                    pass  # Graceful degradation
            
            return {"messages": state["messages"]}

        # Build graph
        builder = StateGraph(MessagesState)
        builder.add_node("retriever", retriever)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        
        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        return builder.compile()
    
    def __call__(self, question: str) -> str:
        """Process a GAIA query - compatible with evaluation runner"""
        try:
            messages = [HumanMessage(content=question)]
            result = self.graph.invoke({"messages": messages})
            
            final_message = result["messages"][-1].content
            
            # Extract final answer if present
            if "FINAL ANSWER:" in final_message:
                return final_message.split("FINAL ANSWER:")[-1].strip()
            else:
                return final_message
                
        except Exception as e:
            return f"Error processing query: {str(e)}"

# For backward compatibility with your existing build_graph function
def build_graph(provider: str = "groq"):
    """Build graph function compatible with your existing code"""
    agent = GaiaAgent(provider=provider)
    return agent.graph