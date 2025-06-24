import os
import io
import sys
import uuid
import base64
import traceback
import contextlib
import tempfile
import subprocess
import sqlite3
import shutil
import signal
from typing import Dict, List, Any, Optional, Union
from smolagents import tool

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    np = None
    pd = None
    plt = None
    Image = None

try:
    import sympy
    import scipy
    from sklearn import datasets
except ImportError:
    sympy = None
    scipy = None
    datasets = None

class SafeCodeInterpreter:
    """A safe code interpreter with multi-language support and security measures."""
    
    def __init__(self, max_execution_time=30, working_directory=None):
        """Initialize the code interpreter with safety measures."""
        self.max_execution_time = max_execution_time
        self.working_directory = working_directory or tempfile.mkdtemp(prefix="code_exec_")
        
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        
        # Set up Python globals with available libraries
        self.python_globals = {
            "__builtins__": {
                'print': print, 'len': len, 'range': range, 'str': str, 'int': int, 
                'float': float, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
                'set': set, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
                'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
                'round': round, 'pow': pow, 'type': type, 'isinstance': isinstance,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
            }
        }
        
        # Add available libraries to globals
        if np is not None:
            self.python_globals["np"] = np
            self.python_globals["numpy"] = np
        if pd is not None:
            self.python_globals["pd"] = pd
            self.python_globals["pandas"] = pd
        if plt is not None:
            self.python_globals["plt"] = plt
            self.python_globals["matplotlib"] = plt
        if Image is not None:
            self.python_globals["Image"] = Image
        if sympy is not None:
            self.python_globals["sympy"] = sympy
        if scipy is not None:
            self.python_globals["scipy"] = scipy
        if datasets is not None:
            self.python_globals["datasets"] = datasets
            
        # Add standard library modules
        import math, random, statistics, datetime, collections, itertools, functools
        import operator, re, json, uuid as uuid_module, tempfile as tempfile_module
        
        self.python_globals.update({
            "math": math, "random": random, "statistics": statistics,
            "datetime": datetime, "collections": collections, "itertools": itertools,
            "functools": functools, "operator": operator, "re": re, "json": json,
            "uuid": uuid_module, "tempfile": tempfile_module
        })
        
        self.temp_sqlite_db = os.path.join(self.working_directory, "temp_db.sqlite")

    def cleanup(self):
        """Clean up temporary files and directories."""
        try:
            if os.path.exists(self.working_directory):
                shutil.rmtree(self.working_directory)
        except Exception:
            pass

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute the provided code in the selected programming language."""
        language = language.lower().strip()
        execution_id = str(uuid.uuid4())[:8]
        
        result = {
            "execution_id": execution_id,
            "language": language,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": [],
            "files_created": []
        }
        
        try:
            if language in ["python", "py"]:
                return self._execute_python(code, execution_id)
            elif language in ["bash", "shell", "sh"]:
                return self._execute_bash(code, execution_id)
            elif language in ["sql", "sqlite"]:
                return self._execute_sql(code, execution_id)
            elif language in ["javascript", "js", "node"]:
                return self._execute_javascript(code, execution_id)
            elif language in ["r"]:
                return self._execute_r(code, execution_id)
            else:
                result["stderr"] = f"Unsupported language: {language}. Supported: python, bash, sql, javascript, r"
        except Exception as e:
            result["stderr"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_python(self, code: str, execution_id: str) -> Dict[str, Any]:
        """Execute Python code with safety measures."""
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        result = {
            "execution_id": execution_id,
            "language": "python",
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": [],
            "files_created": []
        }
        
        try:
            exec_dir = os.path.join(self.working_directory, execution_id)
            os.makedirs(exec_dir, exist_ok=True)
            
            # Set up matplotlib for headless operation
            if plt is not None:
                plt.switch_backend('Agg')
                plt.clf()  # Clear any existing plots
            
            # Create a copy of globals for this execution
            local_globals = self.python_globals.copy()
            local_globals["__execution_dir__"] = exec_dir
            
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                # Use compile and exec for better error handling
                compiled_code = compile(code, f"<execution_{execution_id}>", "exec")
                exec_result = exec(compiled_code, local_globals)
                
                # Capture any matplotlib plots
                if plt is not None and plt.get_fignums():
                    for i, fig_num in enumerate(plt.get_fignums()):
                        try:
                            fig = plt.figure(fig_num)
                            img_path = os.path.join(exec_dir, f"plot_{i}.png")
                            fig.savefig(img_path, dpi=100, bbox_inches='tight')
                            
                            with open(img_path, "rb") as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                result["plots"].append({
                                    "figure_number": fig_num,
                                    "filename": f"plot_{i}.png",
                                    "data": img_data
                                })
                            result["files_created"].append(img_path)
                        except Exception as plot_error:
                            error_buffer.write(f"Error saving plot {i}: {str(plot_error)}\n")
                    
                    plt.close('all')  # Close all figures to free memory
                
                # Capture DataFrames from local variables
                if pd is not None:
                    for var_name, var_value in local_globals.items():
                        if isinstance(var_value, pd.DataFrame) and len(var_value) > 0:
                            try:
                                result["dataframes"].append({
                                    "name": var_name,
                                    "head": var_value.head(10).to_dict(),
                                    "shape": var_value.shape,
                                    "dtypes": var_value.dtypes.to_dict(),
                                    "columns": list(var_value.columns),
                                    "memory_usage": f"{var_value.memory_usage(deep=True).sum() / 1024:.2f} KB"
                                })
                            except Exception as df_error:
                                error_buffer.write(f"Error processing DataFrame {var_name}: {str(df_error)}\n")
                
                # Check for created files
                if os.path.exists(exec_dir):
                    for file in os.listdir(exec_dir):
                        file_path = os.path.join(exec_dir, file)
                        if os.path.isfile(file_path) and file_path not in result["files_created"]:
                            result["files_created"].append(file_path)
            
            result["status"] = "success"
            result["stdout"] = output_buffer.getvalue()
            result["stderr"] = error_buffer.getvalue()
            result["result"] = exec_result
            
        except SyntaxError as e:
            result["stderr"] = f"Syntax Error: {str(e)}\nLine {e.lineno}: {e.text}"
        except Exception as e:
            result["stderr"] = f"{error_buffer.getvalue()}\nExecution Error: {str(e)}\n{traceback.format_exc()}"
        
        return result

    def _execute_bash(self, code: str, execution_id: str) -> Dict[str, Any]:
        """Execute bash commands with timeout and safety measures."""
        result = {
            "execution_id": execution_id,
            "language": "bash",
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": [],
            "files_created": []
        }
        
        try:
            exec_dir = os.path.join(self.working_directory, execution_id)
            os.makedirs(exec_dir, exist_ok=True)
            
            # Basic security: prevent dangerous commands
            dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs', 'fdisk', 'format', ':(){ :|:& };:']
            if any(pattern in code for pattern in dangerous_patterns):
                result["stderr"] = "Command contains potentially dangerous operations and was blocked."
                return result
            
            completed = subprocess.run(
                code, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=self.max_execution_time,
                cwd=exec_dir
            )
            
            result["status"] = "success" if completed.returncode == 0 else "error"
            result["stdout"] = completed.stdout
            result["stderr"] = completed.stderr
            result["result"] = completed.returncode
            
            # Check for created files
            if os.path.exists(exec_dir):
                for file in os.listdir(exec_dir):
                    file_path = os.path.join(exec_dir, file)
                    if os.path.isfile(file_path):
                        result["files_created"].append(file_path)
            
        except subprocess.TimeoutExpired:
            result["stderr"] = f"Execution timed out after {self.max_execution_time} seconds."
        except Exception as e:
            result["stderr"] = f"Bash execution error: {str(e)}"
        
        return result

    def _execute_sql(self, code: str, execution_id: str) -> Dict[str, Any]:
        """Execute SQL queries using SQLite."""
        result = {
            "execution_id": execution_id,
            "language": "sql",
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": [],
            "files_created": []
        }
        
        conn = None
        try:
            conn = sqlite3.connect(self.temp_sqlite_db, timeout=self.max_execution_time)
            cursor = conn.cursor()
            
            # Split and execute multiple statements
            statements = [stmt.strip() for stmt in code.split(';') if stmt.strip()]
            
            for stmt in statements:
                cursor.execute(stmt)
                
                # If it's a SELECT query, capture results
                if stmt.upper().strip().startswith('SELECT'):
                    columns = [description[0] for description in cursor.description] if cursor.description else []
                    rows = cursor.fetchall()
                    
                    if pd is not None and columns and rows:
                        df = pd.DataFrame(rows, columns=columns)
                        result["dataframes"].append({
                            "name": "query_result",
                            "head": df.head(20).to_dict(),
                            "shape": df.shape,
                            "dtypes": {col: "object" for col in columns},  # SQLite doesn't have strong typing
                            "columns": columns
                        })
                    
                    result["stdout"] += f"Query returned {len(rows)} rows.\n"
                else:
                    conn.commit()
                    result["stdout"] += f"Statement executed successfully: {stmt[:50]}...\n"
            
            result["status"] = "success"
            
        except sqlite3.Error as e:
            result["stderr"] = f"SQL Error: {str(e)}"
        except Exception as e:
            result["stderr"] = f"SQL execution error: {str(e)}"
        finally:
            if conn:
                conn.close()
        
        return result

    def _execute_javascript(self, code: str, execution_id: str) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js if available."""
        result = {
            "execution_id": execution_id,
            "language": "javascript",
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": [],
            "files_created": []
        }
        
        try:
            exec_dir = os.path.join(self.working_directory, execution_id)
            os.makedirs(exec_dir, exist_ok=True)
            
            # Check if Node.js is available
            try:
                subprocess.run(["node", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                result["stderr"] = "Node.js is not installed or not available in PATH."
                return result
            
            # Write code to a temporary file
            js_file = os.path.join(exec_dir, "script.js")
            with open(js_file, "w") as f:
                f.write(code)
            
            # Execute with Node.js
            completed = subprocess.run(
                ["node", js_file],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=exec_dir
            )
            
            result["status"] = "success" if completed.returncode == 0 else "error"
            result["stdout"] = completed.stdout
            result["stderr"] = completed.stderr
            result["result"] = completed.returncode
            result["files_created"].append(js_file)
            
        except subprocess.TimeoutExpired:
            result["stderr"] = f"JavaScript execution timed out after {self.max_execution_time} seconds."
        except Exception as e:
            result["stderr"] = f"JavaScript execution error: {str(e)}"
        
        return result

    def _execute_r(self, code: str, execution_id: str) -> Dict[str, Any]:
        """Execute R code using Rscript if available."""
        result = {
            "execution_id": execution_id,
            "language": "r",
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": [],
            "files_created": []
        }
        
        try:
            exec_dir = os.path.join(self.working_directory, execution_id)
            os.makedirs(exec_dir, exist_ok=True)
            
            # Check if R is available
            try:
                subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                result["stderr"] = "R is not installed or Rscript is not available in PATH."
                return result
            
            # Write code to a temporary file
            r_file = os.path.join(exec_dir, "script.R")
            with open(r_file, "w") as f:
                f.write(code)
            
            # Execute with Rscript
            completed = subprocess.run(
                ["Rscript", r_file],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=exec_dir
            )
            
            result["status"] = "success" if completed.returncode == 0 else "error"
            result["stdout"] = completed.stdout
            result["stderr"] = completed.stderr
            result["result"] = completed.returncode
            result["files_created"].append(r_file)
            
            # Look for R plots (typically saved as Rplots.pdf or custom files)
            for file in os.listdir(exec_dir):
                if file.endswith(('.png', '.pdf', '.jpg', '.jpeg')) and file != os.path.basename(r_file):
                    file_path = os.path.join(exec_dir, file)
                    result["files_created"].append(file_path)
            
        except subprocess.TimeoutExpired:
            result["stderr"] = f"R execution timed out after {self.max_execution_time} seconds."
        except Exception as e:
            result["stderr"] = f"R execution error: {str(e)}"
        
        return result

# Global interpreter instance
_interpreter_instance = None

def get_interpreter():
    """Get or create a global interpreter instance."""
    global _interpreter_instance
    if _interpreter_instance is None:
        _interpreter_instance = SafeCodeInterpreter()
    return _interpreter_instance

@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the results including output, plots, and dataframes.
    Args:
        code (str): The Python code to execute.
    Returns:
        str: Formatted string with execution results.
    """
    interpreter = get_interpreter()
    result = interpreter.execute_code(code, "python")
    return _format_execution_result(result)

@tool
def execute_bash_command(command: str) -> str:
    """
    Execute bash commands and return the results.
    Args:
        command (str): The bash command(s) to execute.
    Returns:
        str: Formatted string with execution results.
    """
    interpreter = get_interpreter()
    result = interpreter.execute_code(command, "bash")
    return _format_execution_result(result)

@tool
def execute_sql_query(query: str) -> str:
    """
    Execute SQL queries using SQLite and return the results.
    Args:
        query (str): The SQL query to execute.
    Returns:
        str: Formatted string with execution results.
    """
    interpreter = get_interpreter()
    result = interpreter.execute_code(query, "sql")
    return _format_execution_result(result)

@tool
def execute_code_multilang(code: str, language: str = "python") -> str:
    """
    Execute code in multiple languages (Python, Bash, SQL, JavaScript, R) and return results.
    Args:
        code (str): The source code to execute.
        language (str): The language of the code. Supported: "python", "bash", "sql", "javascript", "r".
    Returns:
        str: Formatted string with execution results including stdout, stderr, plots, and dataframes.
    """
    supported_languages = ["python", "bash", "sql", "javascript", "r"]
    language = language.lower().strip()

    if language not in supported_languages:
        return f"❌ Unsupported language: {language}. Supported languages are: {', '.join(supported_languages)}"

    interpreter = get_interpreter()
    result = interpreter.execute_code(code, language)
    return _format_execution_result(result)

@tool
def create_sql_table(table_name: str, schema: str, data: Optional[str] = None) -> str:
    """
    Create a SQL table with the given schema and optionally insert data.
    Args:
        table_name (str): Name of the table to create.
        schema (str): SQL schema definition (e.g., "id INTEGER PRIMARY KEY, name TEXT").
        data (str, optional): INSERT statements or CSV-like data to populate the table.
    Returns:
        str: Result of table creation and data insertion.
    """
    interpreter = get_interpreter()
    
    # Create table
    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema});"
    result = interpreter.execute_code(create_sql, "sql")
    
    if result["status"] == "error":
        return f"❌ Error creating table: {result['stderr']}"
    
    response = [f"✅ Table '{table_name}' created successfully."]
    
    # Insert data if provided
    if data:
        if data.strip().upper().startswith("INSERT"):
            # Direct SQL INSERT statements
            data_result = interpreter.execute_code(data, "sql")
        else:
            # Assume CSV-like data, create INSERT statements
            lines = data.strip().split('\n')
            if lines:
                # Count columns in schema
                col_count = len([col.strip() for col in schema.split(',') if col.strip()])
                insert_statements = []
                
                for line in lines:
                    values = [val.strip() for val in line.split(',')]
                    if len(values) == col_count:
                        # Add quotes around text values (simple heuristic)
                        formatted_values = []
                        for val in values:
                            try:
                                # Try to convert to number
                                float(val)
                                formatted_values.append(val)
                            except ValueError:
                                formatted_values.append(f"'{val}'")
                        insert_statements.append(f"INSERT INTO {table_name} VALUES ({', '.join(formatted_values)});")
                
                if insert_statements:
                    data_sql = '\n'.join(insert_statements)
                    data_result = interpreter.execute_code(data_sql, "sql")
                else:
                    data_result = {"status": "error", "stderr": "No valid data rows found"}
        
        if data_result["status"] == "success":
            response.append("✅ Data inserted successfully.")
        else:
            response.append(f"❌ Error inserting data: {data_result['stderr']}")
    
    return '\n'.join(response)

@tool
def list_available_libraries() -> str:
    """
    List all available Python libraries in the code interpreter.
    Returns:
        str: List of available libraries and their versions.
    """
    interpreter = get_interpreter()
    
    # Check which libraries are available
    check_code = """
import sys
available_libs = []

# Check standard libraries
standard_libs = ['math', 'random', 'statistics', 'datetime', 'collections', 
                'itertools', 'functools', 'operator', 're', 'json', 'uuid', 'tempfile']

for lib in standard_libs:
    try:
        __import__(lib)
        available_libs.append(lib)
    except ImportError:
        pass

# Check data science libraries
data_libs = {'numpy': 'np', 'pandas': 'pd', 'matplotlib': 'plt', 'scipy': 'scipy', 
             'sklearn': 'sklearn', 'sympy': 'sympy', 'PIL': 'PIL'}

for lib, alias in data_libs.items():
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'unknown')
        available_libs.append(f"{lib} ({version})")
    except ImportError:
        pass

print("Available libraries:")
for lib in sorted(available_libs):
    print(f"  • {lib}")

print(f"\\nPython version: {sys.version}")
"""
    
    result = interpreter.execute_code(check_code, "python")
    
    if result["status"] == "success":
        return result["stdout"]
    else:
        return f"Error checking libraries: {result['stderr']}"

def _format_execution_result(result: Dict[str, Any]) -> str:
    """Format execution result into a readable string."""
    response = []
    
    if result["status"] == "success":
        response.append(f"✅ Code executed successfully in **{result['language'].upper()}** (ID: {result['execution_id']})")
        
        if result.get("stdout"):
            response.append(f"\n**Standard Output:**\n```\n{result['stdout'].strip()}\n```")
        
        if result.get("stderr"):
            response.append(f"\n**Standard Error/Warnings:**\n```\n{result['stderr'].strip()}\n```")
        
        if result.get("result") is not None and result["result"] != 0:
            response.append(f"\n**Return Code:** {result['result']}")
        
        if result.get("dataframes"):
            response.append(f"\n**DataFrames Created ({len(result['dataframes'])}):**")
            for df_info in result["dataframes"]:
                response.append(f"\n• **{df_info['name']}** (Shape: {df_info['shape']})")
                if 'memory_usage' in df_info:
                    response.append(f"  Memory: {df_info['memory_usage']}")
                
                if pd is not None:
                    try:
                        df_preview = pd.DataFrame(df_info["head"])
                        if not df_preview.empty:
                            response.append(f"  First few rows:\n```\n{str(df_preview)}\n```")
                    except Exception:
                        response.append("  (Preview unavailable)")
        
        if result.get("plots"):
            response.append(f"\n**Generated {len(result['plots'])} plot(s):**")
            for i, plot in enumerate(result["plots"]):
                response.append(f"  • Plot {i+1}: {plot.get('filename', 'plot.png')}")
        
        if result.get("files_created"):
            response.append(f"\n**Files created:** {len(result['files_created'])}")
            for file_path in result["files_created"][:5]:  # Show only first 5
                response.append(f"  • {os.path.basename(file_path)}")
            if len(result["files_created"]) > 5:
                response.append(f"  • ... and {len(result['files_created']) - 5} more")
    
    else:
        response.append(f"❌ Code execution failed in **{result['language'].upper()}** (ID: {result['execution_id']})")
        if result.get("stderr"):
            response.append(f"\n**Error Log:**\n```\n{result['stderr'].strip()}\n```")
    
    return "\n".join(response)

# Clean up on module exit
import atexit
def cleanup_interpreter():
    global _interpreter_instance
    if _interpreter_instance:
        _interpreter_instance.cleanup()

atexit.register(cleanup_interpreter)