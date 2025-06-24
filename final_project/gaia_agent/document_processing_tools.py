import os
import tempfile
import uuid
from typing import Optional
from urllib.parse import urlparse
import requests
from smolagents import tool

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import zipfile
    import tarfile
except ImportError:
    zipfile = None
    tarfile = None

@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix='.txt')
        filepath = temp_file.name
        temp_file.close()
    else:
        filepath = os.path.join(temp_dir, filename)

    try:
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content)
        return f"File saved to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def read_text_file(file_path: str) -> str:
    """
    Read the contents of a text file.
    Args:
        file_path (str): the path to the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File content:\n\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    if Image is None or pytesseract is None:
        return "Error: PIL and pytesseract libraries are required for OCR functionality."
    
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Extract text from the image
        text = pytesseract.image_to_string(image)
        
        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

@tool
def extract_text_from_pdf(file_path: str, page_range: Optional[str] = None) -> str:
    """
    Extract text from a PDF file.
    Args:
        file_path (str): the path to the PDF file.
        page_range (str, optional): Page range to extract (e.g., "1-3" or "all"). Default is "all".
    """
    if fitz is None:
        return "Error: PyMuPDF library is required for PDF text extraction."
    
    try:
        doc = fitz.open(file_path)
        text = ""
        
        # Determine pages to extract
        if page_range is None or page_range.lower() == "all":
            pages = range(len(doc))
        else:
            # Parse page range (e.g., "1-3")
            if "-" in page_range:
                start, end = map(int, page_range.split("-"))
                pages = range(start - 1, min(end, len(doc)))
            else:
                page_num = int(page_range) - 1
                pages = [page_num] if 0 <= page_num < len(doc) else []
        
        for page_num in pages:
            page = doc.load_page(page_num)
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        
        doc.close()
        return f"Extracted text from PDF ({len(pages)} pages):\n\n{text}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

@tool
def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Microsoft Word document (.docx).
    Args:
        file_path (str): the path to the DOCX file.
    """
    if Document is None:
        return "Error: python-docx library is required for DOCX text extraction."
    
    try:
        doc = Document(file_path)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return f"Extracted text from DOCX:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

@tool
def analyze_csv_file(file_path: str, query: str = "") -> str:
    """
    Analyze a CSV file using pandas and provide insights.
    Args:
        file_path (str): the path to the CSV file.
        query (str, optional): Specific question about the data.
    """
    if pd is None:
        return "Error: pandas library is required for CSV analysis."
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic information
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Data types
        result += "Data types:\n"
        result += str(df.dtypes) + "\n\n"
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            result += "Missing values:\n"
            result += str(missing[missing > 0]) + "\n\n"
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result += "Summary statistics (numeric columns):\n"
            result += str(df[numeric_cols].describe()) + "\n\n"
        
        # First few rows
        result += "First 5 rows:\n"
        result += str(df.head()) + "\n"
        
        if query:
            result += f"\nRegarding your query '{query}': Please specify what analysis you need."
        
        return result
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel_file(file_path: str, sheet_name: Optional[str] = None, query: str = "") -> str:
    """
    Analyze an Excel file using pandas and provide insights.
    Args:
        file_path (str): the path to the Excel file.
        sheet_name (str, optional): Name of the sheet to analyze. If not provided, uses the first sheet.
        query (str, optional): Specific question about the data.
    """
    if pd is None:
        return "Error: pandas library is required for Excel analysis."
    
    try:
        # Read the Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        # Basic information
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        if sheet_name:
            result += f"Sheet: {sheet_name}\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Data types
        result += "Data types:\n"
        result += str(df.dtypes) + "\n\n"
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            result += "Missing values:\n"
            result += str(missing[missing > 0]) + "\n\n"
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result += "Summary statistics (numeric columns):\n"
            result += str(df[numeric_cols].describe()) + "\n\n"
        
        # First few rows
        result += "First 5 rows:\n"
        result += str(df.head()) + "\n"
        
        if query:
            result += f"\nRegarding your query '{query}': Please specify what analysis you need."
        
        return result
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

@tool
def list_excel_sheets(file_path: str) -> str:
    """
    List all sheet names in an Excel file.
    Args:
        file_path (str): the path to the Excel file.
    """
    if pd is None:
        return "Error: pandas library is required for Excel operations."
    
    try:
        xl_file = pd.ExcelFile(file_path)
        sheets = xl_file.sheet_names
        return f"Excel file contains {len(sheets)} sheets: {', '.join(sheets)}"
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"

@tool
def extract_archive(file_path: str, extract_to: Optional[str] = None) -> str:
    """
    Extract files from a ZIP or TAR archive.
    Args:
        file_path (str): the path to the archive file.
        extract_to (str, optional): Directory to extract to. If not provided, extracts to temp directory.
    """
    if extract_to is None:
        extract_to = tempfile.mkdtemp()
    
    try:
        if file_path.lower().endswith('.zip'):
            if zipfile is None:
                return "Error: zipfile library is required for ZIP extraction."
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                files = zip_ref.namelist()
        elif file_path.lower().endswith(('.tar', '.tar.gz', '.tgz')):
            if tarfile is None:
                return "Error: tarfile library is required for TAR extraction."
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
                files = tar_ref.getnames()
        else:
            return "Error: Unsupported archive format. Supported formats: ZIP, TAR, TAR.GZ"
        
        return f"Archive extracted to {extract_to}. Extracted files: {', '.join(files[:10])}{'...' if len(files) > 10 else ''}"
    except Exception as e:
        return f"Error extracting archive: {str(e)}"

@tool
def get_file_info(file_path: str) -> str:
    """
    Get information about a file (size, type, modification date, etc.).
    Args:
        file_path (str): the path to the file.
    """
    try:
        if not os.path.exists(file_path):
            return f"File does not exist: {file_path}"
        
        stat = os.stat(file_path)
        size = stat.st_size
        
        # Convert size to human readable format
        for unit in ['bytes', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                size_str = f"{size:.1f} {unit}"
                break
            size = size / 1024.0
        else:
            size_str = f"{size:.1f} TB"
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        
        # Modification time
        import datetime
        mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
        
        result = f"File: {os.path.basename(file_path)}\n"
        result += f"Path: {file_path}\n"
        result += f"Size: {size_str}\n"
        result += f"Type: {ext if ext else 'No extension'}\n"
        result += f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"Readable: {os.access(file_path, os.R_OK)}\n"
        result += f"Writable: {os.access(file_path, os.W_OK)}"
        
        return result
    except Exception as e:
        return f"Error getting file info: {str(e)}"

@tool
def count_words_in_text(text: str) -> str:
    """
    Count words, characters, and lines in a text.
    Args:
        text (str): the text to analyze.
    """
    try:
        lines = text.split('\n')
        words = text.split()
        chars = len(text)
        chars_no_spaces = len(text.replace(' ', ''))
        
        result = f"Text statistics:\n"
        result += f"Lines: {len(lines)}\n"
        result += f"Words: {len(words)}\n"
        result += f"Characters: {chars}\n"
        result += f"Characters (no spaces): {chars_no_spaces}\n"
        
        # Most common words (simple analysis)
        word_freq = {}
        for word in words:
            clean_word = word.lower().strip('.,!?";()[]{}')
            if len(clean_word) > 2:  # Skip very short words
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        if word_freq:
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            result += f"\nMost common words: {', '.join([f'{word}({count})' for word, count in top_words])}"
        
        return result
    except Exception as e:
        return f"Error analyzing text: {str(e)}"