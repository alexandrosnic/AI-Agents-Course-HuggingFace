from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from smolagents import Tool, DuckDuckGoSearchTool, tool

@tool
def duckduckgo_search(query: str) -> str:
    """
    Search DuckDuckGo for a query and return up to 3 results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted search results from DuckDuckGo.
    """
    from smolagents import DuckDuckGoSearchTool
    tool = DuckDuckGoSearchTool(max_results=3)
    results = tool.run(query)
    formatted = "\n\n---\n\n".join(
        [
            f'<Document source="{res.get("link", "")}"/>\n{res.get("snippet", "")}\n</Document>'
            for res in results
        ]
    ) if isinstance(results, list) else str(results)
    return {"duckduckgo_results": formatted}

@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia for a query and return up to 2 results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted Wikipedia search results.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}

@tool
def fetch_webpage_content(url: str) -> str:
    """
    Fetch and return the main content from a specific web page URL.
    Args:
        url (str): The URL of the web page to fetch.
    Returns:
        str: Formatted content from the web page.
    """
    docs = WebBaseLoader(url).load()
    formatted = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in docs
        ]
    )
    return {"web_results": formatted}

@tool
def arxiv_search(query: str) -> str:
    """
    Search Arxiv for a query and return up to 3 results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted Arxiv search results.
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}

@tool
def youtube_search(query: str) -> str:
    """
    Search YouTube for videos related to a query.
    Args:
        query (str): The search query.
    Returns:
        str: YouTube search results.
    """
    from langchain_community.tools import YouTubeSearchTool
    tool = YouTubeSearchTool()
    return tool.run(query)

@tool
def screenshot(url: str) -> str:
    """
    Take a screenshot of a web page (requires selenium and a browser).
    Args:
        url (str): The URL of the web page to screenshot.
    Returns:
        str: Path to the saved screenshot image.
    """
    from selenium import webdriver
    driver = webdriver.Chrome()
    driver.get(url)
    screenshot_path = "/tmp/screenshot.png"
    driver.save_screenshot(screenshot_path)
    driver.quit()
    return f"Screenshot saved to {screenshot_path}"