from smolagents import Tool, DuckDuckGoSearchTool, VisitWebpageTool
from smolagents.models import Model

from typing import Dict


class WebAgentTool(Tool):
    """
    A tool for browsing the web and retrieving information.
    """

    name = "web_agent_tool"
    description = (
        "Performs web searches and retrieves information from webpages. "
        "Use this tool to search the web or visit specific URLs."
    )
    inputs = {
        "task_type": {
            "type": "string",
            "description": "The type of task to perform ('search' or 'visit').",
            "required": True,
        },
        "query": {
            "type": "string",
            "description": "The search query or URL to visit.",
            "required": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None):
        # Initialize the web search and webpage visit tools
        super().__init__()
        self.model = model
        self.web_search_tool = DuckDuckGoSearchTool()
        self.visit_webpage_tool = VisitWebpageTool()

    def forward(self, task_type: str, query: str) -> str:
        """
        Executes the task described in the inputs using the appropriate web tool.
        Args:
            task_type: The type of task to perform ('search' or 'visit').
            query: The search query or URL to visit.
        Returns:
            The result of the task.
        """
        if not task_type or not query:
            raise ValueError("Both 'task_type' and 'query' are required.")

        if task_type == "search":
            # Perform a web search
            return self.web_search_tool.forward({"query": query})
        elif task_type == "visit":
            # Visit a webpage
            return self.visit_webpage_tool.forward({"url": query})
        else:
            raise ValueError("Invalid 'task_type'. Must be 'search' or 'visit'.")