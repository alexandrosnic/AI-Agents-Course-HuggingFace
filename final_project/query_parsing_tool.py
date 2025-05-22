from smolagents import Tool, CodeAgent
from smolagents.cli import load_model
from typing import Dict, Any
from smolagents import tool, LiteLLMModel

class QueryParsingTool(Tool):
    """
    A class-based implementation of an Input and Query Parsing Agent.
    Parses and comprehends the user's query, identifying key objectives, relevant keywords, and constraints.
    Converts the raw text into a structured data format for subsequent layers.
    """

    name = "query_parsing_tool"
    description = "Parses user queries into structured data, identifying objectives, keywords, and constraints."
    inputs = {
        "prompt": {
            "type": "string",
            "description": "A natural language query to parse (e.g., 'Find Batman filming locations and calculate travel times').",
            "required": True
        }
    }
    output_type = "object"

    def __init__(self, agent: CodeAgent):
        # Initialize the language model
        self.agent = agent

    def parse_query(self, prompt: str) -> Dict[str, Any]:
        """
        Parses the user's query into structured data.
        Args:
            prompt: The natural language query to parse.
        Returns:
            A dictionary containing the parsed objectives, keywords, and constraints.
        """
        # Define the system prompt for parsing
        system_prompt = f"""
        You are a query parsing agent. Your task is to parse the user's query into structured data.
        Identify the following:
        - Objectives: What the user wants to achieve.
        - Keywords: Important terms or entities in the query.
        - Constraints: Any specific conditions or limitations mentioned in the query.

        Example:
        Input: "Find Batman filming locations and calculate travel times to Gotham."
        Output: {{
            "objectives": ["Find Batman filming locations", "Calculate travel times to Gotham"],
            "keywords": ["Batman", "filming locations", "Gotham"],
            "constraints": []
        }}

        Now parse the following query:
        {prompt}
        """

        # Use the model to parse the query
        # response = self.model.run(system_prompt)
        # return response
        # response = self.model.complete(prompt=system_prompt).strip()
        response = self.agent.run(system_prompt)  
        return response  

    def forward(self, prompt: str) -> Dict:
        """
        Processes the query and parses it into structured data.
        Args:
            query: A dictionary containing the "prompt".
        Returns:
            The parsed query as structured data.
        """
        # prompt = query.get("prompt")
        # if not prompt:
        #     raise ValueError("The 'prompt' field is required.")

        # Parse the query
        parsed_query = self.parse_query(prompt)

        return {"parsed_query": parsed_query}