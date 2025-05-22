from smolagents import Tool
from smolagents.cli import load_model
from typing import Dict, List, Any
from smolagents import tool, LiteLLMModel

class IntegrationSynthesisTool(Tool):
    """
    A tool for integrating outputs from all specialized agents and rechecking for coherence and accuracy.
    """

    name = "integration_synthesis_tool"
    description = (
        "Integrates outputs from specialized agents, ensuring coherence and accuracy in the final response."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "A natural language query to parse a list of results from specialized agents, each containing a subtask description and its result.",
            "required": True
        }
    }
    output_type = "object"

    def __init__(self, model: LiteLLMModel):
        # Initialize the language model
        self.model = model

    def integrate_and_synthesize(self, results: List[Dict[str, Any]]) -> str:
        """
        Integrates and synthesizes the results into a coherent and accurate final response.
        Args:
            results: A list of results from specialized agents.
        Returns:
            A final integrated response as a string.
        """
        prompt = f"""
        You are an integration and synthesis agent. Your task is to combine the following results into a coherent and accurate final response:
        Results: {results}

        Ensure the response is concise, coherent, and free of contradictions. If there are any inconsistencies, resolve them logically.
        """
        # response = self.model.run(prompt)
        # return response.strip()
        response = self.model.complete(prompt=prompt).strip()  
        return response  

    def forward(self, prompt: str) -> Dict:
        """
        Processes the results from specialized agents and synthesizes a final response.
        Args:
            prompt: A text containing the "results" to integrate and synthesize.
        Returns:
            A dictionary containing the final integrated response.
        """
        # results = query.get("results")
        if not prompt:
            raise ValueError("The 'prompt' field is required.")

        # Integrate and synthesize the results
        final_response = self.integrate_and_synthesize(prompt)

        return {"final_response": final_response}