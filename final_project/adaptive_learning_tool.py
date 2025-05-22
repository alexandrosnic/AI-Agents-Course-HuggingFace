from smolagents import Tool
from smolagents.cli import load_model
from typing import Dict, List, Any
from smolagents import tool, LiteLLMModel


class FeedbackLoopTool(Tool):
    """
    A tool for assessing the performance of modules, adjusting strategies, and refining parameters when inconsistencies arise.
    """

    name = "feedback_loop_tool"
    description = (
        "Monitors the performance of modules, identifies inconsistencies, and adjusts strategies or parameters to improve results. Provides actionable recommendations for system improvement."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "A string containing the results and logs in a structured format. For example: 'results: [...], logs: ...'",
            "required": True,
        }
    }
    output_type = "object"

    def __init__(self, model: LiteLLMModel):
        # Initialize the language model
        self.model = model

    def analyze_results(self, results: List[Dict[str, Any]], logs: str) -> List[Dict[str, Any]]:
        """
        Analyzes the results and logs to identify inconsistencies or inefficiencies.
        Args:
            results: A list of results from specialized agents.
            logs: Execution logs from the orchestrator and sub-agents.
        Returns:
            A list of recommendations for improving the system.
        """
        prompt = f"""
        You are a feedback loop agent. Analyze the following results and logs to identify inconsistencies or inefficiencies. Simplify and optimize for token usage:
        Results: {results}
        Logs: {logs}

        Provide a list of recommendations for improving the system. Each recommendation should include:
        - The affected module or tool.
        - The issue identified.
        - The recommended adjustment or strategy.
        """
        # response = self.model.run(prompt)
        # return eval(response.strip())  # Convert the response string to a Python list
        response = self.model.complete(prompt=prompt).strip()  
        return eval(response)  
    
    def apply_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        Applies the recommendations to adjust strategies or refine parameters.
        Args:
            recommendations: A list of recommendations for improving the system.
        Returns:
            A summary of the adjustments made.
        """
        prompt = f"""
        You are a feedback loop agent. Apply the following recommendations to adjust strategies or refine parameters:
        Recommendations: {recommendations}

        Provide a summary of the adjustments made.
        """
        # response = self.model.run(prompt)
        # return response.strip()
        response = self.model.complete(prompt=prompt).strip()  
        return response  

    def forward(self, prompt: str) -> Dict:
        """
        Monitors the performance of modules, identifies inconsistencies, and adjusts strategies or parameters.
        Args:
            prompt: A string containing the results and logs in a structured format.
                    For example: "results: [...], logs: ..."
        Returns:
            A dictionary containing the updated results and a summary of adjustments made.
        """
        if not prompt:
            raise ValueError("The 'prompt' field is required.")

        # Parse the prompt to extract results and logs
        try:
            # Example parsing logic: "results: [...], logs: ..."
            parts = prompt.split(", logs: ")
            results = eval(parts[0].split("results: ")[1].strip())  # Convert the string representation of the list to an actual list
            logs = parts[1].strip()
        except (IndexError, ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid prompt format: {prompt}. Expected format: 'results: [...], logs: ...'") from e

        # Step 1: Analyze the results and logs
        recommendations = self.analyze_results(results, logs)

        # Step 2: Apply the recommendations
        adjustments_summary = self.apply_recommendations(recommendations)

        return {"recommendations": recommendations, "adjustments_summary": adjustments_summary}