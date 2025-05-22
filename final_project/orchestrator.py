import os
import json
from smolagents import Tool, CodeAgent, LiteLLMModel
from smolagents.models import MessageRole
from web_browser_tool import WebAgentTool
from text_reader_tool import TextSummarizationTool
# from data_analysis_tool import DataAnalysisTool
# from integration_synthesis_tool import IntegrationSynthesisTool
# from adaptive_learning_tool import FeedbackLoopTool
from typing import Dict, Any, List

# Initialize the LiteLLMModel with Anthropic Claude
ANTHROPIC_API = os.getenv("ANTHROPIC_API")  # Set your Anthropic API key as an environment variable
model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key=ANTHROPIC_API)

class OrchestratorTool(Tool):
    name = "orchestrator_tool"
    description = (
        "Breaks down complex queries into subtasks, assigns them to specialized tools, "
        "and integrates their outputs into a coherent response."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "A natural language query describing the overall task to perform.",
            "required": True,
        }
    }
    output_type = "string"

    def forward(self, prompt: str) -> str:
        """
        Orchestrates the task by decomposing it, executing subtasks, and integrating results.
        """
        subtasks = self.decompose_task(prompt)
        results = self.execute_subtasks(subtasks)

        # Combine results into a single response
        final_response = "\n".join(
            [f"Subtask: {r['subtask']}\nResult: {r.get('result', r.get('error', 'No result'))}" for r in results]
        )
        return final_response


# class OrchestratorTool(Tool):
#     """
#     Orchestrates complex tasks by breaking them into subtasks, assigning them to specialized tools,
#     and integrating their outputs into a coherent response.
#     """
#     name = "orchestrator_tool"
#     description = (
#         "Breaks down complex queries into subtasks, assigns them to specialized tools, "
#         "and integrates their outputs into a coherent response."
#     )
#     inputs = {
#         "prompt": {
#             "type": "string",
#             "description": "A natural language query describing the overall task to perform.",
#             "required": True,
#         }
#     }
#     output_type = "string"

#     def __init__(self):
#         self.model = model

#         # Initialize tools with the model
#         self.web_agent_tool = WebAgentTool(model)
#         self.text_summarization_tool = TextSummarizationTool(model)

#         # Initialize the CodeAgent with all tools
#         self.agent = CodeAgent(
#             tools=[
#                 self.web_agent_tool,
#                 self.text_summarization_tool,
#             ],
#             model=model,
#         )

#     def parse_query(self, task: str) -> str:
#         """
#         Parses the user's query into structured data using the model.
#         Args:
#             task: The natural language query to parse.
#         Returns:
#             The raw response from the model as a string.
#         """
#         # Define the system prompt for parsing
#         system_prompt = f"""
#         You are a query parsing agent. Your task is to parse the user's query into structured data.
#         Identify the following:
#         - Objectives: What the user wants to achieve.
#         - Keywords: Important terms or entities in the query.
#         - Constraints: Any specific conditions or limitations mentioned in the query.

#         Example:
#         Input: "Find Batman filming locations and calculate travel times to Gotham."
#         Output:
#         Objectives: Find Batman filming locations, Calculate travel times to Gotham
#         Keywords: Batman, filming locations, Gotham
#         Constraints: None

#         Now parse the following query:
#         {task}
#         """

#         # Construct the messages list
#         messages = [
#             {
#                 "role": "system",
#                 "content": "You will have to write a short caption for this file, then answer this question:"
#                     + task,  # Ensure content is a string
#             }
#         ]

#         # Debugging output
#         print(f"Messages sent to model: {messages}")

#         # Use the model to parse the query
#         response = self.model(messages)

#         # Ensure the response has the expected structure
#         if not hasattr(response, "content"):
#             raise ValueError(f"Unexpected response format: {response}")

#         if not response.content:
#             raise ValueError("The model returned an empty response. Check the input and model configuration.")

#         # Debugging output
#         print(f"Model response: {response.content}")

#         return response.content  # Return the raw string response

#     def decompose_task(self, task: str) -> List[str]:
#         """
#         Decomposes the main task into subtasks using the query parsing logic.
#         """
#         parsed_query = self.parse_query(task)

#         # Example: Split the response into subtasks based on objectives
#         subtasks = []
#         for line in parsed_query.split("\n"):
#             if line.startswith("Objectives:"):
#                 objectives = line.replace("Objectives:", "").strip()
#                 subtasks = objectives.split(", ")
#                 break

#         return subtasks

#     def execute_subtasks(self, subtasks: List[str]) -> List[Dict[str, Any]]:
#         """
#         Executes subtasks using the appropriate tools.
#         """
#         results = []
#         for subtask in subtasks:
#             # Determine the tool to use based on the subtask
#             if "search" in subtask.lower():
#                 tool = self.web_agent_tool
#             else:
#                 tool = self.text_summarization_tool

#             try:
#                 result = tool.forward({"prompt": subtask})
#                 results.append({"subtask": subtask, "result": result})
#             except Exception as e:
#                 results.append({"subtask": subtask, "error": str(e)})
#         return results

#     def forward(self, prompt: str) -> str:
#         """
#         Orchestrates the task by decomposing it, executing subtasks, and integrating results.
#         """
#         subtasks = self.decompose_task(prompt)
#         results = self.execute_subtasks(subtasks)

#         # Combine results into a single response
#         final_response = "\n".join(
#             [f"Subtask: {r['subtask']}\nResult: {r.get('result', r.get('error', 'No result'))}" for r in results]
#         )
#         return final_response