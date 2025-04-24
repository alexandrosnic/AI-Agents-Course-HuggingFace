# Import necessary libraries
import random
from smolagents import CodeAgent, HfApiModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset

# Initialize the Hugging Face model
model = HfApiModel()

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)

# # Create Alfred with conversation memory
# alfred_with_memory = CodeAgent(
#     tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
#     model=model,
#     add_base_tools=True,
#     planning_interval=3
# )

# # First interaction
# response1 = alfred_with_memory.run("Tell me about Lady Ada Lovelace.")
# print("🎩 Alfred's First Response:")
# print(response1)

# # Second interaction (referencing the first)
# response2 = alfred_with_memory.run("What projects is she currently working on?", reset=False)
# print("🎩 Alfred's Second Response:")
# print(response2)