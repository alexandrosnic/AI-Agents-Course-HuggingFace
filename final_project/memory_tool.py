from smolagents import Tool

class MemoryTool(Tool):
    name = "memory_tool"
    description = "Stores and retrieves contextual information for continuity."
    inputs = {"key": "The key to store or retrieve data.", "value": "The value to store (optional)."}
    output_type = "string"

    def __init__(self):
        self.memory = {}

    def forward(self, query: Dict) -> str:
        key = query.get("key")
        value = query.get("value")
        if value:
            self.memory[key] = value
            return f"Stored value for key '{key}'."
        return self.memory.get(key, "No value found for the given key.")