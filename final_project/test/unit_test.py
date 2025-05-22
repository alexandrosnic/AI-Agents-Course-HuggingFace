import unittest
from final_project.data_analysis_tool import DataAnalysisTool
from final_project.orchestrator import OrchestratorTool

class TestTools(unittest.TestCase):
    def test_data_analysis_tool(self):
        tool = DataAnalysisTool()
        query = {"operation": "mean", "data": {"numbers": [1, 2, 3, 4, 5]}}
        result = tool.forward(query)
        self.assertEqual(result, 3.0)

    def test_orchestrator(self):
        orchestrator = OrchestratorTool()
        query = {"prompt": "Find Batman filming locations and calculate travel times"}
        response = orchestrator.forward(query)
        self.assertIn("final_response", response)

if __name__ == "__main__":
    unittest.main()