import sys
from orchestrator import OrchestratorTool

def main():
    # Initialize the orchestrator
    orchestrator = OrchestratorTool()

    # Get the task from the command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python app.py '<task>'")
        sys.exit(1)

    task = sys.argv[1]

    # Run the orchestrator
    query = {"prompt": task}
    response = orchestrator.forward(query)

    # Print the results
    print("Final Response:")
    print(response["final_response"])
    print("\nFeedback:")
    print(response["feedback"])

if __name__ == "__main__":
    main()