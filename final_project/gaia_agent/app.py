"""GAIA Agent Evaluation Runner"""
import os
import gradio as gr
import requests
import pandas as pd
import time
from langchain_core.messages import HumanMessage
from gaia_orchestrator import GaiaAgent

# Constants
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class GaiaAgentRunner:
    """GAIA benchmark agent runner"""
    def __init__(self):
        print("GaiaAgentRunner initialized.")
        self.agent = GaiaAgent(provider="groq")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        try:
            answer = self.agent(question)
            # Remove "FINAL ANSWER: " prefix if present for submission
            if answer.startswith("FINAL ANSWER: "):
                return answer[14:]
            return answer
        except Exception as e:
            print(f"Error processing question: {e}")
            return f"Error: {str(e)}"

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = GaiaAgentRunner()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code URL: {agent_code}")

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None

    # 3. Run Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    
    for i, item in enumerate(questions_data):
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        print(f"Processing question {i+1}/{len(questions_data)}: {task_id}")
        
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted Answer": submitted_answer
            })
            print(f"Answer for {task_id}: {submitted_answer}")
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            error_answer = f"AGENT ERROR: {e}"
            answers_payload.append({"task_id": task_id, "submitted_answer": error_answer})
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted Answer": error_answer
            })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Submit Results
    submission_data = {
        "username": username.strip(), 
        "agent_code": agent_code, 
        "answers": answers_payload
    }
    
    print(f"Submitting {len(answers_payload)} answers for user '{username}'...")

    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
        
    except Exception as e:
        error_msg = f"Submission Failed: {str(e)}"
        print(error_msg)
        results_df = pd.DataFrame(results_log)
        return error_msg, results_df

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1. This agent uses your existing tools for comprehensive GAIA benchmark evaluation
        2. Log in to your Hugging Face account using the button below
        3. Click 'Run Evaluation & Submit All Answers' to process all questions
        4. The agent will use web search, mathematical tools, document processing, code execution, and image analysis as needed
        
        **Tools Available:**
        - Web Research: DuckDuckGo, Wikipedia, ArXiv, webpage fetching
        - Mathematics: Full suite of mathematical operations and statistics
        - Documents: PDF, DOCX, CSV, Excel processing with OCR
        - Code: Python, Bash, SQL, JavaScript, R execution
        - Images: Analysis, transformation, generation, combination
        """
    )

    gr.LoginButton()
    
    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    
    status_output = gr.Textbox(label="Status & Results", lines=10, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " GAIA Agent Starting " + "-"*30)
    
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"   Runtime URL: https://{space_host}.hf.space")
    else:
        print("ℹ️  Running locally (no SPACE_HOST found)")

    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
    else:
        print("ℹ️  No SPACE_ID found")

    print("-"*66 + "\n")
    print("Launching GAIA Agent Evaluation Interface...")
    demo.launch(debug=True, share=False)