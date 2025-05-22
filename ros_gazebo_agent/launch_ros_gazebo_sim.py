import os
import subprocess
from smolagents import CodeAgent, HfApiModel, load_tool, tool
import yaml
import logging
import gradio as gr
from huggingface_hub import InferenceClient
# from tools.final_answer import FinalAnswerTool
# from Gradio_UI import GradioUI

@tool
def create_custom_simulation(robot: str, sensor: str, world: str) -> str:
    """Creates a custom Gazebo simulation environment.
    Args:
        robot: The type of robot to include (e.g., 'turtlebot', 'husky').
        sensor: The type of sensor to attach (e.g., 'camera', 'lidar').
        world: The Gazebo world file to use (e.g., 'empty_world', 'city_world').
    Returns:
        The command to launch the custom simulation.
    """
    command = f"ros2 launch gazebo_ros gazebo.launch.py world:={world} robot:={robot} sensor:={sensor}"
    return f"Executing: {command}"

@tool
def launch_simulation(simulation_name: str) -> str:
    """Launches a Gazebo simulation based on the provided name.
    Args:
        simulation_name: The name of the simulation to launch (e.g., 'shapes', 'air_pressure', 'camera').
    Returns:
        The command executed or an error message if the simulation is not recognized.
    """
    simulations = {
        "shapes": "ros2 launch ros_gz_sim gz_sim.launch.py gz_args:='shapes.sdf'",
        "air_pressure": "ros2 launch ros_gz_sim_demos air_pressure.launch.py",
        "camera": "ros2 launch ros_gz_sim_demos image_bridge.launch.py",
        "diff_drive": "ros2 launch ros_gz_sim_demos diff_drive.launch.py",
        "depth_camera": "ros2 launch ros_gz_sim_demos depth_camera.launch.py",
        "gpu_lidar": "ros2 launch ros_gz_sim_demos gpu_lidar.launch.py",
        "imu": "ros2 launch ros_gz_sim_demos imu.launch.py",
        "magnetometer": "ros2 launch ros_gz_sim_demos magnetometer.launch.py",
        "gnss": "ros2 launch ros_gz_sim_demos navsat.launch.py",
        "rgbd_camera": "ros2 launch ros_gz_sim_demos rgbd_camera.launch.py",
        "battery": "ros2 launch ros_gz_sim_demos battery.launch.py",
        "robot_description": "ros2 launch ros_gz_sim_demos robot_description_publisher.launch.py",
        "joint_states": "ros2 launch ros_gz_sim_demos joint_states.launch.py",
        "tf_bridge": "ros2 launch ros_gz_sim_demos tf_bridge.launch.py",
    }

    command = simulations.get(simulation_name.lower())
    if command:
        try:
            subprocess.run(command, shell=True, check=True)
            return f"Executing: {command}"
        except subprocess.CalledProcessError as e:
            return f"Failed to execute: {command}\nError: {e}"
    return "Simulation not recognized."

# final_answer = FinalAnswerTool()
client = InferenceClient(model="mistralai/Mistral-7B-v0.1")  # Replace with your desired model

# model = HfApiModel(
#     max_tokens=2096,
#     temperature=0.5,
#     model_id='Qwen/Qwen2.5-Coder-32B-Instruct'
# )

def hf_inference(input_text: str) -> str:
    """Run inference using Hugging Face's community-hosted model."""
    try:
        response = client.text_generation(input_text)
        return response
    except Exception as e:
        return f"Error during inference: {e}"
    
#     api_key=os.getenv("HF_TOKEN")


# with open("prompts.yaml", 'r') as stream:
#     prompt_templates = yaml.safe_load(stream)

# Load prompt templates from a YAML file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# agent = CodeAgent(
#     model=model,
#     # tools=[final_answer, launch_simulation],
#     tools=[launch_simulation, create_custom_simulation],
#     max_steps=6,
#     verbosity_level=1,
#     prompt_templates=prompt_templates
# )

# Configure logging
logging.basicConfig(filename="agent_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# def run_agent(input_text: str) -> str:
#     try:
#         logging.info(f"User Input: {input_text}")
#         result = agent.run(input_text)
#         logging.info(f"Agent Output: {result}")
#         return result
#     except Exception as e:
#         logging.error(f"Error: {e}")
#         return str(e)

def run_agent(input_text: str) -> str:
    try:
        logging.info(f"User Input: {input_text}")
        result = hf_inference(input_text)  # Use Hugging Face inference
        logging.info(f"Agent Output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        return str(e)

def gradio_interface(input_text):
    return run_agent(input_text)

if __name__ == "__main__":
    while True:
        user_input = input("Describe the simulation environment you want to launch (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        result = run_agent(user_input)
        print(result)
    # gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="Gazebo Simulation Agent").launch()
    # user_input = "Launch the shapes simulation."
    # result = run_agent(user_input)
    # print(result)


# GradioUI(agent).launch()
