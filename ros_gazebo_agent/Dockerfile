# Use the base image
FROM osrf/ros:humble-desktop-full

# Install pip for Python 3
RUN apt-get update && apt-get install -y python3-pip

# Install required Python packages
RUN pip3 install smolagents tools gradio

# Set the working directory
WORKDIR /workspace/ros_gazebo_agent

# Copy the project files into the container
# COPY . /workspace/ros_gazebo_agent


# Set the environment variable for Hugging Face API token
ENV HF_TOKEN=${HF_TOKEN}

# Command to run the script
CMD ["python3", "launch_ros_gazebo_sim.py"]