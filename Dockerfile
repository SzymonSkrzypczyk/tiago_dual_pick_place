# Base image with ROS Noetic
FROM ros:noetic-robot

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set workspace path inside container
ENV CATKIN_WS=/root/tiago_dual_pick_place

# Install OS-level dependencies for ROS, MoveIt, and Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    ros-noetic-moveit \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep (required for dependency installation)
RUN rosdep init && rosdep update

# Create the workspace and src folder
RUN mkdir -p $CATKIN_WS/src

# Copy the ROS package(s) into src
COPY ./src $CATKIN_WS/src

# Set working directory
WORKDIR $CATKIN_WS

# Install package dependencies using rosdep
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && rosdep install --from-paths src --ignore-src -y"

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source ROS and workspace automatically for interactive shells
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source $CATKIN_WS/devel/setup.bash" >> ~/.bashrc

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
