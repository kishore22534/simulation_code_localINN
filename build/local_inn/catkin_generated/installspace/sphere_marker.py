#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import os
import numpy as np

# Initialize ROS node
rospy.init_node('spawn_colored_dots')

# Wait for the spawn service to be available
rospy.wait_for_service('/gazebo/spawn_sdf_model')
spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

# Load SDF content for the sphere
with open("./blue_sphere.sdf", "r") as f:
    sphere_sdf = f.read()

script_dir = os.path.dirname(os.path.abspath(__file__))
npy_file = os.path.join(script_dir, "co_ordinates_data/array_0.1_step_23rddec.npy")

# Coordinates to spawn spheres
coordinates = np.load(npy_file)  #[[1, 2, 0], [2, 3, 0], [3, 1, 0]]  # Replace with your NumPy array

for i, (x, y) in enumerate(coordinates):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.1

    # Give each sphere a unique name
    sphere_name = f"red_sphere_{i}"

    # Spawn the sphere
    spawn_model(sphere_name, sphere_sdf, "", pose, "world")
