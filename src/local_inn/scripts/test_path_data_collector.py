#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import numpy as np
import time
import math

class PoseCollector:
    def __init__(self):
        rospy.init_node("pose_collector_node", anonymous=True)

        # Service client setup for /gazebo/get_model_state
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.robot_name = "volta"  
        self.data = []  # To store [x, y, yaw] over time

        # Timer setup: call self.collect_data every 0.5 seconds
        self.timer_duration = 0.5  # seconds
        self.start_time = time.time()  # To track the 10-second window
        self.collection_duration = 20  # seconds

        # Start the timer
        rospy.Timer(rospy.Duration(self.timer_duration), self.collect_data)

    def get_pose(self):
        """Calls the /gazebo/get_model_state service to get x, y, and yaw."""
        try:
            # Prepare the service request
            request = GetModelStateRequest()
            request.model_name = self.robot_name

            # Call the service
            response = self.get_model_state(request)

            # Extract x, y positions
            x = response.pose.position.x
            y = response.pose.position.y

            # Extract yaw angle from quaternion
            q = response.pose.orientation
            yaw = self.quaternion_to_euler_yaw(q.x, q.y, q.z, q.w)

            return [x, y, yaw]
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None

    def quaternion_to_euler_yaw(self, x, y, z, w):
        """Convert quaternion to yaw (Euler angle)."""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def collect_data(self, event):
        """Collect robot pose data and append it to the list."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time <= self.collection_duration:
            pose = self.get_pose()
            if pose:
                rospy.loginfo(f"Collected pose: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}")
                self.data.append(pose)
        else:
            # Save the collected data and shutdown the node
            self.save_data()
            rospy.signal_shutdown("Data collection complete.")

    def save_data(self):
        """Save the collected data to a NumPy file."""
        data_array = np.array(self.data)
        np.save("robot_pose.npy", data_array)
        rospy.loginfo("Data saved to 'robot_pose.npy'.")


if __name__ == "__main__":
    try:
        collector = PoseCollector()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
