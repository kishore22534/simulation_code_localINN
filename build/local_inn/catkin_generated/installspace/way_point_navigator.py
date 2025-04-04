#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
import math

from geometry_msgs.msg import Pose2D

class WaypointNavigator:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('waypoint_navigator_camera', anonymous=True)
        
        # Publisher to send velocity commands
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber to get the robot's current position from /gazebo/model_states
        #self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        self.pose_sub = rospy.Subscriber('/estimated_pose', Pose2D, self.pose_callback)
        
        # Current position and orientation of the robot
        self.x = 2.0
        self.y = -3.5
        self.yaw = 3.14
        
        # Robot model name in Gazebo
        self.robot_name = "volta"  # Change this to the name of your robot in Gazebo
        
        # Parameters for the controller
        self.linear_speed = 0.5
        self.angular_speed = 1.0
        self.distance_tolerance = 0.15
        
        # List of waypoints (x, y)
        #self.waypoints = [ (0.6, -1.5), (-0.6, 0), (1.0, 1.3), (2.0, 2.7),(0, 3.6), (-3, 2.4), (-2, 0.7)]
        #self.waypoints_lidar = [ (3, -1), (5,2), (3,5), (-0.5, 5.5), (-3.5, 2.5), (-2,-2), (0,0)]
        self.waypoints = [ (-0.38, -4.03), (-1.74, -1.15), (-1.25, 0.60), (2.06, 1.08), (4.38, -1.88), (6.08, -3.45), (7.45, -2.15), (4.98, -0.49), (1.98, -3.77)]
        self.current_waypoint_idx = 0

    
    def pose_callback(self, msg):
        # Extract x, y, and yaw values from the message
        self.x = msg.x
        self.y = msg.y
        self.yaw = msg.theta

    def navigate(self):
        rate = rospy.Rate(10)  # 10 Hz
        vel_msg = Twist()

        while not rospy.is_shutdown():
            if self.current_waypoint_idx < len(self.waypoints):
                goal_x, goal_y = self.waypoints[self.current_waypoint_idx]

                # Calculate the distance and angle to the goal
                distance = math.sqrt((goal_x - self.x)**2 + (goal_y - self.y)**2)
                angle_to_goal = math.atan2(goal_y - self.y, goal_x - self.x)
                angle_error = angle_to_goal - self.yaw

                # Normalize angle_error to [-pi, pi]
                angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

                if distance > self.distance_tolerance:
                    # Proportional controller for linear and angular velocity
                    vel_msg.linear.x = self.linear_speed
                    vel_msg.angular.z = self.angular_speed * angle_error
                else:
                    # Reached the current waypoint, move to the next one
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.0
                    rospy.loginfo(f"Reached waypoint {self.current_waypoint_idx + 1}")
                    self.current_waypoint_idx += 1
            else:
                # All waypoints are completed
                vel_msg.linear.x = 0.0
                vel_msg.angular.z = 0.0
                self.vel_pub.publish(vel_msg)
                rospy.loginfo("All waypoints are reached. Stopping.")
                break

            # Publish the velocity command
            self.vel_pub.publish(vel_msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = WaypointNavigator()
        navigator.navigate()
    except rospy.ROSInterruptException:
        pass
