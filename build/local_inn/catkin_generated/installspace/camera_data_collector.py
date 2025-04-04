
#!/usr/bin/env python3
import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

import random 

class RobotImageCollector:
    def __init__(self, npy_file, save_dir):
        rospy.init_node("robot_image_collector", anonymous=True)

        # Load (x, y) coordinates
        self.coordinates = np.load(npy_file)
        self.save_dir = save_dir
        self.bridge = CvBridge()

        # Prepare the image storage list
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  # Create the directory if it doesn't exist

        # Set up the Gazebo model state service
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.sleep(2)  # Give some time for connections

    def move_robot(self, x, y, yaw):
        """Move the robot model to a specified position (x, y, yaw)."""
        state_msg = ModelState()
        state_msg.model_name = "volta"  # Replace with your model name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.016074
        state_msg.pose.orientation.z = np.sin(yaw / 2.0)
        state_msg.pose.orientation.w = np.cos(yaw / 2.0)

        try:
            res =self.set_model_state(state_msg)
            if not res.success:
                rospy.logerr(f"Set Model State service retuned false")
            rospy.sleep(0.1)  # Small delay to ensure model movement
        except rospy.ServiceException as e:
            rospy.logerr(f"Set Model State service call failed: {e}")

    def capture_image(self, state_time):
        """Capture a single image from the camera topic."""
        try:            
            image_msg = rospy.wait_for_message('/camera/camera', Image, timeout=5.0)  
            while(image_msg.header.stamp < state_time):
                #rospy.logwarn("stale image captured!")
                rospy.sleep(0.1)
                image_msg = rospy.wait_for_message('/camera/camera', Image, timeout=5.0)  

            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            return cv_image
        except rospy.ROSException:
            rospy.logwarn("Timeout while waiting for camera image!")
            return None
        except Exception as e:
            rospy.logerr(f"Error capturing image: {e}")
            return None

    def collect_images(self):
        """Iterate through all coordinates and collect images at 10° rotations."""
        
        seed_value = 42
        random.seed(seed_value)
        for idx, (x, y) in enumerate(self.coordinates): 
            random_number = random.uniform(0, 4.99)
            if( idx <3300):
                continue
            if(idx %100 ==0):
                rospy.loginfo("waiting for 3 seconds")
                rospy.sleep(3)   
            
            rospy.loginfo(f"Processing coordinate {idx+1}/{len(self.coordinates)}: x={x}, y={y}. rand_num = {random_number}")          


            #Rotate the robot from 0° to 360° in 30° increments
            #found = False
            for id in range(72):
                yaw_deg = random_number + id*5 #10
                if(yaw_deg>=360.0):
                    yaw_deg = yaw_deg - 360
                filename = f"{x:.2f}_{y:.2f}_{yaw_deg:.2f}.jpg"  # Format to 1 decimal place
                file_path = os.path.join(self.save_dir, filename)
                
                # if found ==False:
                #     if os.path.exists(file_path):
                #         rospy.loginfo(f"The file exists: {filename}")
                #         continue
                #     else:
                #         found =True
                #         rospy.loginfo(f"continuing from file: {filename}") 
                # else:
                #     rospy.loginfo(f"done") 
                #     return None


            

                yaw_rad = np.deg2rad(yaw_deg)
                self.move_robot(x, y, yaw_rad)
                current_time = rospy.Time.now()

                # Capture the image after moving the robot
                #rospy.loginfo(f"Capturing image at yaw={yaw_deg} degrees")
                captured_image = self.capture_image(current_time)
                if captured_image is not None:
                    # Generate the filename based on x, y, and yaw values                   
                    

                    # Save the captured image as a jpg file
                    cv2.imwrite(file_path, captured_image)
                    #rospy.loginfo(f"Image saved as {file_path}")
                else:
                    rospy.logwarn("Failed to capture image!")

            # Rotate the robot from 0° to 360° in 30° increments
            # for yaw_deg in range(0, 360, 10):
            #     yaw_rad = np.deg2rad(yaw_deg)
            #     self.move_robot(x, y, yaw_rad)
            #     current_time = rospy.Time.now()

            #     # Capture the image after moving the robot
            #     #rospy.loginfo(f"Capturing image at yaw={yaw_deg} degrees")
            #     captured_image = self.capture_image(current_time)
            #     if captured_image is not None:
            #         # Generate the filename based on x, y, and yaw values
            #         filename = f"{x:.1f}_{y:.1f}_{yaw_deg}.jpg"  # Format to 1 decimal place
            #         file_path = os.path.join(self.save_dir, filename)

            #         # Save the captured image as a jpg file
            #         cv2.imwrite(file_path, captured_image)
            #         #rospy.loginfo(f"Image saved as {file_path}")
            #     else:
            #         rospy.logwarn("Failed to capture image!")

        rospy.loginfo("Image collection completed.")

class RobotImageCollector_test:
    def __init__(self, npy_file, save_dir):
        rospy.init_node("robot_image_collector", anonymous=True)

        # Load (x, y) coordinates
        self.coordinates = np.load(npy_file)
        self.save_dir = save_dir
        self.bridge = CvBridge()
        self.image_cout =0


        # Prepare the image storage list
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  # Create the directory if it doesn't exist

        # Set up the Gazebo model state service
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.sleep(2)  # Give some time for connections

    def move_robot(self, x, y, yaw):
        """Move the robot model to a specified position (x, y, yaw)."""
        state_msg = ModelState()
        state_msg.model_name = "volta"  # Replace with your model name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.016074
        state_msg.pose.orientation.z = np.sin(yaw / 2.0)
        state_msg.pose.orientation.w = np.cos(yaw / 2.0)

        try:
            res =self.set_model_state(state_msg)
            if not res.success:
                rospy.logerr(f"Set Model State service retuned false")
            rospy.sleep(0.1)  # Small delay to ensure model movement
        except rospy.ServiceException as e:
            rospy.logerr(f"Set Model State service call failed: {e}")

    def capture_image(self, state_time):
        """Capture a single image from the camera topic."""
        try:            
            image_msg = rospy.wait_for_message('/camera/camera', Image, timeout=5.0)  
            while(image_msg.header.stamp < state_time):
                #rospy.logwarn("stale image captured!")
                rospy.sleep(0.1)
                image_msg = rospy.wait_for_message('/camera/camera', Image, timeout=5.0)  

            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            return cv_image
        except rospy.ROSException:
            rospy.logwarn("Timeout while waiting for camera image!")
            return None
        except Exception as e:
            rospy.logerr(f"Error capturing image: {e}")
            return None

    def collect_images(self):
        """Iterate through all coordinates and collect images at 10° rotations."""
        

        for idx, (x, y, yaw) in enumerate(self.coordinates):   

            yaw_deg = np.rad2deg(yaw)
            rospy.loginfo(f"Processing coordinate {idx+1}/{len(self.coordinates)}: x={x}, y={y}. yaw_deg = {yaw_deg}")             
 
            self.move_robot(x, y, yaw)
            current_time = rospy.Time.now()

            # Capture the image after moving the robot
            #rospy.loginfo(f"Capturing image at yaw={yaw_deg} degrees")
            captured_image = self.capture_image(current_time)
            if captured_image is not None:
                # Generate the filename based on x, y, and yaw values
                filename = f"{self.image_cout}_{x:.2f}_{y:.2f}_{yaw_deg:.2f}.jpg"  # Format to 1 decimal place
                file_path = os.path.join(self.save_dir, filename)
                self.image_cout+=1

                # Save the captured image as a jpg file
                cv2.imwrite(file_path, captured_image)
                #rospy.loginfo(f"Image saved as {file_path}")
            else:
                rospy.logwarn("Failed to capture image!")

        rospy.loginfo("Test Image collection completed.")



if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        npy_file = os.path.join(script_dir, "co_ordinates_data/array_0.1_step_23rddec.npy")
        #npy_file = os.path.join(script_dir, "valid_position_home_0.2.npy")
        #npy_file = "valid_positions_home_world12thDec.npy"  # Replace with input .npy file path

        #save_dir = os.path.join(script_dir, "test_pose_images23rddec")
        save_dir = "./output_images_gazebo_home_6thjan"          # Replace with output directory path
        collector = RobotImageCollector(npy_file, save_dir)
        collector.collect_images()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Node interrupted.")






# if __name__ == "__main__":
#     try:
#         npy_file = "./valid_positions_home_world12thDec.npy"  # Replace with input .npy file path
#         save_file = "./camera_data12dec.npy"       # Replace with output .npy file path
#         #collector = RobotImageCollector(npy_file, save_file)
#         #collector.collect_images()

#         rospy.init_node("robot_image_collector", anonymous=True)
#         bridge = CvBridge()

#         image_msg = rospy.wait_for_message('/camera/camera', Image, timeout=5.0)
#         cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
#         rospy.loginfo(cv_image.shape)
#         rospy.loginfo(cv_image)

#     except rospy.ROSInterruptException:
#         rospy.loginfo("ROS Node interrupted.")
