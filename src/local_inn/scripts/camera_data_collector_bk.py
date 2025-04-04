#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class CameraImageCollector:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('camera_image_collector', anonymous=True)
        
        # Define the image topic to subscribe to
        self.image_topic = "/camera/camera"  # Updated topic name
        
        # Set up CvBridge to convert ROS Image messages to OpenCV format
        self.bridge = CvBridge()
        
        # Directory to save collected images
        self.save_dir = "./collected_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Image counter for naming files
        self.img_counter = 0
        
        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        
        rospy.loginfo("Image collector initialized. Listening to %s", self.image_topic)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Save the image to file
            image_filename = os.path.join(self.save_dir, f"image_{self.img_counter:04d}.jpg")
            cv2.imwrite(image_filename, cv_image)
            
            rospy.loginfo("Saved image %s", image_filename)
            self.img_counter += 1

            rospy.sleep(2)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

if __name__ == '__main__':
    try:
        # Start the image collector
        collector = CameraImageCollector()
        rospy.spin()  # Keep the node alive
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera image collector node terminated.")
