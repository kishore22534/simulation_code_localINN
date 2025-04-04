#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def publish_marker():
    rospy.init_node('marker_publisher')
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(1)  # Wait for the publisher to initialize

    marker = Marker()
    marker.header.frame_id = "World"  # Set frame to world
    marker.header.stamp = rospy.Time.now()
    marker.ns = "basic_shapes"
    marker.id = 0
    marker.type = Marker.SPHERE  # Marker type: sphere
    marker.action = Marker.ADD  # Add marker
    marker.pose.position = Point(1, 1, 1)  # Position the marker at (1, 1, 0)
    marker.scale.x = 0.2  # Size of the sphere
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0  # Full opacity
    marker.color.r = 1.0  # Red color
    marker.color.g = 0.0  # Green
    marker.color.b = 0.0  # Blue

    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rospy.sleep(1)  # Publish once per second

if __name__ == '__main__':
    try:
        publish_marker()
    except rospy.ROSInterruptException:
        pass
