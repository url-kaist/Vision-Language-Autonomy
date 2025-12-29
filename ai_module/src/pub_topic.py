#!/usr/bin/env python

import rospy
import sys
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String, Int16, Bool



def universal_publisher(topic_name, message_text):
    # Initialize node
    rospy.init_node('dynamic_cli_publisher', anonymous=True)

    # Create publisher dynamically based on input
    pub = rospy.Publisher(topic_name, String, queue_size=1, latch=False)

    # Allow time for ROS master to register the connection
    rospy.sleep(0.5)

    if not rospy.is_shutdown():
        rospy.loginfo(f"Publishing '{message_text}' to '{topic_name}'")
        pub.publish(message_text)
        # Brief sleep to ensure the message is sent before the script exits
        rospy.sleep(0.1)

def publish_markerarray(topic_name, points_list=None):
    rospy.init_node('waypoint_visualizer')

    # Note: latch=True is helpful for RViz so new subscribers 
    # see the markers even if they were published before joining.
    pub = rospy.Publisher(topic_name, MarkerArray, queue_size=1, latch=True)
    rospy.sleep(1.0)

    # Define some dummy coordinates
    if points_list is None:
        # points_list = [[0.42, 0.42, 0.0]] # , [0.16, 1.58, 0.0]]
        points_list = [[0.16, 1.58, 0.0]]


    marker_array = MarkerArray()

    for i, pt in enumerate(points_list):
        marker = Marker()
        marker.header.frame_id = "world"  # Ensure this matches your RViz fixed frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = pt[0]
        marker.pose.position.y = pt[1]
        marker.pose.position.z = pt[2]
        marker.pose.orientation.w = 1.0
        
        # Scale (Size of the sphere)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        # Color (RGBA)
        marker.color.a = 1.0 # Don't forget to set alpha to 1.0!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker_array.markers.append(marker)

    # Publish the array
    rospy.loginfo("Publishing waypoints...")
    pub.publish(marker_array)
    # Keep the script alive for a second to ensure delivery
    rospy.sleep(1.0)

def publish_boolean(topic_name, message_text, frequency=10):
    """
    Publishes the arrival status.
    :param has_arrived: Boolean (True or False)
    """
    if not rospy.is_shutdown() and not rospy.core.is_initialized():
        rospy.init_node('boolean_continuous_pub')

    # 2. Setup Publisher
    pub = rospy.Publisher(topic_name, Bool, queue_size=1)
    
    # 3. Set the rate (Frequency in Hz)
    rate = rospy.Rate(frequency) 

    msg = Bool()
    msg.data = message_text

    rospy.loginfo(f"Starting continuous broadcast on {topic_name} at {frequency}Hz")

    # 4. Continuous Loop
    while not rospy.is_shutdown():
        pub.publish(msg)
        # Use rate.sleep() to maintain the frequency
        rate.sleep()

def send_force_signal(topic_name):
    rospy.init_node('force_trigger_node', anonymous=True)
    
    # Setup Publisher
    pub = rospy.Publisher(topic_name, Empty, queue_size=1)
    
    # Wait for the connection to establish
    rospy.sleep(0.5)
    
    # Publish the Empty message
    # Note: We must instantiate the class: Empty()
    pub.publish(Empty())
    
    rospy.loginfo("Force answer signal sent.")

if __name__ == '__main__':
    try:
        topic_name = sys.argv[1]
        if topic_name == 'no_frontier':
            universal_publisher(topic_name='/instruction_following_exp_status', message_text='no_frontier')
        elif topic_name == 'active_waypoints':
            publish_markerarray(topic_name='/active_waypoints')
            print("active_waypoints")
        elif topic_name == 'vg_first':
            universal_publisher(topic_name='/exploration_strategy', message_text='vg_first')
        elif topic_name == 'geometric_frontier':
            universal_publisher(topic_name='/exploration_strategy', message_text='geometric_frontier')
        elif topic_name == 'path_follower_arrived':
            publish_boolean(topic_name='/path_follower/arrived', message_text=True)
            print("path_follower_arrived")
        elif topic_name == 'force':
            send_force_signal(topic_name='/force_answer')
        elif topic_name == 'marker_array':
            publish_markerarray(topic_name='/active_waypoints', points_list=sys.argv[2])


    except rospy.ROSInterruptException:
        pass
