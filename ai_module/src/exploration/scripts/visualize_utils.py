from visualization_msgs.msg import Marker, MarkerArray


def renew_marker_array(publisher, markers):
    delete_marker = MarkerArray()
    m = Marker()
    m.action = Marker.DELETEALL
    delete_marker.markers.append(m)
    publisher.publish(delete_marker)
    publisher.publish(markers)
