#!/usr/bin/env python3
import rospy
import sys
import os
import re
from visualization_msgs.msg import Marker
import tf.transformations as tft

# ====== 설정 ======
OBJECT_FILE = os.path.expanduser("/ws/external/system/unity/src/vehicle_simulator/mesh/unity/object_list.txt")
TOPIC_NAME = "/object_marker_with_label"
FRAME_ID = "map"

# ✅ [1] 정확한 파서 함수 정의 (가장 위 또는 main 위)
def load_objects_by_label(file_path, target_label):
    matched_objects = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue

            label_match = re.search(r'"([^"]+)"\s*$', line)
            if not label_match:
                continue
            label = label_match.group(1)

            line_wo_label = line[:label_match.start()].strip()
            parts = line_wo_label.split()

            if len(parts) < 8:
                continue

            obj = {
                "id": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "scale_x": float(parts[4]),
                "scale_y": float(parts[5]),
                "scale_z": float(parts[6]),
                "yaw": float(parts[7]),
                "label": label
            }

            if label.lower() == target_label.lower():
                matched_objects.append(obj)
    return matched_objects

# ✅ [2] yaw → quaternion
def yaw_to_quaternion(yaw):
    q = tft.quaternion_from_euler(0, 0, yaw)
    return q

# ✅ [3] marker 생성 함수
from visualization_msgs.msg import Marker

def create_markers(obj):
    # 1. CUBE 마커
    cube_marker = Marker()
    cube_marker.header.frame_id = FRAME_ID
    cube_marker.header.stamp = rospy.Time.now()
    cube_marker.ns = obj['label']
    cube_marker.id = obj['id']
    cube_marker.type = Marker.CUBE
    cube_marker.action = Marker.ADD
    cube_marker.pose.position.x = obj['x']
    cube_marker.pose.position.y = obj['y']
    cube_marker.pose.position.z = obj['z']
    q = yaw_to_quaternion(obj['yaw'])
    cube_marker.pose.orientation.x = q[0]
    cube_marker.pose.orientation.y = q[1]
    cube_marker.pose.orientation.z = q[2]
    cube_marker.pose.orientation.w = q[3]
    cube_marker.scale.x = obj['scale_x']
    cube_marker.scale.y = obj['scale_y']
    cube_marker.scale.z = obj['scale_z']
    cube_marker.color.r = 1.0
    cube_marker.color.g = 0.0
    cube_marker.color.b = 0.0
    cube_marker.color.a = 0.6
    cube_marker.lifetime = rospy.Duration(0)

    # 2. TEXT 마커
    text_marker = Marker()
    text_marker.header.frame_id = FRAME_ID
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = obj['label'] + "_text"
    text_marker.id = obj['id'] + 1000  # id 충돌 방지
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = obj['x']
    text_marker.pose.position.y = obj['y']
    text_marker.pose.position.z = obj['z'] + obj['scale_z'] / 2 + 0.1  # 큐브 위에 띄움
    text_marker.scale.z = 0.2  # 글자 크기 (x,y는 무시됨)
    text_marker.color.r = 0.0
    text_marker.color.g = 0.0
    text_marker.color.b = 0.0
    text_marker.color.a = 1.0
    text_marker.text = f"id={obj['id']}"
    text_marker.lifetime = rospy.Duration(0)

    return [cube_marker, text_marker]

def create_marker(obj):
    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()
    marker.ns = obj['label']
    marker.id = obj['id']
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = obj['x']
    marker.pose.position.y = obj['y']
    marker.pose.position.z = obj['z']
    q = yaw_to_quaternion(obj['yaw'])
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]
    marker.scale.x = obj['scale_x']
    marker.scale.y = obj['scale_y']
    marker.scale.z = obj['scale_z']
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 0.6
    marker.lifetime = rospy.Duration(0)
    return marker

# ✅ [4] 전체 퍼블리시 함수
def publish_all(label):
    rospy.init_node('pub_all_marker_node')
    pub = rospy.Publisher(TOPIC_NAME, Marker, queue_size=10)
    rospy.sleep(1.0)

    objects = load_objects_by_label(OBJECT_FILE, label)

    if not objects:
        rospy.logwarn(f"[pub_all] No object with label '{label}' found.")
        return

    for obj in objects:
        markers = create_markers(obj)
        for marker in markers:
            pub.publish(marker)
        rospy.loginfo(f"[pub_all] Published: id={obj['id']}, label={obj['label']}")
        rospy.sleep(0.05)


# ✅ [5] 메인 진입점
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pub_all.py <label>")
        sys.exit(1)

    publish_all(sys.argv[1])

