#!/usr/bin/env python

import rospy
from sem.srv import (
    ObjectNodeFromClassRequest,
    ObjectNodeFromClassResponse,
    ObjectNodeFromClass,
)
import sem.msg
from slam_classes import MapObjectList
import copy


class ObjectNodeService:
    def __init__(self, objects: MapObjectList):
        """
        ObjectNodeService는 ROS 서비스 서버로, 객체 요청을 받고 해당 객체 정보를 반환합니다.
        :param objects: 저장된 객체들 (MapObjectList)
        """
        self.objects = objects
        self.service = rospy.Service(
            "/get_object_node_info", ObjectNodeFromClass, self.handle_request
        )
        rospy.loginfo("ObjectNodeService initialized.")

    def handle_request(
        self, req: ObjectNodeFromClassRequest
    ) -> ObjectNodeFromClassResponse:
        """
        서비스 요청을 처리하는 함수. 주어진 객체 이름으로 객체를 찾아 응답합니다.
        :param req: 요청 객체 (예: 객체 이름)
        :return: 응답 객체 (ObjectNodeResponse)
        """
        requested_object_name = req.object_name
        found_objects = []

        # 객체 검색 (이 부분은 나중에 요청 형태에 맞게 유연하게 확장 가능)
        for obj in copy.deepcopy(self.objects):
            if obj["class_name"].lower() == requested_object_name.lower():
                object_info = sem.msg.ObjectNode()
                object_info.id = obj["id"]
                # object_info.points = (
                # obj["points"].flatten().tolist()
                # )  # points는 1D 리스트로 변환
                object_info.points = []
                object_info.center = obj["center"].tolist()
                object_info.min_pt = obj["min_bbox"].tolist()
                object_info.max_pt = obj["max_bbox"].tolist()
                object_info.class_id = obj["class_id"]
                object_info.class_name = obj["class_name"]
                object_info.conf = obj["conf"]
                found_objects.append(object_info)

        # 응답 메시지 설정
        response = ObjectNodeFromClassResponse()
        response.num_objects = len(found_objects)  # len(found_objects)
        response.objects = found_objects

        return response
