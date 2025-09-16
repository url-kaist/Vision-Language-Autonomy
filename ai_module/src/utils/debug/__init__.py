# -*- coding: utf-8 -*-
"""
Debug shims for common ROS messages/services.
If real ROS modules exist, we do nothing.
If not, we inject permissive stub modules into sys.modules:
- std_msgs.msg: String, Int32
- visualization_msgs.msg: Marker, MarkerArray
- nav_msgs.msg: Path, Odometry, OccupancyGrid
- geometry_msgs.msg: Header, Pose, PoseStamped, Point, Quaternion, Twist
- std_srvs.srv: Trigger, TriggerRequest, TriggerResponse
- visual_grounding.srv: SetSubplans, SetSubplansRequest, SetSubplansResponse (flexible stub)
"""

import sys
import types
import time


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ns(**kwargs):
    """Lightweight namespace object that accepts arbitrary attributes."""
    obj = types.SimpleNamespace()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj

def _mk_mod(modname):
    """Create a Python module object."""
    mod = types.ModuleType(modname)
    mod.__dict__["__package__"] = modname.rsplit(".", 1)[0] if "." in modname else ""
    return mod

def _install_stub(modname, module_obj):
    """Install stub module if real one is not importable."""
    if modname not in sys.modules:
        sys.modules[modname] = module_obj

def _try_import(modname):
    try:
        __import__(modname)
        return True
    except Exception:
        return False

# Small Time/Header compatibility
def _now_stamp():
    t = time.time()
    secs = int(t)
    nsecs = int((t - secs) * 1e9)
    return _ns(secs=secs, nsecs=nsecs)

def _Header(frame_id=""):
    return _ns(stamp=_now_stamp(), frame_id=frame_id)

# ------------------------------------------------------------
# If real ROS is available, exit early
# ------------------------------------------------------------
if all(_try_import(m) for m in [
    "std_msgs.msg",
    "visualization_msgs.msg",
    "nav_msgs.msg",
    "geometry_msgs.msg",
    "std_srvs.srv",
    "sensor_msgs.msg",
]):
    # visual_grounding.srv could be custom; we don't require it to exist
    # If user has it, they'll import the real one; otherwise our stub below will kick in when imported.
    pass
else:
    # --------------------------------------------------------
    # std_msgs.msg
    # --------------------------------------------------------
    std_msgs_msg = _mk_mod("std_msgs.msg")

    class String:
        __slots__ = ("data",)
        def __init__(self, data=""):
            self.data = data

    class Int32:
        __slots__ = ("data",)
        def __init__(self, data=0):
            self.data = int(data)


    class Empty:
        __slots__ = ()

        def __init__(self):
            pass

        def __repr__(self):
            return "Empty()"


    std_msgs_msg.Empty = Empty


    class ColorRGBA:
        __slots__ = ("r", "g", "b", "a")

        def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
            self.r, self.g, self.b, self.a = float(r), float(g), float(b), float(a)


    class Header:
        __slots__ = ("seq", "stamp", "frame_id")

        def __init__(self, seq=0, stamp=None, frame_id=""):
            self.seq = int(seq)
            self.stamp = stamp  # rospy.Time 등 아무거나
            self.frame_id = str(frame_id)

    std_msgs_msg.String = String
    std_msgs_msg.Int32  = Int32
    std_msgs_msg.ColorRGBA = ColorRGBA
    std_msgs_msg.Header = Header
    _install_stub("std_msgs.msg", std_msgs_msg)

    # --------------------------------------------------------
    # geometry_msgs.msg (subset needed by Path/Odometry/Markers)
    # --------------------------------------------------------
    geometry_msgs_msg = _mk_mod("geometry_msgs.msg")

    class Point:
        __slots__ = ("x", "y", "z")
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Quaternion:
        __slots__ = ("x", "y", "z", "w")
        def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
            self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)

    class Pose:
        __slots__ = ("position", "orientation")
        def __init__(self, position=None, orientation=None):
            self.position = position if position is not None else Point()
            self.orientation = orientation if orientation is not None else Quaternion()

    class PoseStamped:
        __slots__ = ("header", "pose")
        def __init__(self, header=None, pose=None):
            self.header = header if header is not None else _Header()
            self.pose   = pose if pose is not None else Pose()

    class Vector3:
        __slots__ = ("x", "y", "z")
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Twist:
        __slots__ = ("linear", "angular")
        def __init__(self, linear=None, angular=None):
            self.linear  = linear if linear is not None else Vector3()
            self.angular = angular if angular is not None else Vector3()

    class Header:
        def __init__(self):
            h = _Header()
            self.stamp = h.stamp
            self.frame_id = h.frame_id

    geometry_msgs_msg.Point       = Point
    geometry_msgs_msg.Quaternion  = Quaternion
    geometry_msgs_msg.Pose        = Pose
    geometry_msgs_msg.PoseStamped = PoseStamped
    geometry_msgs_msg.Vector3     = Vector3
    geometry_msgs_msg.Twist       = Twist
    geometry_msgs_msg.Header      = Header
    _install_stub("geometry_msgs.msg", geometry_msgs_msg)

    # --------------------------------------------------------
    # visualization_msgs.msg
    # --------------------------------------------------------
    visualization_msgs_msg = _mk_mod("visualization_msgs.msg")

    class Marker:
        # ---- type constants ----
        ARROW = 0
        CUBE = 1
        SPHERE = 2
        CYLINDER = 3
        LINE_STRIP = 4
        LINE_LIST = 5
        CUBE_LIST = 6
        SPHERE_LIST = 7
        POINTS = 8
        TEXT_VIEW_FACING = 9
        MESH_RESOURCE = 10
        TRIANGLE_LIST = 11

        # ---- action constants ----
        ADD = 0
        MODIFY = 0  # ROS는 ADD/MODIFY를 0으로 취급
        DELETE = 2
        DELETEALL = 3

        __slots__ = (
            "header", "ns", "id", "type", "action",
            "pose", "scale", "color", "lifetime", "frame_locked",
            "points", "colors", "text", "mesh_resource", "mesh_use_embedded_materials"
        )

        def __init__(self):
            self.header = Header()
            self.ns = ""
            self.id = 0
            self.type = Marker.ARROW
            self.action = Marker.ADD
            self.pose = Pose()
            self.scale = Vector3(1.0, 1.0, 1.0)
            self.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            self.lifetime = 0.0  # sec로 간단화
            self.frame_locked = False
            self.points = []  # List[Point]
            self.colors = []  # List[ColorRGBA]
            self.text = ""
            self.mesh_resource = ""
            self.mesh_use_embedded_materials = False

    class MarkerArray:
        __slots__ = ("markers",)
        def __init__(self, markers=[]):
            self.markers = markers

    visualization_msgs_msg.Marker = Marker
    visualization_msgs_msg.MarkerArray = MarkerArray
    _install_stub("visualization_msgs.msg", visualization_msgs_msg)

    # --------------------------------------------------------
    # nav_msgs.msg
    # --------------------------------------------------------
    nav_msgs_msg = _mk_mod("nav_msgs.msg")

    class Path:
        __slots__ = ("header", "poses")
        def __init__(self, header=None, poses=None):
            self.header = header if header is not None else _Header()
            self.poses  = poses if poses is not None else []  # list of PoseStamped

    class Odometry:
        __slots__ = ("header", "child_frame_id", "pose", "twist")
        def __init__(self):
            self.header = _Header()
            self.child_frame_id = ""
            # Keep it permissive: covariance not modeled (rarely used in debug)
            self.pose  = _ns(pose=geometry_msgs_msg.Pose())
            self.twist = _ns(twist=geometry_msgs_msg.Twist())

    class MapMetaData:
        __slots__ = ("map_load_time", "resolution", "width", "height", "origin")
        def __init__(self):
            self.map_load_time = _now_stamp()
            self.resolution = 0.05
            self.width = 0
            self.height = 0
            self.origin = geometry_msgs_msg.Pose()

    class OccupancyGrid:
        __slots__ = ("header", "info", "data")
        def __init__(self):
            self.header = _Header()
            self.info = MapMetaData()
            self.data = []  # list[int] in [-1, 100]

    nav_msgs_msg.Path          = Path
    nav_msgs_msg.Odometry      = Odometry
    nav_msgs_msg.OccupancyGrid = OccupancyGrid
    nav_msgs_msg.MapMetaData   = MapMetaData
    _install_stub("nav_msgs.msg", nav_msgs_msg)

    # --------------------------------------------------------
    # std_srvs.srv
    # --------------------------------------------------------
    std_srvs_srv = _mk_mod("std_srvs.srv")

    class TriggerRequest:
        __slots__ = ()
        def __init__(self): pass

    class TriggerResponse:
        __slots__ = ("success", "message")
        def __init__(self, success=False, message=""):
            self.success = bool(success)
            self.message = str(message)

    class Trigger:
        _request_class  = TriggerRequest
        _response_class = TriggerResponse

    std_srvs_srv.Trigger         = Trigger
    std_srvs_srv.TriggerRequest  = TriggerRequest
    std_srvs_srv.TriggerResponse = TriggerResponse


    class EmptyRequest:
        __slots__ = ()

        def __init__(self):  # no fields
            pass


    class EmptyResponse:
        __slots__ = ()

        def __init__(self):  # no fields
            pass


    class Empty:
        _request_class = EmptyRequest
        _response_class = EmptyResponse


    std_srvs_srv.Empty = Empty
    std_srvs_srv.EmptyRequest = EmptyRequest
    std_srvs_srv.EmptyResponse = EmptyResponse
    _install_stub("std_srvs.srv", std_srvs_srv)

    # --------------------------------------------------------
    # std_srvs.srv - SetBool
    # --------------------------------------------------------
    try:
        std_srvs_srv  # reuse if already created above
    except NameError:
        std_srvs_srv = _mk_mod("std_srvs.srv")


    class SetBoolRequest:
        __slots__ = ("data",)

        def __init__(self, data=False):
            self.data = bool(data)


    class SetBoolResponse:
        __slots__ = ("success", "message")

        def __init__(self, success=False, message=""):
            self.success = bool(success)
            self.message = str(message)


    class SetBool:
        _request_class = SetBoolRequest
        _response_class = SetBoolResponse


    std_srvs_srv.SetBool = SetBool
    std_srvs_srv.SetBoolRequest = SetBoolRequest
    std_srvs_srv.SetBoolResponse = SetBoolResponse

    # --------------------------------------------------------
    # visual_grounding.srv (custom; make it permissive)
    # --------------------------------------------------------
    vg_srv_pkg = "visual_grounding.srv"
    if not _try_import(vg_srv_pkg):
        visual_grounding_srv = _mk_mod(vg_srv_pkg)

        class SetSubplansRequest:
            # Make it permissive: accept arbitrary kwargs for your project fields
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class SetSubplansResponse:
            def __init__(self, success=False, message=""):
                self.success = bool(success)
                self.message = str(message)

        class SetSubplans:
            _request_class  = SetSubplansRequest
            _response_class = SetSubplansResponse

        visual_grounding_srv.SetSubplans         = SetSubplans
        visual_grounding_srv.SetSubplansRequest  = SetSubplansRequest
        visual_grounding_srv.SetSubplansResponse = SetSubplansResponse
        _install_stub(vg_srv_pkg, visual_grounding_srv)

    # --------------------------------------------------------
    # sensor_msgs.msg  (ADD THIS)
    # --------------------------------------------------------
    sensor_msgs_msg = _mk_mod("sensor_msgs.msg")


    class PointField:
        INT8 = 1;
        UINT8 = 2;
        INT16 = 3;
        UINT16 = 4
        INT32 = 5;
        UINT32 = 6;
        FLOAT32 = 7;
        FLOAT64 = 8

        def __init__(self, name="", offset=0, datatype=7, count=1):
            self.name = name
            self.offset = int(offset)
            self.datatype = int(datatype)
            self.count = int(count)


    class PointCloud2:
        """
        Minimal stub of sensor_msgs/PointCloud2
        """
        __slots__ = (
            "header", "height", "width", "fields", "is_bigendian",
            "point_step", "row_step", "data", "is_dense"
        )

        def __init__(self):
            self.header = _Header()
            self.height = 0
            self.width = 0
            self.fields = []  # list[PointField]
            self.is_bigendian = False
            self.point_step = 0
            self.row_step = 0
            self.data = b""  # bytes
            self.is_dense = True


    sensor_msgs_msg.PointField = PointField
    sensor_msgs_msg.PointCloud2 = PointCloud2
    _install_stub("sensor_msgs.msg", sensor_msgs_msg)


    class Image:
        __slots__ = ("header", "height", "width", "encoding",
                     "is_bigendian", "step", "data")

        def __init__(self):
            self.header = Header()
            self.height = 0
            self.width = 0
            self.encoding = ""  # ex) "bgr8", "mono8", "16UC1"
            self.is_bigendian = 0
            self.step = 0  # bytes per row
            self.data = b""  # raw bytes


    class CompressedImage:
        __slots__ = ("header", "format", "data")

        def __init__(self):
            self.header = Header()
            self.format = ""  # "jpeg" | "png"
            self.data = b""  # compressed bytes


    sensor_msgs_msg.Image = Image
    sensor_msgs_msg.CompressedImage = CompressedImage

    # --------------------------------------------------------
    # sensor_msgs.point_cloud2 (helpers stub)
    # --------------------------------------------------------
    pc2_mod = _mk_mod("sensor_msgs.point_cloud2")
    if "sensor_msgs" not in sys.modules:
        sensor_msgs_pkg = _mk_mod("sensor_msgs")
        _install_stub("sensor_msgs", sensor_msgs_pkg)
    else:
        sensor_msgs_pkg = sys.modules["sensor_msgs"]


    def read_points(cloud, field_names=("x", "y", "z"), skip_nans=False):
        yield from []


    def create_cloud(header, fields, points):
        pc = PointCloud2()
        pc.header = header
        return pc


    def create_cloud_xyz32(header, points):
        pc = PointCloud2()
        pc.header = header
        return pc


    pc2_mod.read_points = read_points
    pc2_mod.create_cloud = create_cloud
    pc2_mod.create_cloud_xyz32 = create_cloud_xyz32

    # sys.modules 등록 + 부모에 속성 연결
    _install_stub("sensor_msgs.point_cloud2", pc2_mod)
    setattr(sensor_msgs_pkg, "point_cloud2", pc2_mod)

    cv_bridge = _mk_mod("cv_bridge")

    # CvBridgeError
    class CvBridgeError(Exception):
        pass


    cv_bridge.CvBridgeError = CvBridgeError


    # 내부 유틸
    def _dtype_from_encoding(enc):
        import numpy as np
        enc = (enc or "").lower()
        if enc in ("bgr8", "rgb8", "mono8", "bgra8", "rgba8"):
            return np.uint8
        if enc in ("16uc1",):
            return np.uint16
        if enc in ("32fc1",):
            return np.float32
        # 기본
        return np.uint8


    def _channels_from_encoding(enc):
        enc = (enc or "").lower()
        if enc in ("mono8", "16uc1", "32fc1"):
            return 1
        if enc in ("bgr8", "rgb8"):
            return 3
        if enc in ("bgra8", "rgba8"):
            return 4
        # 모르면 3으로 가정
        return 3


    def _bytes_per_channel(enc):
        import numpy as np
        dt = _dtype_from_encoding(enc)
        return np.dtype(dt).itemsize


    def _guess_encoding(arr):
        import numpy as np
        if arr.ndim == 2:
            if arr.dtype == np.uint8:
                return "mono8"
            if arr.dtype == np.uint16:
                return "16UC1"
            if arr.dtype == np.float32:
                return "32FC1"
            return "mono8"
        if arr.ndim == 3:
            c = arr.shape[2]
            if arr.dtype == np.uint8:
                if c == 3: return "bgr8"
                if c == 4: return "bgra8"
        return "bgr8"


    def _convert_encoding(img, src, dst):
        import cv2
        src = (src or "").lower()
        dst = (dst or "").lower()

        if dst == "passthrough" or src == dst:
            return img

        # 깊이 타입은 변환 미지원(간단화)
        if src in ("16uc1", "32fc1") or dst in ("16uc1", "32fc1"):
            if src == dst:
                return img
            raise CvBridgeError(f"Cannot convert depth encoding {src} -> {dst}")

        # 컬러/그레이 변환
        if src == "bgr8" and dst == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if src == "rgb8" and dst == "bgr8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if src in ("bgr8", "rgb8") and dst == "mono8":
            code = cv2.COLOR_BGR2GRAY if src == "bgr8" else cv2.COLOR_RGB2GRAY
            return cv2.cvtColor(img, code)
        if src == "mono8" and dst == "bgr8":
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if src == "mono8" and dst == "rgb8":
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if src == "bgra8" and dst == "bgr8":
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if src == "rgba8" and dst == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if src == "bgra8" and dst == "rgb8":
            return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        if src == "rgba8" and dst == "bgr8":
            return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)

        raise CvBridgeError(f"Unsupported conversion {src} -> {dst}")


    # 본체
    class CvBridge:
        __slots__ = ()

        def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
            try:
                import numpy as np
                h, w = int(img_msg.height), int(img_msg.width)
                enc = img_msg.encoding or "bgr8"
                ch = _channels_from_encoding(enc)
                dt = _dtype_from_encoding(enc)

                buf = np.frombuffer(img_msg.data, dtype=dt)
                if ch == 1:
                    arr = buf.reshape((h, w))
                else:
                    arr = buf.reshape((h, w, ch))

                if desired_encoding and desired_encoding.lower() != "passthrough":
                    arr = _convert_encoding(arr, enc, desired_encoding)
                return arr
            except Exception as e:
                raise CvBridgeError(f"imgmsg_to_cv2 failed: {e}")

        def cv2_to_imgmsg(self, cv_img, encoding="bgr8"):
            try:
                import numpy as np
                msg = Image()
                msg.height, msg.width = int(cv_img.shape[0]), int(cv_img.shape[1])
                msg.encoding = encoding
                msg.is_bigendian = 0
                # step = width * (bytes_per_channel * channels)
                ch = (1 if cv_img.ndim == 2 else int(cv_img.shape[2]))
                msg.step = msg.width * (_bytes_per_channel(encoding) * ch)
                # 데이터 복사 (연속 메모리 보장)
                msg.data = memoryview(np.ascontiguousarray(cv_img)).tobytes()
                return msg
            except Exception as e:
                raise CvBridgeError(f"cv2_to_imgmsg failed: {e}")

        def compressed_imgmsg_to_cv2(self, cimg_msg, desired_encoding="bgr8"):
            try:
                import numpy as np, cv2
                np_arr = np.frombuffer(cimg_msg.data, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise CvBridgeError("imdecode returned None")
                if desired_encoding and desired_encoding.lower() != "passthrough":
                    img = _convert_encoding(img, _guess_encoding(img), desired_encoding)
                return img
            except Exception as e:
                raise CvBridgeError(f"compressed_imgmsg_to_cv2 failed: {e}")

        def cv2_to_compressed_imgmsg(self, cv_img, dst_format="jpeg"):
            try:
                import cv2
                fmt = (dst_format or "jpeg").lower()
                ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"
                ok, buf = cv2.imencode(ext, cv_img)
                if not ok:
                    raise CvBridgeError("imencode failed")
                msg = CompressedImage()
                msg.format = "jpeg" if ext == ".jpg" else "png"
                msg.data = buf.tobytes()
                return msg
            except Exception as e:
                raise CvBridgeError(f"cv2_to_compressed_imgmsg failed: {e}")


    cv_bridge.CvBridge = CvBridge

# ------------------------------------------------------------
# Done: after importing this module, your normal imports work:
# from nav_msgs.msg import Path, Odometry, OccupancyGrid
# from std_msgs.msg import String, Int32
# from visualization_msgs.msg import Marker, MarkerArray
# from visual_grounding.srv import SetSubplans, SetSubplansResponse
# from std_srvs.srv import Trigger, TriggerResponse
# ------------------------------------------------------------
