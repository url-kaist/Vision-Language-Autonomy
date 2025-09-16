import sys
import threading as _threading
sys.path.append("/ws/external/ai_module/src/utils/debug")
import ai_module.src.utils.debug
from std_srvs.srv import Empty, EmptyResponse, Trigger, TriggerResponse


# in-memory param server (thread-safe)
_PARAM_LOCK = _threading.RLock()
_PARAMS = {}


def _resolve_param_name(name: str) -> str:
    # 최소 구현: 실제 ROS처럼 이름공간 해석은 생략하고 그대로 사용
    # 필요하면 여기서 '~' 처리나 prefix 붙이기 구현
    return str(name)


def set_param(name, value):
    with _PARAM_LOCK:
        _PARAMS[_resolve_param_name(name)] = value


def get_param(name, default=None):
    if name == "~debug":
        return True
    with _PARAM_LOCK:
        return _PARAMS.get(_resolve_param_name(name), default)


def has_param(name) -> bool:
    with _PARAM_LOCK:
        return _resolve_param_name(name) in _PARAMS


def delete_param(name):
    with _PARAM_LOCK:
        _PARAMS.pop(_resolve_param_name(name), None)


def get_name():
    return "PythonVersion"


def loginfo(msg, *args, **kwargs): print(msg)


def logdebug(msg, *args, **kwargs): print(msg)


def logwarn(msg, *args, **kwargs): print(msg)


def logerr(msg, *args, **kwargs): print(msg)


class Duration:
    __slots__ = ("secs", "nsecs")

    def __init__(self, secs: float = 0.0):
        total_ns = int(round(secs * 1e9))
        self.secs = total_ns // 1_000_000_000
        self.nsecs = total_ns % 1_000_000_000

    @staticmethod
    def from_sec(secs: float) -> "Duration":
        return Duration(secs)

    def to_sec(self) -> float:
        return self.secs + self.nsecs / 1e9

    # arithmetic
    def __add__(self, other):
        if isinstance(other, Duration):
            return Duration(self.to_sec() + other.to_sec())
        raise TypeError(f"unsupported operand type(s) for +: 'Duration' and '{type(other).__name__}'")

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Duration):
            return Duration(self.to_sec() - other.to_sec())
        raise TypeError(f"unsupported operand type(s) for -: 'Duration' and '{type(other).__name__}'")

    # comparisons
    def __lt__(self, other):
        return self.to_sec() < other.to_sec()

    def __le__(self, other):
        return self.to_sec() <= other.to_sec()

    def __gt__(self, other):
        return self.to_sec() > other.to_sec()

    def __ge__(self, other):
        return self.to_sec() >= other.to_sec()

    def __repr__(self):
        return f"Duration({self.to_sec():.9f}s)"


class _TimeObj:
    __slots__ = ("secs", "nsecs")

    def __init__(self, secs: int, nsecs: int):
        self.secs = int(secs)
        self.nsecs = int(nsecs)

    def to_sec(self) -> float:
        return self.secs + self.nsecs / 1e9

    # Time - Time -> Duration
    def __sub__(self, other):
        if isinstance(other, _TimeObj):
            return Duration(self.to_sec() - other.to_sec())
        raise TypeError(f"unsupported operand type(s) for -: '_TimeObj' and '{type(other).__name__}'")

    # Time + Duration -> Time
    def __add__(self, other):
        if isinstance(other, Duration):
            total = self.to_sec() + other.to_sec()
            return Time.from_sec(total)
        raise TypeError(f"unsupported operand type(s) for +: '_TimeObj' and '{type(other).__name__}'")

    # Duration + Time -> Time
    __radd__ = __add__

    def __repr__(self):
        return f"Time(secs={self.secs}, nsecs={self.nsecs})"


def _normalize_time(secs: int, nsecs: int):
    # nsecs를 0 <= nsecs < 1e9 범위로 정규화
    BILLION = 1_000_000_000
    # 음수/과다 ns 정규화
    carry, nsecs = divmod(nsecs, BILLION)
    secs += carry
    # divmod는 음수도 정규화되지만, secs<0 & nsecs>0 같은 케이스 추가 처리 원하면 여기에 보강
    return int(secs), int(nsecs)


class Time:
    def __new__(cls, *args, **kwargs):
        # 지원 시그니처:
        #   Time() -> (0, 0)
        #   Time(float|int total_secs)
        #   Time(secs, nsecs)
        #   Time(secs=..., nsecs=...)

        if not args and not kwargs:
            secs, nsecs = 0, 0

        elif len(args) == 1 and not kwargs:
            v = args[0]
            if isinstance(v, (int,)):
                secs, nsecs = int(v), 0
            elif isinstance(v, float):
                s = int(v)
                ns = int((v - s) * 1e9)
                secs, nsecs = s, ns
            else:
                # 다른 Time 객체/튜플을 넣어주는 경우까지 허용하고 싶다면 여기서 처리
                raise TypeError(f"Unsupported single-arg type for Time: {type(v)}")

        elif len(args) == 2 and not kwargs:
            secs, nsecs = int(args[0]), int(args[1])

        else:
            if 'secs' in kwargs or 'nsecs' in kwargs:
                secs = int(kwargs.get('secs', 0))
                nsecs = int(kwargs.get('nsecs', 0))
            else:
                raise TypeError("Time() accepts (), (total_sec), (secs, nsecs), or secs=/nsecs= kwargs")

        secs, nsecs = _normalize_time(secs, nsecs)
        return _TimeObj(secs, nsecs)

    @staticmethod
    def now() -> _TimeObj:
        import time as _time
        t = _time.time()
        secs = int(t)
        nsecs = int((t - secs) * 1e9)
        return _TimeObj(secs, nsecs)

    @staticmethod
    def from_sec(f: float) -> _TimeObj:
        secs = int(f)
        nsecs = int((f - secs) * 1e9)
        return _TimeObj(secs, nsecs)


import time as _time
import threading as _threading

# 전역 shutdown 플래그
_shutdown_lock = _threading.Lock()
_last_call_time = None  # 마지막 호출 시각


def is_shutdown() -> bool:
    global _last_call_time
    with _shutdown_lock:
        now = _time.time()

        if _last_call_time is None:
            # 타이머 시작
            _last_call_time = now
            return False

        if now - _last_call_time >= 1.0:
            # 1초 경과 → True 반환 & 타이머 리셋
            _last_call_time = None
            return True
        else:
            return False


def signal_shutdown(reason: str = ""):
    global _shutdown_flag
    with _shutdown_lock:
        _shutdown_flag = True
    print(f"(shim) rospy shutdown requested: {reason}")


class Rate:
    def __init__(self, hz: float):
        if hz <= 0:
            raise ValueError("hz must be > 0")
        self.sleep_dur = 1.0 / float(hz)
        self.last_time = _time.time()

    def sleep(self):
        now = _time.time()
        elapsed = now - self.last_time
        remaining = self.sleep_dur - elapsed
        if remaining > 0:
            _time.sleep(remaining)
        self.last_time = _time.time()


class ROSException(Exception): ...


class ServiceException(Exception): ...


# --- Service stub ---
class Service:
    def __init__(self, name, srv_class, handler):
        self.resolved_name = name
        self.srv_class = srv_class
        self.handler = handler
        # 내부 레지스트리에 등록해두면 ServiceProxy가 호출 가능
        _SERVICE_REGISTRY[name] = self
        loginfo(f"(shim) Service advertised: {name} [{getattr(srv_class, '__name__', srv_class)}]")

    def shutdown(self):
        if self.resolved_name in _SERVICE_REGISTRY:
            del _SERVICE_REGISTRY[self.resolved_name]
            loginfo(f"(shim) Service shutdown: {self.resolved_name}")


# --- ServiceProxy stub (client side) ---
class ServiceProxy:
    def __init__(self, name, srv_class=None, persistent=False):
        self.name = name
        self.srv_class = srv_class
        self.persistent = persistent

    def __call__(self, *args, **kwargs):
        srv = _SERVICE_REGISTRY.get(self.name)
        if srv is None:
            raise ServiceException(f"(shim) no service named {self.name}")
        try:
            return srv.handler(*args, **kwargs)
        except Exception as e:
            raise ServiceException(f"(shim) error in service [{self.name}]: {e}")


# internal registry
_SERVICE_REGISTRY = {}


# --------------------------
# Publisher/Subscriber/Service stubs
# --------------------------
class Publisher:
    def __init__(self, name, data_class=None, queue_size=10, latch=False):
        self.name, self.data_class = name, data_class
        loginfo(f"(shim) Publisher {name} [{getattr(data_class, '__name__', data_class)}]")

    def publish(self, msg):
        loginfo(f"(shim) publish to {self.name}: {type(msg).__name__}")


class Subscriber:
    def __init__(self, name, data_class, callback, queue_size=10):
        self.name, self.data_class, self.callback = name, data_class, callback
        loginfo(f"(shim) Subscriber {name}")


class ServiceProxy:
    def __init__(self, name, srv_class=None, persistent=False):
        self.name = name
        self.srv_class = srv_class

    def __call__(self, *a, **kw):
        # raise ServiceException(f"(shim) no service [{self.name}]")
        if self.srv_class is Trigger:
            return Trigger._response_class(True)
        loginfo(f"(shim) no service [{self.name}]")


def wait_for_service(name, timeout=None):
    logwarn(f"(shim) wait_for_service({name}) – always ok")
