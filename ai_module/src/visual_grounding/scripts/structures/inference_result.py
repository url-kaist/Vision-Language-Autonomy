import threading
import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from math import log, exp
from typing import List, Union
from ai_module.src.visual_grounding.scripts.utils.utils_message import object_to_marker
from ai_module.src.visual_grounding.scripts.structures.keyframe import Keyframes
from ai_module.src.visual_grounding.scripts.structures.entity import Entities

try:
    from std_msgs.msg import String, Int32
except:
    pass


def get_confidence(etype):
    if etype == 'object':       return 0.55
    elif etype == 'detection':  return 0.5
    elif etype == 'image':      return 0.6
    elif etype == 'all':        return 0.6
    else:
        raise ValueError(f"etype must be in ['object', 'detection', 'image'], but {etype} was given.")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _logit(p: float, eps: float = 1e-9) -> float:
    p = min(max(p, eps), 1 - eps)
    return log(p / (1 - p))


def _clip01(p: float) -> float:
    return min(max(p, 0.0), 1.0)


@dataclass # TODO: integrate with InferenceResult and InferenceResults
class InferenceResult2:
    def __init__(self, objects=None, keyframes=None, reason="", stamp=0.0, confidence=0.0, validation_count=0, has_candidate=False):
        self.objects = objects if objects else Entities() # TODO: objects -> entities
        self.keyframes = keyframes if keyframes else Keyframes()
        self.reason = reason
        self.stamp = stamp
        self.confidence = confidence
        self.validation_count = validation_count
        self.has_candidate = has_candidate

    def __str__(self) -> str:
        return (
            f"InferenceResult("
            f"object_ids={self.objects.ids}, "
            f"keyframe_ids={self.keyframes.ids if self.keyframes else []}, "
            f"confidence={self.confidence:.2f}, "
            f"validation_count={self.validation_count}, "
            f"has_candidate={self.has_candidate})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def pretty(self) -> str:
        try:
            keyframes_len = len(self.keyframes) if self.keyframes else 0
        except (TypeError, AttributeError):
            keyframes_len = 0
            
        return (
            f"object_ids       : {self.objects.ids}\n"
            f"keyframes        : {keyframes_len}\n"
            f"reason           : {self.reason}\n"
            f"stamp            : {self.stamp}\n"
            f"confidence       : {self.confidence}\n"
            f"validation_count : {self.validation_count}"
        )

    def get_answer_msg(self, action, **kwargs):
        if action in ['find']:
            object_id = self.objects.ids[0]
            object = self.objects.get_single(object_id)
            return object_to_marker(object, object_id, color=(0.0, 0.0, 1.0, 1.0), style='cube')
        elif action in ['count']:
            count = len(self.objects.ids)
            return Int32(count)
        else:
            return String("")

    def get_answer(self, action, **kwargs):
        if action in ['find']:
            object_id = self.objects.ids[0]
            return object_id
        elif action in ['count']:
            count = len(self.objects.ids)
            return str(count)
        else:
            object_ids = self.objects.ids
            return object_ids

    def get_answer_vis_msg(self, action, color=(0.0, 0.0, 1.0, 1.0), style='box'):
        if action in ['find']:
            object_id = self.objects.ids[0]
            object = self.objects.get_single(object_id)
            marker = object_to_marker(object, object_id, color=color, style=style)
            return [marker]
        elif action in ['count']:
            # Create markers for all counted objects
            markers = []
            for object_id, object in self.objects.items():
                marker = object_to_marker(object, object_id, color=color, style=style)
                markers.append(marker)
            return markers
        else:
            markers = []
            for object_id, object in self.objects.items():
                marker = object_to_marker(object, object_id, color=color, style=style)
                markers.append(marker)
            return markers
        
    def update_confidence(self, obs_confidence):
        prev_n = self.validation_count
        prev_c = self.confidence

        # Update confidence using running average
        new_n = prev_n + 1
        new_c = (prev_c * (prev_n + 1) + obs_confidence) / (new_n + 1)
        
        self.confidence = new_c
        self.validation_count = new_n

    def __lt__(self, other):
        if self.has_candidate != other.has_candidate:
            return self.has_candidate
        return self.stamp > other.stamp

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __eq__(self, other):
        if not isinstance(other, InferenceResult2):
            return False
        return (self.objects == other.objects and
                self.keyframes == other.keyframes and
                self.reason == other.reason and
                self.stamp == other.stamp and
                self.has_candidate == other.has_candidate)

class InferenceResult2PriorityQueue:
    """
    InferenceResult2 객체들을 효율적으로 관리하는 우선순위 큐
    - has_candidate가 True인 객체가 가장 높은 우선순위
    - 그 다음에는 stamp가 클수록(더 늦게 생성된 것) 우선순위가 높음
    - maxlen 설정 시 최대 개수 제한 (가장 낮은 우선순위부터 제거)
    """

    def __init__(self, maxlen=None):
        self._heap = []
        self._size = 0
        self._maxlen = maxlen

    def push(self, result: InferenceResult2):
        """우선순위 큐에 InferenceResult2 객체 추가"""
        if not isinstance(result, InferenceResult2):
            raise TypeError("Only InferenceResult2 objects can be added to the queue")

        # heapq는 min-heap이므로 우선순위를 음수로 변환하여 max-heap처럼 동작
        # InferenceResult2의 __lt__ 메서드를 활용
        heapq.heappush(self._heap, (-self._size, result))  # -self._size로 안정성 보장
        self._size += 1

        # maxlen이 설정되어 있고 초과했으면 가장 낮은 우선순위 제거
        if self._maxlen is not None and len(self._heap) > self._maxlen:
            self._remove_lowest_priority()

    def pop(self) -> InferenceResult2:
        """가장 높은 우선순위의 InferenceResult2 객체 반환 및 제거"""
        if self.empty():
            raise IndexError("Queue is empty")

        _, result = heapq.heappop(self._heap)
        return result

    def peek(self) -> InferenceResult2:
        """가장 높은 우선순위의 InferenceResult2 객체 반환 (제거하지 않음)"""
        if self.empty():
            raise IndexError("Queue is empty")

        _, result = self._heap[0]
        return result

    def empty(self) -> bool:
        """큐가 비어있는지 확인"""
        return len(self._heap) == 0

    def size(self) -> int:
        """큐의 크기 반환"""
        return len(self._heap)

    def clear(self):
        """큐 비우기"""
        self._heap.clear()
        self._size = 0

    def _remove_lowest_priority(self):
        """가장 낮은 우선순위의 객체 제거 (maxlen 초과 시 사용)"""
        if not self._heap:
            return

        # 모든 객체를 우선순위 순서로 정렬하여 가장 낮은 우선순위 찾기
        sorted_heap = sorted(self._heap, key=lambda x: x[1], reverse=True)  # 내림차순 정렬
        lowest_item = sorted_heap[0]  # 가장 낮은 우선순위

        # 해당 아이템을 힙에서 제거
        self._heap.remove(lowest_item)
        heapq.heapify(self._heap)  # 힙 속성 복원

    def to_list(self) -> list:
        """큐의 모든 객체를 우선순위 순서대로 리스트로 반환 (큐는 변경하지 않음)"""
        sorted_heap = sorted(self._heap, key=lambda x: x[1])  # InferenceResult2의 비교 메서드 사용
        return [result for _, result in sorted_heap]

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return not self.empty()

    def __iter__(self):
        """우선순위 순서대로 반복"""
        sorted_heap = sorted(self._heap, key=lambda x: x[1])  # InferenceResult2의 비교 메서드 사용
        for _, result in sorted_heap:
            yield result

    @property
    def maxlen(self):
        """최대 개수 반환"""
        return self._maxlen

    @maxlen.setter
    def maxlen(self, value):
        """최대 개수 설정 (기존 큐 크기 초과 시 가장 낮은 우선순위부터 제거)"""
        self._maxlen = value
        if value is not None and len(self._heap) > value:
            # 초과분만큼 가장 낮은 우선순위 제거
            while len(self._heap) > value:
                self._remove_lowest_priority()

@dataclass
class InferenceResult:
    def __init__(
            self, answer, confidence: float = 0.5, count: int = 1,
            gid: int = None, eids: Union[List, None] = None, *args, **kwargs):
        self.answer = answer
        self.confidence = confidence
        self.count = count
        self.gid = gid
        self.eids = eids or []


@dataclass
class InferenceResults:
    _counter = itertools.count()
    def __init__(
            self, results=None,
            method: str = "logit_pool", prior: float = 0.5, keep_top_k: int = 10,
            min_query: int = 10, max_query: int = 15,
    ):
        self._lock = threading.Lock()
        self.keyframes = None
        self.results = results or {}
        self.method = method
        self.prior = float(prior)
        self.keep_top_k = int(keep_top_k)

        self.group_query_counter = defaultdict(int)   # 완료된 쿼리 수
        self.group_scheduled_counter = defaultdict(int)  # 큐에 "예약"된 쿼리 수(아직 미완료)
        self.group_eids = defaultdict(set)

        self.min_query, self.max_query = min_query, max_query

    def __str__(self) -> str:
        if not self.results:
            if len(self.group_query_counter) == 0:
                return "InferenceResults(empty)"
            else:
                return f"InferenceResults(empty) | Counter={self.group_query_counter}"
        items = sorted(self.results.items(),
                       key=lambda kv: (kv[1]['confidence'], kv[1]['count']),
                       reverse=True)[:self.keep_top_k]
        parts = [f"{ans}({data['confidence']:.2f},{data['count']})"
                 for ans, data in items]
        return f"InferenceResults(conf,cnt): " + ", ".join(parts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 언피클 시 다시 새 락 생성
        self._lock = threading.Lock()

    @property
    def confidence(self):
        if not self.results:
            return 0.0
        return max(data['confidence'] for data in self.results.values())

    @property
    def count(self):
        if not self.results:
            return 0.0
        return max(data['count'] for data in self.results.values())

    @property
    def answer(self):
        if not self.results:
            return None
        best_ans, _ = max(
            self.results.items(),
            key=lambda kv: (kv[1]['confidence'], kv[1]['count'], kv[1].get('order', -1))
        )
        return best_ans

    def effective_count(self, gid: int) -> int:
        # 완료 + 예약 합산
        with self._lock:
            return self.group_query_counter.get(gid, 0) + self.group_scheduled_counter.get(gid, 0)

    def schedule(self, gid: int, n: int = 1) -> None:
        # enqueue 직후 호출해서 예약 수 증가
        if n <= 0: return
        with self._lock:
            self.group_scheduled_counter[gid] += n

    def release(self, gid: int, n: int = 1) -> None:
        # 작업 1건 처리(성공/실패 불문) 시 예약 해제
        if n <= 0: return
        with self._lock:
            c = self.group_scheduled_counter.get(gid, 0)
            self.group_scheduled_counter[gid] = max(0, c - n)

    def pending_gids(self, strict: bool = False):
        """
        strict=True  -> (#완료 + #예약) < max_query
        strict=False -> (#완료 + #예약) < min_query
        """
        trial_thres = self.max_query if strict else self.min_query

        # counter에 없는 gid도 포함하려면 union 키셋 사용
        all_gids = set(self.group_query_counter.keys()) | \
                   set(self.group_scheduled_counter.keys()) | \
                   set(self.group_eids.keys())

        # (gid, 유효 카운트)로 정렬
        sorted_eff = sorted(
            ((gid, self.effective_count(gid)) for gid in all_gids),
            key=lambda x: x[1]
        )

        gids = []
        for gid, eff_cnt in sorted_eff:
            if eff_cnt >= trial_thres:
                break
            gids.append(gid)
        return gids

    def _combine(self, old_p:float, new_p: float) -> float:
        m = self.method
        if m == 'logit_pool':
            L = _logit(old_p) + _logit(new_p)
            return _sigmoid(L)
        else:
            raise ValueError(f"Unknown method: {m}")

    def _prune(self) -> None:
        top_k = self.keep_top_k
        if top_k <= 0 or len(self.results) <= top_k:
            return
        items = sorted(
            self.results.items(),
            key=lambda kv: (kv[1]['confidence'], kv[1]['count'], kv[1].get('order', -1)),
            reverse=True
        )
        self.results = {k: v for k, v in items[:top_k]}

    def update(self, others: "InferenceResult"):
        try:
            ans = others.answer
            conf = float(others.confidence)
            cnt = max(1, int(others.count))
            ord_no = next(self._counter)
        except Exception as e:
            raise RuntimeError(f"InferenceResults.err.1")

        try:
            gid = others.gid
            if gid is not None:
                with self._lock:
                    self.group_query_counter[gid] += 1

                    eids = others.eids or []
                    if not isinstance(eids, (set, list, tuple)):
                        eids = [eids]
                    self.group_eids[gid].update(eids)
        except Exception as e:
            raise RuntimeError(f"InferenceResults.err.2")

        try:
            if ans.count is not None:
                if ans.count == 0:
                    print("Count is 0")
                    return
            elif ans.object is not None:
                if ans.object.id is None:
                    print("object is None")
                    return
        except Exception as e:
            raise RuntimeError(f"InferenceResults.err.3")

        try:
            if ans not in self.results:
                combined = self._combine(self.prior, conf)
                self.results[ans] = {"confidence": _clip01(combined), "count": cnt, "order": ord_no}
            else:
                prev = self.results[ans]
                prev['confidence'] = _clip01(self._combine(prev['confidence'], conf))
                prev['count'] += cnt
                prev['order'] = ord_no

            self._prune()
        except Exception as e:
            raise RuntimeError(f"InferenceResults.err.4")
        return self
