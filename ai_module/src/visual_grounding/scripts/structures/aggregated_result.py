import itertools
import math
import threading
from typing import List, Any, Tuple, Union, Dict


# -------------------- helpers --------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-9) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


def _clip01(p: float) -> float:
    return max(0.0, min(1.0, float(p)))


class InferenceResults:
    _order_counter = itertools.count()

    def __init__(
            self, id: int, method: str = "logit_pool", prior: float = 0.5, keep_top_k: int = 3):
        self.id = id
        self.method = method
        self.prior = prior
        self.keep_top_k = int(keep_top_k)

        self._lock = threading.RLock()

        self.results: Dict[Any, Dict[str, Any]] = {}

        self.num_queries: int = 0

    def __repr__(self):
        data = []
        for k, v in self.results.items():
            data.append(f"{str(k)}({v.get('confidence', 0.0):.2f}/{v.get('count', 0)}/{v.get('order', -1)})")
        return f"InferenceResults(conf/cnt/ord): {data}"

    # ---------- 내부 유틸 ----------
    def _combine(self, old_p: float, new_p: float) -> float:
        if self.method == "logit_pool":
            L = _logit(old_p) + _logit(new_p)
            return _clip01(_sigmoid(L))
        raise ValueError(f"Unknown method: {self.method}")

    def _prune(self) -> None:
        k = self.keep_top_k
        if k <= 0 or len(self.results) <= k:
            return
        items = sorted(
            self.results.items(),
            key=lambda kv: (kv[1]["confidence"], kv[1]["count"], kv[1].get("order", -1)),
            reverse=True,
        )
        self.results = {a: d for a, d in items[:k]}

    # ---------- 업데이트 ----------
    def inc_queries(self, n: int = 1) -> None:
        self.num_queries += int(n)

    def update(self, answer, confidence, count, *args, **kwargs):
        ord_no = next(self._order_counter)
        answer = str(answer)
        with self._lock:
            if answer not in self.results:
                combined = self._combine(self.prior, confidence)
                self.results[answer] = {
                    "confidence": _clip01(combined),
                    "count": count,
                    "order": ord_no,
                }
            else:
                prev = self.results[answer]
                prev["confidence"] = _clip01(self._combine(prev["confidence"], confidence))
                prev["count"] += count
                prev["order"] = ord_no
        self._prune()

    # ---------- 조회 ----------
    @property
    def best_confidence(self) -> float:
        with self._lock:
            if not self.results:
                return 0.0
            return max(d["confidence"] for d in self.results.values())

    @property
    def best_answer(self) -> Any:
        with self._lock:
            if not self.results:
                return None
            best_ans, _ = max(
                self.results.items(),
                key=lambda kv: (kv[1]["confidence"], kv[1]["count"], kv[1].get("order", -1)),
            )
            return best_ans

    def snapshot(self) -> dict:
        with self._lock:
            items = sorted(
                self.results.items(),
                key=lambda kv: (kv[1]["confidence"], kv[1]["count"], kv[1].get("order", -1)),
                reverse=True,
            )
            return {
                "id": self.id,
                "num_queries": self.num_queries,
                "results": [
                    {
                        "answer": str(a),  # or a 자체가 직렬화 가능하면 그대로
                        "confidence": d["confidence"],
                        "count": d["count"],
                        "order": d.get("order", -1),
                    }
                    for a, d in items
                ],
            }


class InferenceResultsPerEntity(InferenceResults):
    _order_counter = itertools.count()

    def __init__(self, id=0, *args, **kwargs):
        super().__init__(id=id, *args, **kwargs)
        self.num_queries = {}

    def update(self, answer, confidence, count, *args, **kwargs):
        ord_no = next(self._order_counter)

        candidate_eids = answer.eids
        with self._lock:
            # others = [eid for eid in candidate_eids if str(eid) != str(answer)]
            # other_conf = (1.0 - confidence) / len(others) if others else 0.0
            other_conf = (1.0 - confidence)

            for eid in candidate_eids:
                conf = confidence if str(eid) == str(answer) else other_conf

                if eid not in self.results:
                    combined = self._combine(self.prior, conf)
                    self.results[eid] = {
                        "confidence": _clip01(combined),
                        "count": count,
                        "order": ord_no,
                    }
                else:
                    prev = self.results[eid]
                    prev["confidence"] = _clip01(self._combine(prev["confidence"], conf))
                    prev["count"] += count
                    prev["order"] = ord_no
        self._prune()

    # ---------- 업데이트 ----------
    def inc_queries(self, eids: Union[List[int], Tuple[int]] = []) -> None:
        for eid in eids:
            self.num_queries[eid] = self.num_queries.get(eid, 0) + 1


class AggregatedResult:
    def __init__(self, min_query=5, inference_cfg=None, action=None, *args, **kwargs):
        self.min_query = min_query
        self.inference_cfg = inference_cfg or {}
        self.action = action

        self.results: Dict[int, InferenceResults] = {}
        self.results_by_entity = InferenceResultsPerEntity(**inference_cfg)

        self._scheduled: Dict[int, int] = {}
        self._scheduled_by_entity: Dict[int, int] = {}
        self._lock = threading.RLock()

    def __repr__(self):
        if self.action == 'find':
            n = len(self.results_by_entity.results)
            if n == 0:
                return "AggResults(#=0)"
            lines = [
                f"  > EID[{eid}] ({r.get('confidence', 0.0):.2f}/{r.get('count', 0)}/{r.get('order', 0)})"
                for eid, r in sorted(self.results_by_entity.results.items(), key=lambda x: x[0])
            ]
            return f"AggResults(#={n}) (conf/count/order):\n" + "\n".join(lines)
        elif self.action == 'count':
            n = len(self.results)
            if n == 0:
                return "AggResults(#=0)"
            lines = [
                f"  > GID[{gid}]: BestAns={r.best_answer} ({r.best_confidence:.2f})"
                for gid, r in sorted(self.results.items(), key=lambda x: x[0])
            ]
            return f"AggResults(#={n}) BestAnswer (conf/count):\n" + "\n".join(lines)
        else:
            raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")

    # ---------- Pending(최소 질의 미달) ----------
    def get_pending_results(self) -> List[InferenceResults]:
        """
        min_query보다 적게 query된 결과들을 'num_queries 오름차순'으로 반환
        """
        with self._lock:
            pending = [results for id, results in self.results.items() if results.num_queries < self.min_query]
            pending.sort(key=lambda r: r.num_queries)
            return pending

    def remaining_queries(self, id: int) -> int:
        """gid에 대해 min_query까지 남은 쿼리 수 계산."""
        return max(0, self.min_query - self.results[id].num_queries)

    def remaining_queries_by_entity(self, eid: int) -> int:
        """gid에 대해 min_query까지 남은 쿼리 수 계산."""
        return max(0, self.min_query - self.results_by_entity.num_queries.get(eid, 0))

    def get_pending_ids(self) -> List[int]:
        return [r.id for r in self.get_pending_results()]

    def get_pending_eids(self) -> List[int]:
        with self._lock:
            pending_eids = sorted(
                [eid for eid, num_query in self.results_by_entity.num_queries.items()
                 if num_query < self.min_query],
                key=lambda eid: self.results_by_entity.num_queries[eid]
            )
        return pending_eids

    # ---------- 결과/카운트 누적 ----------
    def generate(self, gid: int, eids: Union[List[int], Tuple[int]]=[]) -> None:
        with self._lock:
            self.results[gid] = InferenceResults(id=gid, **self.inference_cfg)
            self._scheduled.setdefault(gid, 0)
            for eid in eids:
                self._scheduled_by_entity.setdefault(eid, 0)
            return

    def update(self, gid: int, answer: Any, confidence: float, count: int = 1, *args, **kwargs) -> None:
        with self._lock:
            if gid not in self.results:
                self.generate(gid, answer.eids)
            self.results[gid].update(answer, confidence, count, *args, **kwargs)
            self.results_by_entity.update(answer, confidence, count, *args, **kwargs)

    # ---------- 예약/완료 ----------
    def schedule(self, gid: int, n: int = 1, data=None) -> None:
        if n <= 0:
            return
        with self._lock:
            self._scheduled[gid] = self._scheduled.get(gid, 0) + n
            for d in (data or []):
                for eid in d['eids']:
                    self._scheduled_by_entity[eid] = self._scheduled_by_entity.get(eid, 0) + 1

    def release(self, gid: int, n: int = 1, eids: Union[List[int], Tuple[int]]=[]) -> None:
        if n <= 0:
            return
        with self._lock:
            self._scheduled[gid] = max(0, self._scheduled.get(gid, 0) - n)
            for eid in eids:
                self._scheduled_by_entity[eid] = max(0, self._scheduled_by_entity.get(eid, 0) - 1)

    def inc_queries(self, gid: int, n: int = 1, eids: Union[List[int], Tuple[int]]=[]) -> None:
        with self._lock:
            self.results[gid].inc_queries(n)
            self.results_by_entity.inc_queries(eids=eids)

    # ---------- 조회 ----------
    @property
    def best_confidence(self) -> float:
        with self._lock:
            if not self.results:
                return 0.0
            if self.action == 'find':
                return self.results_by_entity.best_confidence
            elif self.action == 'count':
                return max([d.best_confidence for d in self.results.values()])
            else:
                raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")

    @property
    def best_answer(self) -> Any:
        with self._lock:
            if self.action == 'find':
                results = self.results_by_entity.results
            elif self.action == 'count':
                results = self.results
            else:
                raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")
        if not results:
            return None
        best_id = None
        best_c = float("-inf")
        best_ans = None
        for id, r in results.items():
            if isinstance(r, dict):
                conf = r.get('confidence', 0.0)
            else:
                conf = r.best_confidence
            if conf > best_c:
                best_c = conf
                best_id = id
                if self.action == 'find':       best_ans = id
                elif self.action == 'count':    best_ans = r.best_answer
                else: raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")
        return best_ans

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "min_query": self.min_query,
                "groups": {
                    r.id: r.snapshot()
                    for i, r in self.results.items()
                },
                "best_answer": self.best_answer,
                "best_confidence": self.best_confidence,
            }
