from typing import Dict, List


class DSU:
    def __init__(self):
        self.ids: List[int] = []
        self.idx: Dict[int, int] = {}
        self.p: List[int] = []
        self.r: List[int] = []

    def add(self, eid: int):
        self.idx[eid] = len(self.ids)
        self.ids.append(eid)
        self.p.append(len(self.p))
        self.r.append(0)

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def root(self, eid: int) -> int:
        return self.find(self.idx[eid])

    def union_eids(self, a: int, b: int):
        ia, ib = self.idx[a], self.idx[b]
        ra, rb = self.find(ia), self.find(ib)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
