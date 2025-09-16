import time
from collections import defaultdict


class Timer:
    def __init__(self):
        self.sum = defaultdict(float)
        self.cnt = defaultdict(int)
        self._stk = []

    def tic(self, k):
        self._stk.append((k, time.perf_counter()))

    def toc(self):
        k, t0 = self._stk.pop()
        dt = time.perf_counter() - t0
        self.sum[k] += dt
        self.cnt[k] += 1

    def report(self, reset=True):
        print("---- TIMER ----")
        for k in sorted(self.sum, key=lambda x: -self.sum[x]):
            s = self.sum[k]
            n = self.cnt[k]
            print(f"{k:20s}: {s:.4f}s ({n}x) avg={s / max(1, n):.6f}")
        if reset:
            self.sum.clear()
            self.cnt.clear()


class Stats:
    def __init__(self):
        self.c = defaultdict(int)
        self.m = defaultdict(list)

    def inc(self, k, v=1):
        self.c[k] += v

    def push(self, k, v):
        self.m[k].append(v)

    def report(self, reset=True):
        print("---- STATS ----")
        for k in sorted(self.c):
            print(f"{k:20s}: {self.c[k]}")
        for k in sorted(self.m):
            arr = self.m[k]
            print(f"{k:20s}: mean={sum(arr) / max(1, len(arr)):.2f}, max={max(arr) if arr else 0}")
        if reset:
            self.c.clear()
            self.m.clear()
