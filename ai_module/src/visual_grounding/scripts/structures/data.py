import copy
from typing import Dict, List
from collections.abc import Iterable, MutableMapping
_MISSING = object()


class Data:
    _rename_map = {}
    _delete_keys = []
    _equal_keys = []
    _repr_keys = []

    def __init__(self, data: Dict, *args, **kwargs):
        mapped = {self._rename_map.get(k, k): v for k, v in data.items() if not k in self._delete_keys}
        self.__dict__.update(mapped)

    def _repr_value(self, v, max_list_len=5):
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                return f"<ndarray shape={v.shape}>"
        except Exception:
            pass

        try:
            import torch
            if isinstance(v, torch.Tensor):
                return f"<tensor shape={tuple(v.shape)} dtype={v.dtype}>"
        except Exception:
            pass

        if isinstance(v, float):
            return f"{v:.2f}"
        if isinstance(v, list):
            if len(v) > max_list_len:
                return f"<list len={len(v)}>"
            return "[" + ", ".join(self._repr_value(x, max_list_len=max_list_len) for x in v) + "]"
        if isinstance(v, dict):
            return f"<dict keys={list(v.keys())}>"
        return repr(v)

    def __repr__(self):
        attrs = []
        for key in self._repr_keys:
            if hasattr(self, key):
                attrs.append(f"{key}={self._repr_value(getattr(self, key))}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def equal(self, other, keys=None) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot compare Object with {type(other)}")
        if keys is None:
            keys = self._equal_keys

        result = {}
        for k in keys:
            v1 = getattr(self, k, None)
            v2 = getattr(other, k, None)
            result[k] = (v1 == v2)

        return all(result.values())


class Datas(MutableMapping):
    def __init__(self, init=None, *args, **kwargs):
        self._data = dict(init or {})

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        old = self._data.get(key, _MISSING)
        self._data[key] = value
        if old is _MISSING:
            self.on_insert(key, value)
        else:
            self.on_update(key, old, value)

    def __delitem__(self, key):
        old = self._data[key]
        del self._data[key]
        self.on_delete(key, old)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        T = type(self)
        if not isinstance(other, Datas):
            return NotImplementedError(f"Input must be Datas type, but: {other}")
        return T({**self._data, **other._data})

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def on_insert(self, key, value):
        pass

    def on_update(self, key, old, value):
        pass

    def on_delete(self, key, old):
        pass

    def update(self, *args, **kwargs):
        pass

    @property
    def ids(self):
        return list(self._data.keys())

    def has_id(self, id) -> bool:
        return id in self.ids

    def get(self, ids):
        T = type(self)
        if isinstance(ids, int):
            obj = copy.deepcopy(self._data.get(ids))
            return T({ids: obj}) if obj is not None else T()
        if isinstance(ids, Iterable) and not isinstance(ids, (str, bytes)):
            out = {_id: copy.deepcopy(self._data[_id]) for _id in ids if _id in self._data}
            return T(out)
        raise TypeError(f'id must be int or list, but got {ids}')

    def get_single(self, id):
        return copy.deepcopy(self._data.get(id))

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def keys(self):
        return self._data.keys()

    def single(self):
        if len(self._data) != 1:
            raise ValueError(f"{self.__class__.__name__} must contain exactly one item, "
                             f"but has {len(self._data)}")
        return next(iter(self._data.values()))
