import numpy as np


def parse_id(s):
    try:
        return int(s.split("_")[-1])
    except:
        return -1


