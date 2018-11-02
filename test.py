import numpy as np
y = np.zeros((10,10))
y[2][2] = 2
d = y.reshape(1,y.shape[1]*y.shape[1])

def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

def dedupe_dict(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield val
            seen.add(val)

l = [1,5,9,4,8,7,5,4,8,7,2,3,6,5]
print(list(dedupe(l)))


a = [
    {'x': 1, 'y': 2},
    {'x': 1, 'y': 3},
    {'x': 2, 'y': 3}
]
print(list(dedupe_dict(a, key=lambda d:(d['x'], d['y']))))
print(list(dedupe_dict(a, key=lambda d:d['x'])))


items = [1,2,3,4,5,6,7,8,9,10]
a = slice(2,5,1)
print(a.start,a.stop, a.step)