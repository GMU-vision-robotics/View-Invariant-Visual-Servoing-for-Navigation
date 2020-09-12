# Code based on the following two web pages.
# https://www.cs.helsinki.fi/group/goa/mallinnus/lines/bresenh.html
# https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
#
# Paul McKerley (G00616949)

def slope(p1, p2):
    if p1[0] == p2[0]:
        raise ValueError("Undefined slope for points ({}, {}) and ({}, {})".format(p1[0], p1[1], p2[0], p2[1]))
    return (float(p2[1])-float(p1[1]))/(float(p2[0])-float(p1[0]))

def line_octant(p1, p2):
    m = slope(p1, p2)

    if 0 <= m <= 1 and p1[0] < p2[0]:
        return 0

    if m > 1 and p1[1] < p2[1]:
        return 1

    if m < -1 and p1[1] < p2[1]:
        return 2

    if 0 >= m >= -1 and p2[0] < p1[0]:
        return 3

    if 0 < m <= 1 and p2[0] < p1[0]:
        return 4

    if m > 1 and p2[1] < p1[1]:
        return 5

    if -1 > m and p2[1] < p1[1]: 
        return 6

    if 0 > m >= -1 and p1[0] < p2[0]:
        return 7

    raise Exception("Unknown quadrant for points ({}, {}) and ({}, {})".format(p1[0], p1[1], p2[0], p2[1]))

octant_func_from = [
    lambda p: (p[0]  ,p[1]),
    lambda p: (p[1]  ,p[0]),
    lambda p: (p[1]  ,-p[0]),
    lambda p: (-p[0] ,p[1]),
    lambda p: (-p[0] ,-p[1]),
    lambda p: (-p[1] ,-p[0]),
    lambda p: (-p[1] ,p[0]),
    lambda p: (p[0]  ,-p[1]),
    ]

def get_octant_func_from(p1, p2):
    return octant_func_from[line_octant(p1, p2)]

octant_func_to = [
    lambda p: (p[0]  ,p[1]),
    lambda p: (p[1]  ,p[0]),
    lambda p: (-p[1] ,p[0]),
    lambda p: (-p[0] ,p[1]),
    lambda p: (-p[0] ,-p[1]),
    lambda p: (-p[1] ,-p[0]),
    lambda p: (p[1]  ,-p[0]),
    lambda p: (p[0]  ,-p[1]),
    ]

def get_octant_func_to(p1, p2):
    return octant_func_to[line_octant(p1, p2)]

def line_points(p1, p2):
    if p1[0] != p2[0]:
        from_func = get_octant_func_from(p1, p2)
        to_func   = get_octant_func_to(p1, p2)
        p1 = from_func(p1)
        p2 = from_func(p2)
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        x  = p1[0]
        y  = p1[1]
        eps = 0

        while True:
            yield to_func((x,y))
            x += 1
            if x > p2[0]:
                raise StopIteration
            eps += dy
            if eps * 2 >= dx:
                y += 1
                eps -= dx
    else:
        step = adj  = 1
        if p1[1] > p2[1]:
            step = adj = -1

        for y in range(p1[1], p2[1] + adj, step):
            yield (p1[0],y)
