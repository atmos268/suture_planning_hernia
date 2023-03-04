def on_segment(p1, p2, p):
        return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])
    
def cross_product(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]

def direction(p1, p2, p3):
    return cross_product((p3[0]-p1[0], p3[1] - p1[1]), (p2[0]-p1[0], p2[1] - p1[1]))

# checks if line segment p1p2 and p3p4 intersect and returns -1 if True and 1 when False
def intersect(p1, p2, p3, p4):
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return -1
    elif d1 == 0 and on_segment(p3, p4, p1):
        return -1
    elif d2 == 0 and on_segment(p3, p4, p2):
        return -1
    elif d3 == 0 and on_segment(p1, p2, p3):
        return -1
    elif d4 == 0 and on_segment(p1, p2, p4):
        return -1
    else:
        return 1
    
print(intersect([0,0], [1,1], [0, 0], [0,1]))