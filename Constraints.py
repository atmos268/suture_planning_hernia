import DistanceCalculator

class Constraints:
    def __init__(self):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = 0.3 # TODO Varun: this is a random #, lookup

    def con2(self, t):
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)   
        h = 0.3
        return [i - h for i in insert_dists] + [i - h for i in center_dists] + [i - h for i in extract_dists]
    
    def con3(self, t): # max distance b/w 2 sutures
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)   
        h = 0.7
        return [h - i for i in insert_dists] + [h - i for i in center_dists] + [h - i for i in extract_dists]
    
    # checks if p lies on the segment p1p2
    def on_segment(self, p1, p2, p):
        return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])
    
    def cross_product(self, p1, p2):
        return p1[0] * p2[1] - p2[0] * p1[1]
    
    def direction(self, p1, p2, p3):
        return self.cross_product((p3[0]-p1[0], p3[1] - p1[1]), (p2[0]-p1[0], p2[1] - p1[1]))
    
    # checks if line segment p1p2 and p3p4 intersect and returns -1 if True and 1 when False
    def intersect(self, p1, p2, p3, p4):
        d1 = self.direction(p3, p4, p1)
        d2 = self.direction(p3, p4, p2)
        d3 = self.direction(p1, p2, p3)
        d4 = self.direction(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return -1
        elif d1 == 0 and self.on_segment(p3, p4, p1):
            return -1
        elif d2 == 0 and self.on_segment(p3, p4, p2):
            return -1
        elif d3 == 0 and self.on_segment(p1, p2, p3):
            return -1
        elif d4 == 0 and self.on_segment(p1, p2, p4):
            return -1
        else:
            return 1
        
    def con4(self, t): #to avoid crossings
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)  
        lst = [self.intersect(insert_pts[i], extract_pts[i], insert_pts[i+1], extract_pts[i+1]) for i in range(len(insert_pts)-1)] 
        return lst

    def constraints(self):
        start = self.wound_points[0]
        end = self.wound_points[-1]
        return ({'type': 'eq', 'fun': lambda t: t[0] - start}, {'type': 'eq', 'fun': lambda t: t[-1] - end}, 
               {'type': 'ineq', 'fun': lambda t: self.con2(t)}, {'type': 'ineq', 'fun': lambda t: self.con3(t)},
               {'type': 'ineq', 'fun': lambda t: self.con4(t)}
               )