import DistanceCalculator

class Constraints:
    def __init__(self):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = 0.3 # TODO Varun: this is a random #, lookup

    def con2(self, t):
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)   
        h = 0.3
        return [i - h for i in insert_dists] + [i - h for i in center_dists] + [i - h for i in extract_dists]

    def constraints(self):
        start = self.wound_points[0]
        end = self.wound_points[-1]
        return ({'type': 'eq', 'fun': lambda t: t[0] - start}, {'type': 'eq', 'fun': lambda t: t[-1] - end}, 
               {'type': 'ineq', 'fun': lambda t: self.con2(t)}
               )