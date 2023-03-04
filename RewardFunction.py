import torch as torch

class RewardFunction():
    def __init__(self):
        self.insert_dists = []
        self.center_dists = []
        self.extract_dists = []
    
    # distance lists added to this object by SuturePlacer.
    # variance
    def final_loss(self, a = 1, b = 1):
        return a * self.lossVar() + b * self.lossIdeal()

    def lossVar(self):
        mean_insert = sum(self.insert_dists) / len(self.insert_dists)
        var_insert = sum([(i - mean_insert)**2 for i in self.insert_dists])
        
        mean_center = sum(self.center_dists) / len(self.center_dists)
        var_center = sum([(i - mean_center)**2 for i in self.center_dists])
        
        mean_extract = sum(self.extract_dists) / len(self.extract_dists)
        var_extract = sum([(i - mean_extract)**2 for i in self.extract_dists])
        
        return var_insert + var_center + var_extract
    
    def lossIdeal(self):
        ideal = 0.4
        [(i - ideal) ** 2 for i in self.insert_dists]
        return sum([(i - ideal) ** 2 for i in self.insert_dists] + [(i - ideal) ** 2 for i in self.center_dists] + [(i - ideal) ** 2 for i in self.extract_dists])

    #min - max
    def lossMinMax(self):
        return (max(self.insert_dists) + max(self.center_dists) + max(self.extract_dists))
    


    # ... and so forth: refer to slide 14 from the presentation for details on how to design this.
    # It may be influenced by the optimizer as well.