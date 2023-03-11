import torch as torch

class RewardFunction():
    def __init__(self):
        self.insert_dists = []
        self.center_dists = []
        self.extract_dists = []
    
    # distance lists added to this object by SuturePlacer.
    # variance
    def final_loss(self, a = 1, b = 1):
        return a * self.lossVar()

    def lossVar(self):
        mean_insert = sum(self.insert_dists) / len(self.insert_dists)
        var_insert = sum([(i - mean_insert)**2 for i in self.insert_dists])
        
        mean_center = sum(self.center_dists) / len(self.center_dists)
        var_center = sum([(i - mean_center)**2 for i in self.center_dists])
        
        mean_extract = sum(self.extract_dists) / len(self.extract_dists)
        var_extract = sum([(i - mean_extract)**2 for i in self.extract_dists])
        
        return var_insert + var_center + var_extract
    
    # def lossIdeal(self):
    #     ideal = 0.4
    #     power = 2
    #     extra_pen = 100
    #     insertion = []
    #     extraction = []
    #     center = []
    #     for i in range(len(self.insert_dists)):
    #         ins = self.insert_dists[i]
    #         if ins < ideal:
    #             insertion.append((ins-ideal) ** power * extra_pen)
    #         else:
    #             insertion.append((ins-ideal) ** power)
    #         ext = self.extract_dists[i]
    #         if ext < ideal:
    #             extraction.append((ext-ideal) ** power * extra_pen)
    #         else:
    #             extraction.append((ext-ideal) ** power)
    #         cen = self.insert_dists[i]
    #         if cen < ideal:
    #             center.append((cen-ideal) ** power * extra_pen)
    #         else:
    #             center.append((cen-ideal) ** power)
    #     return sum(insertion + center + extraction)

    #min - max
    def lossMinMax(self):
        return (max(self.insert_dists) + max(self.center_dists) + max(self.extract_dists))
    
    def hyperLoss(self):
        ideal = 5
        return sum([(i - ideal)**2 for i in self.insert_dists]) + sum([(i - ideal)**2 for i in self.extract_dists]) + sum([(i - ideal)**2 for i in self.center_dists])


    # ... and so forth: refer to slide 14 from the presentation for details on how to design this.
    # It may be influenced by the optimizer as well.