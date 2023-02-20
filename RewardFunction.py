import torch as torch

class RewardFunction():
    def __init__(self):
        self.insert_dists = []
        self.center_dists = []
        self.extract_dists = []
        self.gradients = {}
        self.gradients["center"] = 0
        self.gradients["extract"] = 0
        self.gradients["insert"] = 0
    
    # distance lists added to this object by SuturePlacer.
    # variance
    def rewardX(self):
        mean_insert = sum(self.insert_dists) / len(self.insert_dists)
        var_insert = sum([(i - mean_insert)**2] for i in self.insert_dists)
        
        mean_center = sum(self.center_dists) / len(self.center_dists)
        var_center = sum([(i - mean_center)**2] for i in self.center_dists)
        
        mean_extract = sum(self.extract_dists) / len(self.extract_dists)
        var_extract = sum([(i - mean_extract)**2] for i in self.extract_dists)
        
        return -(var_insert + var_center + var_extract)

    #min - max
    def rewardA(self):
        return - (max(self.insert_dists) + max(self.center_dists) + max(self.extract_dists))
    
    def rewardX_torch(self):
        insert_dists = torch.tensor(self.insert_dists, requires_grad=True)
        center_dists = torch.tensor(self.center_dists, requires_grad=True)
        extract_dists = torch.tensor(self.extract_dists, requires_grad=True)
        
        loss = -(torch.var(insert_dists) + torch.var(center_dists) + torch.var(extract_dists))
        loss.backward()

        self.gradients["center"] = center_dists.grad
        self.gradients["extract"] = extract_dists.grad
        self.gradients["insert"] = insert_dists.grad

        return loss

    def reward_gradient(self):
        return self.gradients


    # ... and so forth: refer to slide 14 from the presentation for details on how to design this.
    # It may be influenced by the optimizer as well.