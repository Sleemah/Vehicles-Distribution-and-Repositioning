class Hunt(object):
    def __init__(self,id,action,supply):
        self.id = id
        self.supply = supply
        self.action  = action
        self.huntV = abs(self.supply*self.action)