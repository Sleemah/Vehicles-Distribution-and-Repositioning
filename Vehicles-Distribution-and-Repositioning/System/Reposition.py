class Reposition(object):
    def __init__(self,id,action,supply):
        self.id = id
        self.supply = supply
        self.action  = action
        self.repositionV = abs(self.supply*self.action)