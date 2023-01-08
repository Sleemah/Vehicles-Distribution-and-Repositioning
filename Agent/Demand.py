from Tool import ToolFunction


class Demand(object):
    def __init__(self, time, origin, destination, value = 0, lifeTime= 2):
        self.time = int(time)                                      #demand time
        self.origin = int(origin)                                    #demand location
        self.destination = int(destination)                          #demand target
        self.value = value                                      #demand value
        self.tripTime = ToolFunction.getManDist(self.origin, self.destination)   #demand travel time
        self.lifeTime = lifeTime
        if self.tripTime == 0:
            self.tripTime += 1

    def __lt__(self, other):  # override <operator
        if self.time < other.time:
            return True
        return False