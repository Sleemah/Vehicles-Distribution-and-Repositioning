

class Supply(object):
    def __init__(self, time, origin, destination, dist):
        self.time = time                        #departure time
        self.origin = origin                    #starting point
        self.destination = destination          #arrived
        self.arriveTime = self.time+dist        #Time of arrival

    def __lt__(self, other):  # override <operator
        if self.arriveTime < other.arriveTime:
            return True
        return False