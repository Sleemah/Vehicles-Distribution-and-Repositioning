from torch.multiprocessing import Pool, Process, set_start_method

#if __name__  != 'main':
#    try:
#        set_start_method('spawn')
#    except RuntimeError:
#        pass

class MyProcess(object):
    def __init__(self, processNum):
        self.pool = Pool(processes = processNum)
        self.result = []

    def run(self, func,args=()):
        self.result.append(self.pool.apply_async(func,args))

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

    def reInitial(self):
        self.result = []