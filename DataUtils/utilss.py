

class Best_Result:
    """
    Best_Result
    """
    def __init__(self):
        self.current_dev_score = -1
        self.best_dev_score = -1
        self.best_score = -1
        self.best_epoch = 1
        self.best_test = False
        self.early_current_patience = 0
        self.p = -1
        self.r = -1
        self.f = -1


