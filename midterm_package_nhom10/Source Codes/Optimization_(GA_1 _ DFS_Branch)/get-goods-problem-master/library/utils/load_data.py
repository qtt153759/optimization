import numpy as np

class Load:
    def __init__(self) -> None:
        self.num_kinds: int
        self.num_bins: int
        self.mat_info: np.ndarray
        self.mat_dis: np.ndarray
        self.order: np.ndarray
    def __call__(self, path):
        data = open(path)
        
        # Load number categories goods, number bins
        self.num_kinds, self.num_bins  = np.array(data.readline().split(), dtype=int)
        
        # Load matrix information about bins
        self.mat_info = np.zeros((self.num_kinds, self.num_bins),dtype=int)
        for line in range(self.num_kinds):
            self.mat_info[line,:] = np.array(data.readline().split(), dtype=int)

        # Load matrix distance between bins
        self.mat_dis = np.zeros((self.num_bins + 1, self.num_bins + 1),dtype=int)
        for line in range(self.num_bins + 1):
            self.mat_dis[line, :] = np.array(data.readline().split(), dtype=int)

        self.order = np.array(data.readline().split(), dtype=int)
        # self.solution_path = np.array(data.readline().split(), dtype=int)
        # self.min_cost = int(data.readline())

    def __repr__(self) -> str:
        return f"\t+ Number of the categories goods is {self.num_kinds}.\
        \n\t+ Number of the bins is {self.num_bins}   \
        \n\t+ Matrix Q - {self.num_kinds}x{self.num_bins} save information of the goods position and bins with {self.num_kinds} categories goods and {self.num_bins} bins.\
        \n\t+ Matrix d - {self.num_bins + 1}x{self.num_bins + 1} save distance between bins\
        \n\t+ q size {len(self.order)} is array save the orders with each category goods: \n\t   {self.order}"
        # \n\t+ Shortest path: {self.solution_path}\
        # \n\t+ Cost: {self.min_cost}"
