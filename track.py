import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import KDTree


class FireFlyTracker(object):
    def __init__(self, coords: np.ndarray, max_distance: float = 10) -> None:
        """first dimension is interpreted as time"""

        self.G = nx.DiGraph()

        coords = np.sort(coords, axis=0)

        self.tree = KDTree(coords)

        neighbors = self.tree.query_ball_point(coords, max_distance)

        



        
        

        # dist = pdist(coords[:, 1:])

        # for (i, ci), (j, cj) in combinations(enumerate(coords), 2):
        #     n = len(coords) * i + j - ((i + 2) * (i + 1)) // 2
        #     d = dist[n]
            # print(d, np.sqrt(np.sum(ci[1:]-cj[1:])**2))

        # self.G.add_edges_from(
        # [
        #     (1, 2, {"capacity": 12, "weight": 4}),
        #     (1, 3, {"capacity": 20, "weight": 6}),
        #     (2, 3, {"capacity": 6, "weight": -3}),
        #     (2, 6, {"capacity": 14, "weight": 1}),
        #     (3, 4, {"weight": 9}),
        #     (3, 5, {"capacity": 10, "weight": 5}),
        #     (4, 2, {"capacity": 19, "weight": 13}),
        #     (4, 5, {"capacity": 4, "weight": 0}),
        #     (5, 7, {"capacity": 28, "weight": 2}),
        #     (6, 5, {"capacity": 11, "weight": 1}),
        #     (6, 7, {"weight": 8}),
        #     (7, 4, {"capacity": 6, "weight": 6}),
        # ]
        # )


if __name__ == "__main__":

    n = 1000

    x = np.random.uniform(0, 1, (n, 2))
    x[:, 0] = 0
    y = np.random.uniform(0, 1, (n, 2))
    y[:, 0] = 1

    coords = np.concatenate((x, y))

    t = FireFlyTracker(coords, max_distance=0.1)
