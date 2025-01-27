import numpy as np

# Constante du nombre d'or
phi = (1 + np.sqrt(5)) / 2  # ≈ 1.61803
inv_phi = 1 / phi           # ≈ 0.61803

# Dictionnaire contenant tous les solides de Platon
solids = {
    # Solides de Platon
    'tetrahedron': {
        'vertices': np.array([
            [1, 1, 1],     # Sommet 0
            [-1, -1, 1],   # Sommet 1
            [-1, 1, -1],   # Sommet 2
            [1, -1, -1]    # Sommet 3
        ]),
        'edges': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        'faces': [[0, 2, 1], [2, 3, 1], [1, 3, 0], [0, 1, 3], [1, 2, 3], [3, 2, 0], [2, 1, 0], [0, 3, 2]]
    },
    'cube': {
        'vertices': np.array([
            [-1, -1, -1],  # Sommet 0
            [-1, -1,  1],  # Sommet 1
            [-1,  1, -1],  # Sommet 2
            [-1,  1,  1],  # Sommet 3
            [1, -1, -1],   # Sommet 4
            [1, -1,  1],   # Sommet 5
            [1,  1, -1],   # Sommet 6
            [1,  1,  1]    # Sommet 7
        ]),
        'edges': [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)],
        'faces': [[4, 5, 1, 0], [1, 5, 7, 3], [1, 3, 2, 0], [4, 6, 7, 5], [3, 7, 6, 2], [0, 1, 3, 2], [2, 3, 7, 6], [0, 2, 6, 4], [2, 6, 4, 0], [6, 7, 5, 4], [5, 7, 3, 1], [0, 4, 5, 1]]
    },
    'octahedron': {
        'vertices': np.array([
            [1, 0, 0],     # Sommet 0
            [-1, 0, 0],    # Sommet 1
            [0, 1, 0],     # Sommet 2
            [0, -1, 0],    # Sommet 3
            [0, 0, 1],     # Sommet 4
            [0, 0, -1]     # Sommet 5
        ]),
        'edges': [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
        'faces': [[0, 2, 4], [3, 5, 0], [5, 3, 1], [4, 3, 0], [1, 4, 2], [1, 3, 4], [3, 4, 1], [0, 4, 3], [1, 5, 3], [0, 5, 2], [2, 5, 1], [1, 2, 5], [5, 2, 0], [2, 4, 0], [0, 3, 5], [4, 2, 1]]
    },
    'dodecahedron': {
        'vertices': np.array([
            # Sommets (±1, ±1, ±1)
            [-1, -1, -1],  # 0
            [-1, -1, 1],   # 1
            [-1, 1, -1],   # 2
            [-1, 1, 1],    # 3
            [1, -1, -1],   # 4
            [1, -1, 1],    # 5
            [1, 1, -1],    # 6
            [1, 1, 1],     # 7
            # Sommets (0, ±inv_phi, ±phi)
            [0, -inv_phi, -phi],  # 8
            [0, -inv_phi, phi],   # 9
            [0, inv_phi, -phi],   # 10
            [0, inv_phi, phi],    # 11
            # Sommets (±inv_phi, ±phi, 0)
            [-inv_phi, -phi, 0],  # 12
            [-inv_phi, phi, 0],   # 13
            [inv_phi, -phi, 0],   # 14
            [inv_phi, phi, 0],    # 15
            # Sommets (±phi, 0, ±inv_phi)
            [-phi, 0, -inv_phi],  # 16
            [-phi, 0, inv_phi],   # 17
            [phi, 0, -inv_phi],   # 18
            [phi, 0, inv_phi]     # 19
        ]),
        'edges': [(0, 8), (0, 12), (0, 16), (1, 9), (1, 12), (1, 17), (2, 10), (2, 13), (2, 16), (3, 11), (3, 13), (3, 17), (4, 8), (4, 14), (4, 18), (5, 9), (5, 14), (5, 19), (6, 10), (6, 15), (6, 18), (7, 11), (7, 15), (7, 19), (8, 10), (9, 11), (12, 14), (13, 15), (16, 17), (18, 19)],
        'faces': [[6, 15, 7, 19, 18], [0, 8, 4, 14, 12], [19, 7, 11, 9, 5], [3, 11, 7, 15, 13], [0, 12, 1, 17, 16], [1, 12, 14, 5, 9], [18, 19, 5, 14, 4], [2, 13, 15, 6, 10], [16, 17, 3, 13, 2], [12, 1, 17, 16, 0], [11, 7, 15, 13, 3], [4, 18, 19, 5, 14], [5, 19, 7, 11, 9], [0, 16, 2, 10, 8], [15, 7, 19, 18, 6], [8, 4, 14, 12, 0], [8, 10, 6, 18, 4], [9, 11, 3, 17, 1], [2, 16, 17, 3, 13], [1, 9, 11, 3, 17], [16, 2, 10, 8, 0], [13, 15, 6, 10, 2], [12, 14, 5, 9, 1], [4, 8, 10, 6, 18]]
    },
    'icosahedron': {
        'vertices': np.array([
            # Sommets (0, ±1, ±phi)
            [0, -1, -phi], [0, -1, phi], [0, 1, -phi], [0, 1, phi],
            # Sommets (±1, ±phi, 0)
            [-1, -phi, 0], [-1, phi, 0], [1, -phi, 0], [1, phi, 0],
            # Sommets (±phi, 0, ±1)
            [-phi, 0, -1], [-phi, 0, 1], [phi, 0, -1], [phi, 0, 1]
        ]),
        'edges': [(0, 2), (0, 4), (0, 6), (0, 8), (0, 10), (1, 3), (1, 4), (1, 6), (1, 9), (1, 11), (2, 5), (2, 7), (2, 8), (2, 10), (3, 5), (3, 7), (3, 9), (3, 11), (4, 6), (4, 8), (4, 9), (5, 7), (5, 8), (5, 9), (6, 10), (6, 11), (7, 10), (7, 11), (8, 9), (10, 11)],
        'faces': [[0, 4, 8], [11, 10, 7], [3, 11, 7], [3, 5, 9], [2, 5, 7], [2, 7, 10], [2, 10, 0], [0, 8, 2], [1, 3, 9], [1, 9, 4], [0, 2, 10], [2, 8, 5], [0, 6, 4], [10, 6, 0], [1, 6, 11], [3, 7, 5], [11, 7, 3], [4, 8, 0], [11, 3, 1], [3, 9, 1], [6, 11, 1], [5, 9, 3], [1, 11, 3], [6, 4, 0], [4, 6, 1], [9, 8, 4], [7, 10, 2], [4, 9, 8], [5, 8, 9], [7, 5, 3], [8, 9, 5], [8, 5, 2], [5, 7, 2], [7, 11, 10], [1, 4, 6], [9, 4, 1], [6, 10, 11], [10, 11, 6], [0, 10, 6], [8, 2, 0]]
    },

    # Solides d'Archimèdes
    'truncated_tetrahedron':{
        'vertices' : [[-0.5222329678670935, -1.5666989036012806, 0.5222329678670935], [-1.5666989036012806, -0.5222329678670935, 0.5222329678670935], [-0.5222329678670935, 1.5666989036012806, -0.5222329678670935], [-0.5222329678670935, -0.5222329678670935, 1.5666989036012806], [1.5666989036012806, -0.5222329678670935, -0.5222329678670935], [0.5222329678670935, 1.5666989036012806, 0.5222329678670935], [0.5222329678670935, 0.5222329678670935, 1.5666989036012806], [-1.5666989036012806, 0.5222329678670935, -0.5222329678670935], [1.5666989036012806, 0.5222329678670935, 0.5222329678670935], [-0.5222329678670935, 0.5222329678670935, -1.5666989036012806], [0.5222329678670935, -0.5222329678670935, -1.5666989036012806], [0.5222329678670935, -1.5666989036012806, -0.5222329678670935]],
        'edges' : [(0, 1), (0, 3), (0, 11), (1, 3), (1, 7), (2, 5), (2, 7), (2, 9), (3, 6), (4, 8), (4, 10), (4, 11), (5, 6), (5, 8), (6, 8), (7, 9), (9, 10), (10, 11)],
        'faces' : [[9, 7, 2], [3, 6, 5, 2, 7, 1], [0, 3, 1], [6, 8, 5], [0, 1, 7, 9, 10, 11], [11, 4, 8, 6, 3, 0], [1, 7, 9, 10, 11, 0], [5, 6, 8], [3, 1, 0], [11, 10, 4], [2, 9, 7], [2, 5, 8, 4, 10, 9], [5, 8, 4, 10, 9, 2], [0, 11, 4, 8, 6, 3], [1, 3, 6, 5, 2, 7], [4, 11, 10]]
    },
    'icosidodecahedron':{
        'vertices' : [[-1.618033988749895, 0.0, 0.0], [-1.3090169943749475, -0.5, -0.8090169943749475], [-1.3090169943749475, -0.5, 0.8090169943749475], [-1.3090169943749475, 0.5, -0.8090169943749475], [-1.3090169943749475, 0.5, 0.8090169943749475], [-0.8090169943749475, -1.3090169943749475, -0.5], [-0.8090169943749475, -1.3090169943749475, 0.5], [-0.8090169943749475, 1.3090169943749475, -0.5], [-0.8090169943749475, 1.3090169943749475, 0.5], [-0.5, -0.8090169943749475, -1.3090169943749475], [-0.5, -0.8090169943749475, 1.3090169943749475], [-0.5, 0.8090169943749475, -1.3090169943749475], [-0.5, 0.8090169943749475, 1.3090169943749475], [0.0, -1.618033988749895, 0.0], [0.0, 0.0, -1.618033988749895], [0.0, 0.0, 1.618033988749895], [0.0, 1.618033988749895, 0.0], [0.5, -0.8090169943749475, -1.3090169943749475], [0.5, -0.8090169943749475, 1.3090169943749475], [0.5, 0.8090169943749475, -1.3090169943749475], [0.5, 0.8090169943749475, 1.3090169943749475], [0.8090169943749475, -1.3090169943749475, -0.5], [0.8090169943749475, -1.3090169943749475, 0.5], [0.8090169943749475, 1.3090169943749475, -0.5], [0.8090169943749475, 1.3090169943749475, 0.5], [1.3090169943749475, -0.5, -0.8090169943749475], [1.3090169943749475, -0.5, 0.8090169943749475], [1.3090169943749475, 0.5, -0.8090169943749475], [1.3090169943749475, 0.5, 0.8090169943749475], [1.618033988749895, 0.0, 0.0]],
        'edges' : [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 5), (1, 9), (2, 4), (2, 6), (2, 10), (3, 7), (3, 11), (4, 8), (4, 12), (5, 6), (5, 9), (5, 13), (6, 10), (6, 13), (7, 8), (7, 11), (7, 16), (8, 12), (8, 16), (9, 14), (9, 17), (10, 15), (10, 18), (11, 14), (11, 19), (12, 15), (12, 20), (13, 21), (13, 22), (14, 17), (14, 19), (15, 18), (15, 20), (16, 23), (16, 24), (17, 21), (17, 25), (18, 22), (18, 26), (19, 23), (19, 27), (20, 24), (20, 28), (21, 22), (21, 25), (22, 26), (23, 24), (23, 27), (24, 28), (25, 27), (25, 29), (26, 28), (26, 29), (27, 29), (28, 29)],
        'faces' : [[14, 19, 27, 25, 17], [1, 3, 11, 14, 9], [1, 9, 5], [7, 8, 16], [15, 18, 26, 28, 20], [12, 15, 20], [20, 28, 24], [0, 2, 4], [4, 12, 8], [8, 12, 20, 24, 16], [0, 1, 5, 6, 2], [6, 13, 22, 18, 10], [26, 29, 28], [3, 7, 11], [7, 16, 23, 19, 11], [2, 6, 10], [2, 10, 15, 12, 4], [13, 21, 22], [5, 9, 17, 21, 13], [21, 25, 29, 26, 22], [23, 24, 28, 29, 27], [5, 13, 6], [11, 19, 14], [0, 4, 8, 7, 3], [17, 25, 21], [0, 3, 1], [25, 27, 29], [10, 18, 15], [16, 24, 23], [9, 14, 17], [19, 23, 27], [18, 22, 26]]
    },
    'truncated_cuboctahedron':{
        'vertices' : [[0.3333333333333333, -0.8047378541243649, -1.2761423749153968], [0.8047378541243649, 1.2761423749153968, 0.3333333333333333], [0.8047378541243649, 1.2761423749153968, -0.3333333333333333], [0.8047378541243649, -0.3333333333333333, -1.2761423749153968], [0.8047378541243649, 0.3333333333333333, -1.2761423749153968], [0.3333333333333333, -0.8047378541243649, 1.2761423749153968], [0.8047378541243649, -1.2761423749153968, 0.3333333333333333], [1.2761423749153968, 0.8047378541243649, 0.3333333333333333], [-0.3333333333333333, 0.8047378541243649, -1.2761423749153968], [0.8047378541243649, -1.2761423749153968, -0.3333333333333333], [1.2761423749153968, 0.8047378541243649, -0.3333333333333333], [-0.8047378541243649, -0.3333333333333333, -1.2761423749153968], [-0.8047378541243649, 0.3333333333333333, -1.2761423749153968], [0.8047378541243649, -0.3333333333333333, 1.2761423749153968], [0.8047378541243649, 0.3333333333333333, 1.2761423749153968], [-0.8047378541243649, 1.2761423749153968, 0.3333333333333333], [-0.8047378541243649, 1.2761423749153968, -0.3333333333333333], [1.2761423749153968, -0.3333333333333333, -0.8047378541243649], [-1.2761423749153968, 0.8047378541243649, 0.3333333333333333], [1.2761423749153968, 0.3333333333333333, -0.8047378541243649], [-0.3333333333333333, 0.8047378541243649, 1.2761423749153968], [-1.2761423749153968, 0.8047378541243649, -0.3333333333333333], [-0.8047378541243649, -0.3333333333333333, 1.2761423749153968], [-0.8047378541243649, 0.3333333333333333, 1.2761423749153968], [-0.3333333333333333, -1.2761423749153968, -0.8047378541243649], [-0.8047378541243649, -1.2761423749153968, 0.3333333333333333], [-0.3333333333333333, 1.2761423749153968, -0.8047378541243649], [-0.8047378541243649, -1.2761423749153968, -0.3333333333333333], [1.2761423749153968, -0.3333333333333333, 0.8047378541243649], [1.2761423749153968, 0.3333333333333333, 0.8047378541243649], [-1.2761423749153968, -0.3333333333333333, -0.8047378541243649], [-1.2761423749153968, 0.3333333333333333, -0.8047378541243649], [-0.3333333333333333, -1.2761423749153968, 0.8047378541243649], [-0.3333333333333333, 1.2761423749153968, 0.8047378541243649], [0.3333333333333333, 1.2761423749153968, -0.8047378541243649], [1.2761423749153968, -0.8047378541243649, 0.3333333333333333], [-0.3333333333333333, -0.8047378541243649, -1.2761423749153968], [-1.2761423749153968, -0.3333333333333333, 0.8047378541243649], [0.3333333333333333, 0.8047378541243649, -1.2761423749153968], [-1.2761423749153968, 0.3333333333333333, 0.8047378541243649], [1.2761423749153968, -0.8047378541243649, -0.3333333333333333], [0.3333333333333333, 1.2761423749153968, 0.8047378541243649], [0.3333333333333333, -1.2761423749153968, -0.8047378541243649], [-1.2761423749153968, -0.8047378541243649, 0.3333333333333333], [-0.3333333333333333, -0.8047378541243649, 1.2761423749153968], [-1.2761423749153968, -0.8047378541243649, -0.3333333333333333], [0.3333333333333333, 0.8047378541243649, 1.2761423749153968], [0.3333333333333333, -1.2761423749153968, 0.8047378541243649]],
        'edges' : [(0, 3), (0, 36), (0, 42), (1, 2), (1, 7), (1, 41), (2, 10), (2, 34), (3, 4), (3, 17), (4, 19), (4, 38), (5, 13), (5, 44), (5, 47), (6, 9), (6, 35), (6, 47), (7, 10), (7, 29), (8, 12), (8, 26), (8, 38), (9, 40), (9, 42), (10, 19), (11, 12), (11, 30), (11, 36), (12, 31), (13, 14), (13, 28), (14, 29), (14, 46), (15, 16), (15, 18), (15, 33), (16, 21), (16, 26), (17, 19), (17, 40), (18, 21), (18, 39), (20, 23), (20, 33), (20, 46), (21, 31), (22, 23), (22, 37), (22, 44), (23, 39), (24, 27), (24, 36), (24, 42), (25, 27), (25, 32), (25, 43), (26, 34), (27, 45), (28, 29), (28, 35), (30, 31), (30, 45), (32, 44), (32, 47), (33, 41), (34, 38), (35, 40), (37, 39), (37, 43), (41, 46), (43, 45)],
        'faces' : [[7, 29, 28, 35, 40, 17, 19, 10], [5, 13, 14, 46, 20, 23, 22, 44], [18, 21, 31, 30, 45, 43, 37, 39], [0, 36, 11, 12, 8, 38, 4, 3], [1, 2, 34, 26, 16, 15, 33, 41], [6, 47, 32, 25, 27, 24, 42, 9], [8, 26, 34, 38], [3, 4, 19, 17], [13, 28, 29, 14], [22, 23, 39, 37], [1, 7, 10, 2], [11, 30, 31, 12], [0, 42, 24, 36], [25, 43, 45, 27], [6, 9, 40, 35], [20, 46, 41, 33], [5, 44, 32, 47], [15, 16, 21, 18], [1, 41, 46, 14, 29, 7], [0, 3, 17, 40, 9, 42], [2, 10, 19, 4, 38, 34], [5, 47, 6, 35, 28, 13], [22, 37, 43, 25, 32, 44], [11, 36, 24, 27, 45, 30], [8, 12, 31, 21, 16, 26], [15, 18, 39, 23, 20, 33]]
   },
   'truncated_dodecahedron':{
        'vertices' : [[0.3090169943749474, 0.8090169943749475, -1.618033988749895], [-0.8090169943749475, 1.0, -1.3090169943749475], [0.3090169943749474, 1.8090169943749475, -0.0], [0.3090169943749474, 0.8090169943749475, 1.618033988749895], [1.8090169943749475, -0.0, 0.3090169943749474], [-1.3090169943749475, -0.8090169943749475, -1.0], [0.8090169943749475, 1.0, 1.3090169943749475], [-1.3090169943749475, -0.8090169943749475, 1.0], [-1.618033988749895, 0.3090169943749474, 0.8090169943749475], [-0.0, -0.3090169943749474, -1.8090169943749475], [0.8090169943749475, -1.0, -1.3090169943749475], [-0.8090169943749475, 1.0, 1.3090169943749475], [-1.618033988749895, 0.3090169943749474, -0.8090169943749475], [-1.8090169943749475, -0.0, 0.3090169943749474], [1.3090169943749475, -0.8090169943749475, -1.0], [0.8090169943749475, 1.618033988749895, 0.3090169943749474], [1.8090169943749475, -0.0, -0.3090169943749474], [1.618033988749895, -0.3090169943749474, 0.8090169943749475], [0.8090169943749475, -1.618033988749895, 0.3090169943749474], [-0.8090169943749475, -1.0, -1.3090169943749475], [1.3090169943749475, -0.8090169943749475, 1.0], [-0.3090169943749474, -0.8090169943749475, -1.618033988749895], [-0.0, 0.3090169943749474, -1.8090169943749475], [-0.3090169943749474, -1.8090169943749475, -0.0], [-0.0, -0.3090169943749474, 1.8090169943749475], [1.618033988749895, -0.3090169943749474, -0.8090169943749475], [-0.8090169943749475, 1.618033988749895, 0.3090169943749474], [-0.3090169943749474, -0.8090169943749475, 1.618033988749895], [0.8090169943749475, -1.0, 1.3090169943749475], [-1.3090169943749475, 0.8090169943749475, -1.0], [-1.3090169943749475, 0.8090169943749475, 1.0], [-1.8090169943749475, -0.0, -0.3090169943749474], [1.618033988749895, 0.3090169943749474, 0.8090169943749475], [1.0, -1.3090169943749475, 0.8090169943749475], [0.8090169943749475, 1.618033988749895, -0.3090169943749474], [0.8090169943749475, -1.618033988749895, -0.3090169943749474], [-0.8090169943749475, -1.618033988749895, 0.3090169943749474], [-0.8090169943749475, -1.0, 1.3090169943749475], [1.3090169943749475, 0.8090169943749475, -1.0], [-0.0, 0.3090169943749474, 1.8090169943749475], [1.618033988749895, 0.3090169943749474, -0.8090169943749475], [1.0, 1.3090169943749475, 0.8090169943749475], [1.3090169943749475, 0.8090169943749475, 1.0], [1.0, -1.3090169943749475, -0.8090169943749475], [-1.618033988749895, -0.3090169943749474, 0.8090169943749475], [-0.8090169943749475, 1.618033988749895, -0.3090169943749474], [-1.0, 1.3090169943749475, 0.8090169943749475], [0.3090169943749474, -0.8090169943749475, -1.618033988749895], [0.3090169943749474, -1.8090169943749475, -0.0], [1.0, 1.3090169943749475, -0.8090169943749475], [-1.0, -1.3090169943749475, 0.8090169943749475], [0.3090169943749474, -0.8090169943749475, 1.618033988749895], [-0.3090169943749474, 0.8090169943749475, -1.618033988749895], [-1.618033988749895, -0.3090169943749474, -0.8090169943749475], [-0.8090169943749475, -1.618033988749895, -0.3090169943749474], [-1.0, 1.3090169943749475, -0.8090169943749475], [-0.3090169943749474, 1.8090169943749475, -0.0], [-0.3090169943749474, 0.8090169943749475, 1.618033988749895], [0.8090169943749475, 1.0, -1.3090169943749475], [-1.0, -1.3090169943749475, -0.8090169943749475]],
        'edges' : [(0, 22), (0, 52), (0, 58), (1, 29), (1, 52), (1, 55), (2, 15), (2, 34), (2, 56), (3, 6), (3, 39), (3, 57), (4, 16), (4, 17), (4, 32), (5, 19), (5, 53), (5, 59), (6, 41), (6, 42), (7, 37), (7, 44), (7, 50), (8, 13), (8, 30), (8, 44), (9, 21), (9, 22), (9, 47), (10, 14), (10, 43), (10, 47), (11, 30), (11, 46), (11, 57), (12, 29), (12, 31), (12, 53), (13, 31), (13, 44), (14, 25), (14, 43), (15, 34), (15, 41), (16, 25), (16, 40), (17, 20), (17, 32), (18, 33), (18, 35), (18, 48), (19, 21), (19, 59), (20, 28), (20, 33), (21, 47), (22, 52), (23, 36), (23, 48), (23, 54), (24, 27), (24, 39), (24, 51), (25, 40), (26, 45), (26, 46), (26, 56), (27, 37), (27, 51), (28, 33), (28, 51), (29, 55), (30, 46), (31, 53), (32, 42), (34, 49), (35, 43), (35, 48), (36, 50), (36, 54), (37, 50), (38, 40), (38, 49), (38, 58), (39, 57), (41, 42), (45, 55), (45, 56), (49, 58), (54, 59)],
        'faces' : [[24, 27, 51], [0, 52, 1, 55, 45, 56, 2, 34, 49, 58], [3, 57, 39], [4, 17, 20, 33, 18, 35, 43, 14, 25, 16], [5, 59, 54, 36, 50, 7, 44, 13, 31, 53], [7, 37, 27, 24, 39, 57, 11, 30, 8, 44], [6, 42, 41], [26, 56, 45], [9, 47, 21], [2, 15, 34], [11, 46, 30], [4, 32, 17], [10, 43, 35, 48, 23, 54, 59, 19, 21, 47], [7, 50, 37], [3, 39, 24, 51, 28, 20, 17, 32, 42, 6], [4, 16, 40, 38, 49, 34, 15, 41, 42, 32], [18, 48, 35], [0, 58, 38, 40, 25, 14, 10, 47, 9, 22], [18, 33, 28, 51, 27, 37, 50, 36, 23, 48], [16, 25, 40], [20, 28, 33], [8, 13, 44], [1, 29, 55], [38, 58, 49], [12, 53, 31], [5, 19, 59], [1, 52, 22, 9, 21, 19, 5, 53, 12, 29], [2, 56, 26, 46, 11, 57, 3, 6, 41, 15], [8, 30, 46, 26, 45, 55, 29, 12, 31, 13], [23, 36, 54], [10, 14, 43], [0, 22, 52]]
    },
    'truncated_icosidodecahedron':{
        'vertices' : [[1.5393446629166316, 0.20601132958329826, 0.20601132958329826], [-0.20601132958329826, -0.20601132958329826, -1.5393446629166316], [0.20601132958329826, -0.20601132958329826, -1.5393446629166316], [-0.7453559924999299, -0.6666666666666666, 1.2060113295832984], [-0.7453559924999299, 0.6666666666666666, 1.2060113295832984], [0.872677996249965, -1.2847006554165616, 0.20601132958329826], [-0.20601132958329826, -0.872677996249965, 1.2847006554165616], [-0.6666666666666666, -1.2060113295832984, -0.7453559924999299], [-0.20601132958329826, 0.872677996249965, -1.2847006554165616], [1.4120226591665965, -0.4120226591665965, -0.5393446629166316], [-1.0, 1.0786893258332633, 0.5393446629166316], [-0.4120226591665965, -0.5393446629166316, -1.4120226591665965], [-1.5393446629166316, 0.20601132958329826, -0.20601132958329826], [-0.20601132958329826, 0.20601132958329826, 1.5393446629166316], [-0.7453559924999299, -0.6666666666666666, -1.2060113295832984], [-0.7453559924999299, 0.6666666666666666, -1.2060113295832984], [-1.2847006554165616, -0.20601132958329826, -0.872677996249965], [0.5393446629166316, 1.0, 1.0786893258332633], [-1.0, -1.0786893258332633, -0.5393446629166316], [-0.5393446629166316, 1.4120226591665965, -0.4120226591665965], [1.2060113295832984, -0.7453559924999299, 0.6666666666666666], [-1.4120226591665965, -0.4120226591665965, 0.5393446629166316], [-1.2060113295832984, 0.7453559924999299, 0.6666666666666666], [-1.2847006554165616, 0.20601132958329826, 0.872677996249965], [1.2060113295832984, -0.7453559924999299, -0.6666666666666666], [-1.2060113295832984, 0.7453559924999299, -0.6666666666666666], [-0.20601132958329826, -1.5393446629166316, -0.20601132958329826], [-0.20601132958329826, 0.20601132958329826, -1.5393446629166316], [-0.872677996249965, -1.2847006554165616, -0.20601132958329826], [1.2847006554165616, -0.20601132958329826, -0.872677996249965], [1.5393446629166316, 0.20601132958329826, -0.20601132958329826], [1.4120226591665965, 0.4120226591665965, -0.5393446629166316], [0.5393446629166316, -1.4120226591665965, 0.4120226591665965], [-0.5393446629166316, -1.4120226591665965, -0.4120226591665965], [0.872677996249965, -1.2847006554165616, -0.20601132958329826], [0.20601132958329826, -1.5393446629166316, 0.20601132958329826], [1.0786893258332633, 0.5393446629166316, -1.0], [0.6666666666666666, -1.2060113295832984, -0.7453559924999299], [-1.0, 1.0786893258332633, -0.5393446629166316], [0.5393446629166316, -1.0, -1.0786893258332633], [1.5393446629166316, -0.20601132958329826, 0.20601132958329826], [-0.5393446629166316, 1.0, -1.0786893258332633], [0.4120226591665965, 0.5393446629166316, 1.4120226591665965], [0.6666666666666666, 1.2060113295832984, -0.7453559924999299], [-1.4120226591665965, 0.4120226591665965, 0.5393446629166316], [0.20601132958329826, -0.872677996249965, 1.2847006554165616], [0.20601132958329826, 0.872677996249965, -1.2847006554165616], [-1.2847006554165616, 0.20601132958329826, -0.872677996249965], [-1.0786893258332633, -0.5393446629166316, 1.0], [-1.4120226591665965, -0.4120226591665965, -0.5393446629166316], [-0.20601132958329826, 1.5393446629166316, -0.20601132958329826], [-0.20601132958329826, 0.872677996249965, 1.2847006554165616], [-0.5393446629166316, -1.0, 1.0786893258332633], [0.20601132958329826, 0.20601132958329826, 1.5393446629166316], [1.0, -1.0786893258332633, 0.5393446629166316], [0.20601132958329826, 1.5393446629166316, -0.20601132958329826], [-0.872677996249965, 1.2847006554165616, 0.20601132958329826], [0.5393446629166316, 1.4120226591665965, 0.4120226591665965], [0.4120226591665965, -0.5393446629166316, 1.4120226591665965], [-0.6666666666666666, 1.2060113295832984, -0.7453559924999299], [0.20601132958329826, -1.5393446629166316, -0.20601132958329826], [-1.5393446629166316, -0.20601132958329826, 0.20601132958329826], [0.20601132958329826, 0.20601132958329826, -1.5393446629166316], [0.872677996249965, 1.2847006554165616, 0.20601132958329826], [-1.0786893258332633, 0.5393446629166316, 1.0], [1.5393446629166316, -0.20601132958329826, -0.20601132958329826], [1.2847006554165616, 0.20601132958329826, 0.872677996249965], [-0.4120226591665965, -0.5393446629166316, 1.4120226591665965], [-1.4120226591665965, 0.4120226591665965, -0.5393446629166316], [1.0, 1.0786893258332633, 0.5393446629166316], [0.7453559924999299, -0.6666666666666666, 1.2060113295832984], [1.0, -1.0786893258332633, -0.5393446629166316], [0.7453559924999299, 0.6666666666666666, 1.2060113295832984], [1.0786893258332633, -0.5393446629166316, 1.0], [0.6666666666666666, -1.2060113295832984, 0.7453559924999299], [1.4120226591665965, -0.4120226591665965, 0.5393446629166316], [0.4120226591665965, 0.5393446629166316, -1.4120226591665965], [-0.872677996249965, 1.2847006554165616, -0.20601132958329826], [0.5393446629166316, -1.4120226591665965, -0.4120226591665965], [0.20601132958329826, 0.872677996249965, 1.2847006554165616], [0.7453559924999299, -0.6666666666666666, -1.2060113295832984], [-0.4120226591665965, 0.5393446629166316, 1.4120226591665965], [0.7453559924999299, 0.6666666666666666, -1.2060113295832984], [-1.5393446629166316, -0.20601132958329826, -0.20601132958329826], [0.6666666666666666, 1.2060113295832984, 0.7453559924999299], [0.872677996249965, 1.2847006554165616, -0.20601132958329826], [-0.6666666666666666, -1.2060113295832984, 0.7453559924999299], [0.5393446629166316, 1.0, -1.0786893258332633], [-1.0786893258332633, 0.5393446629166316, -1.0], [0.5393446629166316, -1.0, 1.0786893258332633], [0.4120226591665965, -0.5393446629166316, -1.4120226591665965], [1.0786893258332633, 0.5393446629166316, 1.0], [-0.5393446629166316, 1.0, 1.0786893258332633], [-0.20601132958329826, -0.872677996249965, -1.2847006554165616], [-0.5393446629166316, 1.4120226591665965, 0.4120226591665965], [0.20601132958329826, -0.872677996249965, -1.2847006554165616], [1.4120226591665965, 0.4120226591665965, 0.5393446629166316], [1.2847006554165616, -0.20601132958329826, 0.872677996249965], [-0.6666666666666666, 1.2060113295832984, 0.7453559924999299], [1.2060113295832984, 0.7453559924999299, 0.6666666666666666], [1.2060113295832984, 0.7453559924999299, -0.6666666666666666], [0.5393446629166316, 1.4120226591665965, -0.4120226591665965], [1.0786893258332633, -0.5393446629166316, -1.0], [-0.5393446629166316, -1.4120226591665965, 0.4120226591665965], [-1.0786893258332633, -0.5393446629166316, -1.0], [-1.2060113295832984, -0.7453559924999299, 0.6666666666666666], [-1.2060113295832984, -0.7453559924999299, -0.6666666666666666], [-1.5393446629166316, 0.20601132958329826, 0.20601132958329826], [-0.20601132958329826, 1.5393446629166316, 0.20601132958329826], [-0.5393446629166316, -1.0, -1.0786893258332633], [-0.20601132958329826, -0.20601132958329826, 1.5393446629166316], [0.20601132958329826, -0.20601132958329826, 1.5393446629166316], [1.2847006554165616, 0.20601132958329826, -0.872677996249965], [0.20601132958329826, 1.5393446629166316, 0.20601132958329826], [-1.0, -1.0786893258332633, 0.5393446629166316], [-1.2847006554165616, -0.20601132958329826, 0.872677996249965], [-0.4120226591665965, 0.5393446629166316, -1.4120226591665965], [1.0, 1.0786893258332633, -0.5393446629166316], [-0.20601132958329826, -1.5393446629166316, 0.20601132958329826], [-0.872677996249965, -1.2847006554165616, 0.20601132958329826]],
        'edges' : [(0, 30), (0, 40), (0, 96), (1, 2), (1, 11), (1, 27), (2, 62), (2, 90), (3, 48), (3, 52), (3, 67), (4, 64), (4, 81), (4, 92), (5, 32), (5, 34), (5, 54), (6, 45), (6, 52), (6, 67), (7, 18), (7, 33), (7, 109), (8, 41), (8, 46), (8, 116), (9, 24), (9, 29), (9, 65), (10, 22), (10, 56), (10, 98), (11, 14), (11, 93), (12, 68), (12, 83), (12, 107), (13, 53), (13, 81), (13, 110), (14, 104), (14, 109), (15, 41), (15, 88), (15, 116), (16, 47), (16, 49), (16, 104), (17, 72), (17, 79), (17, 84), (18, 28), (18, 106), (19, 50), (19, 59), (19, 77), (20, 54), (20, 73), (20, 75), (21, 61), (21, 105), (21, 115), (22, 44), (22, 64), (23, 44), (23, 64), (23, 115), (24, 71), (24, 102), (25, 38), (25, 68), (25, 88), (26, 33), (26, 60), (26, 118), (27, 62), (27, 116), (28, 33), (28, 119), (29, 102), (29, 112), (30, 31), (30, 65), (31, 100), (31, 112), (32, 35), (32, 74), (34, 71), (34, 78), (35, 60), (35, 118), (36, 82), (36, 100), (36, 112), (37, 39), (37, 71), (37, 78), (38, 59), (38, 77), (39, 80), (39, 95), (40, 65), (40, 75), (41, 59), (42, 53), (42, 72), (42, 79), (43, 87), (43, 101), (43, 117), (44, 107), (45, 58), (45, 89), (46, 76), (46, 87), (47, 68), (47, 88), (48, 105), (48, 115), (49, 83), (49, 106), (50, 55), (50, 108), (51, 79), (51, 81), (51, 92), (52, 86), (53, 111), (54, 74), (55, 101), (55, 113), (56, 77), (56, 94), (57, 63), (57, 84), (57, 113), (58, 70), (58, 111), (60, 78), (61, 83), (61, 107), (62, 76), (63, 69), (63, 85), (66, 91), (66, 96), (66, 97), (67, 110), (69, 84), (69, 99), (70, 73), (70, 89), (72, 91), (73, 97), (74, 89), (75, 97), (76, 82), (80, 90), (80, 102), (82, 87), (85, 101), (85, 117), (86, 103), (86, 114), (90, 95), (91, 99), (92, 98), (93, 95), (93, 109), (94, 98), (94, 108), (96, 99), (100, 117), (103, 118), (103, 119), (104, 106), (105, 114), (108, 113), (110, 111), (114, 119)],
        'faces' : [[7, 18, 106, 104, 14, 109], [18, 28, 119, 114, 105, 21, 61, 83, 49, 106], [16, 104, 106, 49], [10, 56, 77, 38, 25, 68, 12, 107, 44, 22], [46, 87, 82, 76], [10, 98, 94, 56], [36, 82, 87, 43, 117, 100], [15, 88, 25, 38, 59, 41], [19, 77, 56, 94, 108, 50], [19, 59, 38, 77], [6, 52, 86, 103, 118, 35, 32, 74, 89, 45], [43, 101, 85, 117], [9, 24, 102, 29], [0, 30, 31, 100, 117, 85, 63, 69, 99, 96], [39, 95, 90, 80], [22, 44, 23, 64], [31, 112, 36, 100], [8, 116, 15, 41], [17, 84, 57, 113, 108, 94, 98, 92, 51, 79], [20, 73, 70, 89, 74, 54], [7, 33, 28, 18], [26, 118, 103, 119, 28, 33], [11, 93, 109, 14], [3, 48, 105, 114, 86, 52], [26, 60, 35, 118], [55, 113, 57, 63, 85, 101], [24, 71, 37, 39, 80, 102], [42, 53, 111, 58, 70, 73, 97, 66, 91, 72], [5, 32, 35, 60, 78, 34], [6, 45, 58, 111, 110, 67], [5, 54, 74, 32], [45, 89, 70, 58], [12, 83, 61, 107], [57, 84, 69, 63], [0, 40, 65, 30], [13, 53, 42, 79, 51, 81], [34, 78, 37, 71], [8, 41, 59, 19, 50, 55, 101, 43, 87, 46], [2, 62, 76, 82, 36, 112, 29, 102, 80, 90], [50, 108, 113, 55], [5, 34, 71, 24, 9, 65, 40, 75, 20, 54], [13, 110, 111, 53], [3, 52, 6, 67], [9, 29, 112, 31, 30, 65], [3, 67, 110, 13, 81, 4, 64, 23, 115, 48], [21, 115, 23, 44, 107, 61], [4, 92, 98, 10, 22, 64], [12, 68, 47, 16, 49, 83], [7, 109, 93, 95, 39, 37, 78, 60, 26, 33], [1, 11, 14, 104, 16, 47, 88, 15, 116, 27], [17, 79, 42, 72], [1, 27, 62, 2], [8, 46, 76, 62, 27, 116], [21, 105, 48, 115], [17, 72, 91, 99, 69, 84], [20, 75, 97, 73], [1, 2, 90, 95, 93, 11], [25, 88, 47, 68], [86, 114, 119, 103], [66, 96, 99, 91], [4, 81, 51, 92], [0, 96, 66, 97, 75, 40]]
    }
}
