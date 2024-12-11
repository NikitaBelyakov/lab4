import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def generate_random_points(
    n: int,
    x_range: Tuple[float, float] = (0, 100),
    y_range: Tuple[float, float] = (0, 100),
    z_range: Tuple[float, float] = (0, 100)
) -> np.ndarray:
    points: np.ndarray = np.random.uniform(low=[x_range[0], y_range[0],z_range[0]],
            high=[x_range[1], y_range[1], z_range[1]],
            size=(n, 3))

    return points

def generate_graph(points: np.ndarray, directed: bool = False) -> nx.Graph:
    graph = nx.DiGraph() if directed else nx.Graph()
    for i, (x, y, z) in enumerate(points):
        graph.add_node(i, pos=(x, y, z))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            graph.add_edge(i, j, weight=dist)
            if not directed:
                graph.add_edge(j, i, weight=dist)

    return graph

WEIGHT_PROP = "weight"

def update_pheromone(
    pheromone: np.ndarray,
    paths: List[List[int]],
    path_lengths: List[float],
    rho: float,
    Q: float
) -> None:
    pheromone *= (1 - rho)
    for path, path_length in zip(paths, path_lengths):
        for i in range(len(path) - 1):
            pheromone[path[i]][path[i + 1]] += Q / path_length

def select_next_node(
    graph: nx.Graph,
    current_node: int,
    visited: List[bool],
    pheromone: np.ndarray,
    alpha: float,
    beta: float
) -> Optional[int]:
    unvisited_nodes = [node for node in graph.nodes if not visited[node]]
    if not unvisited_nodes:
        return None

    probabilities = []
    for node in unvisited_nodes:
        pheromone_value = pheromone[current_node][node]
        distance = graph[current_node][node][WEIGHT_PROP]
        probability = (pheromone_value ** alpha) * ((1 / distance) ** beta)
        probabilities.append(probability)

    probabilities = np.array(probabilities) / sum(probabilities)
    next_node = np.random.choice(unvisited_nodes, p=probabilities)

    return next_node

def ant_colony_optimization(
    graph: nx.Graph,
    n_ants: int = 10,
    n_iterations: int = 100,
    alpha: float = 1,
    beta: float = 2,
    rho: float = 0.5,
    Q: float = 100
) -> Tuple[List[int], float]:
    n = len(graph.nodes)
    pheromone = np.ones((n, n))
    best_path = None
    best_path_length = float('inf')

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n
            current_node = random.choice(list(graph.nodes))
            path = [current_node]
            path_length = 0

            for _ in range(n - 1):
                visited[current_node] = True
                next_node = select_next_node(graph, current_node, visited, pheromone, alpha, beta)
                path.append(next_node)
                path_length += graph[current_node][next_node][WEIGHT_PROP]
                current_node = next_node
            path.append(path[0])
            path_length += graph[current_node][path[0]][WEIGHT_PROP]
            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        update_pheromone(pheromone, paths, path_lengths, rho, Q)

    return best_path, best_path_length


def visualize_path(points: np.ndarray, path: List[int]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o')
    path_points = points[path]
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], c='blue', marker='o')
    plt.show()

points = generate_random_points(20)
graph = generate_graph(points, directed=False)
best_path, best_path_length = ant_colony_optimization(graph)
visualize_path(points, best_path)

# import random
# import matplotlib.pyplot as plt
#
#
# def gen_coords(n):
#     c = []
#     for i in range(n):
#         x = random.uniform(-1000000, 1000000)
#         y = random.uniform(-1000000, 1000000)
#         z = random.uniform(-1000000, 1000000)
#         c.append((x, y, z))
#     return c
#
# def gen_roads(n):
#     roads = set()
#     for i in range(n):
#         num_connections = random.randint(1, 2)  # 1 или 2 дороги для каждой точки
#         connections = random.sample(range(n), num_connections)  # Выбираем случайные точки
#         for j in connections:
#             if i != j:  # Исключаем связь с самой собой
#                 road_type = random.choice(['one-way', 'two-way'])
#                 roads.add((i, j))
#                 if road_type == 'two-way':
#                     roads.add((j, i))
#     return list(roads)
#
# n = 100  # Количество точек
# coords = gen_coords(n)
# roads = gen_roads(n)
#
# # Визуализация
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# # Рисуем точки
# xs = [coord[0] for coord in coords]
# ys = [coord[1] for coord in coords]
# zs = [coord[2] for coord in coords]
#
# ax.scatter(xs, ys, zs, c='r', marker='o', label='Точки')
#
# # Рисуем дороги (линии между точками)
# for road in roads:
#     point1 = coords[road[0]]
#     point2 = coords[road[1]]
#     ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c='b', alpha=0.5)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.title(f"Точки: {n}")
# plt.show()