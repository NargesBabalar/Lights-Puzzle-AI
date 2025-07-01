import random
import heapq
import collections
import numpy as np
from typing import List, Tuple, Callable
from time import time
from functools import partial
from copy import deepcopy
from tabulate import tabulate
import signal

# LightsOutPuzzle class
class LightsOutPuzzle:
    def __init__(self, board: List[List[int]]):
        self.board = np.array(board, dtype=np.uint8)
        self.size = len(board)

    def toggle(self, x, y):
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.board[nx, ny] ^= 1

    def is_solved(self):
        return np.all(self.board == 0)

    def get_moves(self):
        return [(x, y) for x in range(self.size) for y in range(self.size)]

    def __str__(self):
        return '\n'.join(' '.join(str(cell) for cell in row) for row in self.board)

# Heuristics
# Heuristic 1: Count of lights still on
def heuristic_count_on(puzzle: LightsOutPuzzle) -> int:
    return np.sum(puzzle.board)

# Heuristic 2: Manhattan distance to turn off all lights
def heuristic_manhattan(puzzle: LightsOutPuzzle) -> int:
    count = 0
    for x in range(puzzle.size):
        for y in range(puzzle.size):
            if puzzle.board[x, y] == 1:
                count += (x + y)
    return count

# BFS Algorithm
def bfs_solve(puzzle: LightsOutPuzzle):
    queue = collections.deque([(puzzle.board.copy(), [])])
    visited = set()
    nodes_visited = 0

    while queue:
        board, path = queue.popleft()
        nodes_visited += 1
        puzzle.board = board

        if puzzle.is_solved():
            return path, nodes_visited

        for move in puzzle.get_moves():
            puzzle.board = board.copy()
            puzzle.toggle(*move)
            new_board = puzzle.board.copy()
            if str(new_board.tolist()) not in visited:
                visited.add(str(new_board.tolist()))
                queue.append((new_board, path + [move]))

    return None, nodes_visited

# IDS Algorithm
def ids_solve(puzzle: LightsOutPuzzle):
    def dls(board, depth, path, visited):
        if puzzle.is_solved():
            return path
        if depth == 0:
            return None

        for move in puzzle.get_moves():
            puzzle.board = board.copy()
            puzzle.toggle(*move)
            new_board = puzzle.board.copy()
            if str(new_board.tolist()) not in visited:
                visited.add(str(new_board.tolist()))
                result = dls(new_board, depth - 1, path + [move], visited)
                if result is not None:
                    return result
        return None

    nodes_visited = 0
    for depth in range(1, 100):
        visited = set()
        result = dls(puzzle.board.copy(), depth, [], visited)
        nodes_visited += len(visited)
        if result is not None:
            return result, nodes_visited

    return None, nodes_visited

# A* Algorithm
def astar_solve(puzzle: LightsOutPuzzle, heuristic: Callable[[LightsOutPuzzle], int]):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(puzzle), 0, puzzle.board.copy(), []))
    visited = set()
    nodes_visited = 0

    while open_list:
        _, cost, board, path = heapq.heappop(open_list)
        nodes_visited += 1
        puzzle.board = board

        if puzzle.is_solved():
            return path, nodes_visited

        for move in puzzle.get_moves():
            puzzle.board = board.copy()
            puzzle.toggle(*move)
            new_board = puzzle.board.copy()
            board_tuple = tuple(map(tuple, new_board))
            if board_tuple not in visited:
                visited.add(board_tuple)
                new_cost = cost + 1
                heapq.heappush(open_list, (new_cost + heuristic(puzzle), new_cost, new_board, path + [move]))

    return None, nodes_visited

# Weighted A* Algorithm
def weighted_astar_solve(puzzle: LightsOutPuzzle, heuristic: Callable[[LightsOutPuzzle], int], alpha: float):
    open_list = []
    heapq.heappush(open_list, (alpha * heuristic(puzzle), 0, puzzle.board.copy(), []))
    visited = set()
    nodes_visited = 0

    while open_list:
        _, cost, board, path = heapq.heappop(open_list)
        nodes_visited += 1
        puzzle.board = board

        if puzzle.is_solved():
            return path, nodes_visited

        for move in puzzle.get_moves():
            puzzle.board = board.copy()
            puzzle.toggle(*move)
            new_board = puzzle.board.copy()
            board_tuple = tuple(map(tuple, new_board))
            if board_tuple not in visited:
                visited.add(board_tuple)
                new_cost = cost + 1
                heapq.heappush(open_list, (new_cost + alpha * heuristic(puzzle), new_cost, new_board, path + [move]))

    return None, nodes_visited

# Function to run tests with time limits
def run_with_time_limit(func, args, time_limit):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    try:
        return func(*args)
    except TimeoutError:
        return None, "Time Limit Exceeded"
    finally:
        signal.alarm(0)

# Running tests and collecting results
def run_tests():
    sizes = [3, 4, 5]
    heuristics = [heuristic_count_on, heuristic_manhattan]
    weights = [1.5, 2.0]
    algorithms = [
        ("BFS", bfs_solve, 5),
        ("IDS", ids_solve, 5),
        ("A* (Count On)", partial(astar_solve, heuristic=heuristic_count_on), 120),
        ("A* (Manhattan)", partial(astar_solve, heuristic=heuristic_manhattan), 120),
    ]
    for weight in weights:
        algorithms.append((f"Weighted A* (Count On, alpha={weight})", partial(weighted_astar_solve, heuristic=heuristic_count_on, alpha=weight), 120))
        algorithms.append((f"Weighted A* (Manhattan, alpha={weight})", partial(weighted_astar_solve, heuristic=heuristic_manhattan, alpha=weight), 120))

    results = []
    for size in sizes:
        board = create_random_board(size, seed=42)
        puzzle = LightsOutPuzzle(board)
        for name, algorithm, time_limit in algorithms:
            result, nodes_visited = run_with_time_limit(algorithm, (deepcopy(puzzle),), time_limit)
            results.append([f"{size}x{size}", name, result if result != "Time Limit Exceeded" else "TLE", nodes_visited])

    headers = ["Puzzle Size", "Algorithm", "Solution", "Nodes Visited"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

# Utility function to create a random board
def create_random_board(size: int, seed: int = None, num_toggles: int = None):
    random.seed(time() if seed is None else seed)
    board = [[0 for _ in range(size)] for _ in range(size)]
    puzzle = LightsOutPuzzle(board)
    if num_toggles is None:
        num_toggles = random.randint(1, size * size)
    for _ in range(num_toggles):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        puzzle.toggle(x, y)
    return puzzle.board

# Run the tests
if __name__ == "__main__":
    run_tests()