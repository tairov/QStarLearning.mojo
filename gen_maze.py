
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque

WALL = 1
FREE = 0


def plot_maze(maze):
    """Plot a visual representation of the maze."""
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()

# Define a function to draw the path in green


def draw_full_maze(maze, path, start, end):
    # Plot the maze with the green path
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    # Overlay the path on the maze using green color
    if path:
        for position in path:
            plt.scatter(position[1], position[0], c='green', s=10)
    plt.scatter(start[1], start[0], c='red', s=50)
    plt.scatter(end[1], end[0], c='red', s=50)
    plt.show()


# 1 is wall, 0 is path
def make_maze_prim2(w=10, h=10, format="str"):

    MAZE_TASK = {
        "start": (1e9, 1e9),
        "finish": (-1, -1)
    }

    grid = [[WALL for x in range(h)] for y in range(w)]

    def frontier(x, y):
        f = set()
        if x >= 0 and x < w and y >= 0 and y < h:
            if x > 1 and grid[x-2][y] == WALL:
                f.add((x-2, y))
            if x + 2 < w and grid[x+2][y] == WALL:
                f.add((x+2, y))
            if y > 1 and grid[x][y-2] == WALL:
                f.add((x, y-2))
            if y + 2 < h and grid[x][y+2] == WALL:
                f.add((x, y+2))

        return f

    def neighbours(x, y):
        n = set()
        if x >= 0 and x < w and y >= 0 and y < h:
            if x > 1 and grid[x-2][y] == FREE or x == 1:
                n.add((x-2, y))
            if x + 2 < w and grid[x+2][y] == FREE or x == w-2:
                n.add((x+2, y))
            if y > 1 and grid[x][y-2] == FREE or y == 1:
                n.add((x, y-2))
            if y + 2 < h and grid[x][y+2] == FREE or y == h-2:
                n.add((x, y+2))

        return n

    def connect(x1, y1, x2, y2):
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        if x1 < h and y1 < w and x1 >= 0 and y1 >= 0:
            grid[x1][y1] = FREE
            calc_task(x1, y1)
        grid[x][y] = FREE
        calc_task(x, y)

    def calc_task(x, y):
        if (x, y) < MAZE_TASK["start"]:
            MAZE_TASK["start"] = (x, y)
        if (x, y) > MAZE_TASK["finish"]:
            MAZE_TASK["finish"] = (x, y)

    def generate():
        s = set()
        x, y = (random.randint(0, w-1), random.randint(0, h-1))
        grid[x][y] = FREE
        calc_task(x, y)
        fs = frontier(x, y)
        for f in fs:
            s.add(f)
        while s:
            x, y = random.choice(tuple(s))
            s.remove((x, y))
            ns = neighbours(x, y)
            if ns:
                nx, ny = random.choice(tuple(ns))
                connect(x, y, nx, ny)
            fs = frontier(x, y)
            for f in fs:
                s.add(f)

    generate()

    if format == "str":
        maze_str = ""
        for row in grid:
            maze_str += ''.join(['*' if cell ==
                                FREE else '#' for cell in row]) + "\n"
        return maze_str
    else:
        return grid, MAZE_TASK
