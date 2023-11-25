
import random
from maze import MAZE
from gen_maze import (
    draw_full_maze,
    make_maze_prim2,
    WALL,
    FREE
)

from astar import astar

from qlearn import QLearning


WIDTH, HEIGHT = 50, 50

new_maze, task = make_maze_prim2(WIDTH, HEIGHT, 'list')

start = task['start']
end = task['finish']

path = astar(new_maze, start, end)
if end not in path:
    path1 = astar(new_maze, end, start)
    path = set(path + path1)

ql = QLearning(WIDTH * HEIGHT, 10, 0.5, 0.5)
ql.train_astar(new_maze, 1000, 0.5)
draw_full_maze(new_maze, path, start, end)