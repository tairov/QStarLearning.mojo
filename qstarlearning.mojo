from python import Python

from memory import memset_zero
from algorithm import vectorize, parallelize
from math import trunc, mod
from random import rand
from mmatrix import Matrix
from optimum import (Node, astar)
from qlearn import QEnvironment, QLearning



alias WIDTH = 11
alias HEIGHT = 11
alias FREE = 0
alias WALL = 1

alias Q_REWARD = 100
alias Q_WALL_PENALTY = -100
alias Q_FREE_PENALTY = -1
alias Q_TERMINAL_STATE_VALUE = Q_FREE_PENALTY
alias Q_STEPS = 1000

alias nelts = 16 * simdwidthof[DType.int8]()


fn main() raises:
    Python.add_to_path(".")
    var t: Matrix[DType.int8] = Matrix[DType.int8](WIDTH, HEIGHT)
    # var t = TensorI8(WIDTH, HEIGHT)
    t.fill(Q_WALL_PENALTY)
    t.print()

    let ret: PythonObject
    let pth: PythonObject

    var start_x = 0
    var start_y = 0
    var end_x = 0
    var end_y = 0

    try:
        let gm = Python.import_module("gen_maze")
        ret = gm.make_maze_prim2(WIDTH, HEIGHT, 'list')
        start_x = atol(ret[1]['start'][0].to_string())
        start_y = atol(ret[1]['start'][1].to_string())

        end_x = atol(ret[1]['finish'][0].to_string())
        end_y = atol(ret[1]['finish'][1].to_string())
        
        # pth = astr.astar(ret[0], start, end)
    except e:
        # next release
        print("Failed to import:", e)
        return

    print('start', start_x, ':', start_y, ' | finish', end_x, ':', end_y)

    t[end_x, end_y] = Q_REWARD

    for i in range(WIDTH):
        for j in range(HEIGHT):
            let state = atol(ret[0][i][j].to_string())
            # set reward -1 for free cells
            if state == FREE:
                t[i, j] = Q_FREE_PENALTY
    let qenv = QEnvironment(t, Q_TERMINAL_STATE_VALUE)
    var qlearn = QLearning(qenv, Q_STEPS)
    qlearn.train()
    qlearn.print_results()
    


