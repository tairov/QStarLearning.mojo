from python import Python

from memory import memset_zero
from algorithm import vectorize, parallelize
from math import trunc, mod
from random import rand, seed
from mmatrix import Matrix

struct QEnvironment:
    var grid: Matrix[DType.int8]
    var rows: Int
    var columns: Int
    var terminal_reward: Int

    fn __init__(inout self, owned grid: Matrix[DType.int8], terminal_reward: Int):
        self.grid = grid
        self.rows = grid.dim0
        self.columns = grid.dim1
        self.terminal_reward = terminal_reward

    fn get_starting_location(self) -> StaticIntTuple[2]:
        var ret = StaticIntTuple[2](0, 0)
        let r = rand[DType.uint16](2)
        ret[0] = r[0].to_int() % self.rows
        ret[1] = r[1].to_int() % self.columns
        var iters = 0
        while self.is_terminal_state(ret[0], ret[1]):
            let r = rand[DType.uint16](2)
            ret[0] = r[0].to_int() % self.rows
            ret[1] = r[1].to_int() % self.columns
            iters += 1
            if iters > 100:
                print("Could not find a starting location that is not a terminal state.")
                break
        return ret



    fn get_next_location(self, row_index: Int, column_index: Int, action: Int) -> StaticIntTuple[2]:
        var ret = StaticIntTuple[2](0, 0)
        if action == 0:
            ret[0] = row_index - 1
            ret[1] = column_index
        elif action == 1:
            ret[0] = row_index + 1
            ret[1] = column_index
        elif action == 2:
            ret[0] = row_index
            ret[1] = column_index - 1
        elif action == 3:
            ret[0] = row_index
            ret[1] = column_index + 1
        else:
            ret[0] = row_index
            ret[1] = column_index
        return ret

    fn is_terminal_state(self, row_index: Int, column_index: Int) -> Bool:
        return self.grid[row_index, column_index] == self.terminal_reward

    fn get_reward(self, row_index: Int, column_index: Int) -> Int:
        return self.grid[row_index, column_index].to_int()


struct QLearning:
    var env: QEnvironment
    var actions: VariadicList[Int]
    var q_table: Matrix[DType.float32]
    var epsilon: Float16
    var discount_factor: Float32
    var learning_rate: Float32
    var episodes: Int

    fn __init__(inout self, env: QEnvironment, steps: Int):
        seed()
        # move env into self
        self.env = QEnvironment(env.grid, env.terminal_reward)

        self.actions = VariadicList[Int](0, 1, 2, 3)
        self.q_table = Matrix[DType.float32](self.env.rows, self.env.columns, len(self.actions))
        self.q_table.fill(0.0)
        self.epsilon = 0.9
        self.discount_factor = 0.9
        self.learning_rate = 0.9

        self.episodes = steps

    @always_inline
    fn argmax(self, cur_row: Int, cur_col: Int) -> Int:
        # return argmax of v
        var max_i: Int = 0
        var max_p: Float32 = self.q_table[cur_row, cur_col, 0]
        for i in range(self.q_table.dim2):
            let vv = self.q_table[cur_row, cur_col, i]
            if vv > max_p:
                max_i = i
                max_p = vv
        return max_i

    @always_inline
    fn get_action(self, cur_row: Int, cur_col: Int) -> Int:
        # if a randomly chosen value between 0 and 1 is less than epsilon,
        # then choose the most promising value from the Q-table for this state.
        let _rnd = rand[DType.float16](1)
        if  _rnd[0] < self.epsilon:
            return self.argmax(cur_row, cur_col)
        else:  # choose a random action
            let ret = rand[DType.uint16](1)[0] % len(self.actions)
            return ret.to_int()


    fn train(inout self) raises:
        for episode in range(self.episodes):
            var pos = self.env.get_starting_location()
            while not self.env.is_terminal_state(pos[0], pos[1]):
                let action = self.get_action(pos[0], pos[1])
                let next_pos = self.env.get_next_location(pos[0], pos[1], action)
                let next_action = self.get_action(next_pos[0], next_pos[1])
                let reward = self.env.get_reward(next_pos[0], next_pos[1])
                let old_q_value = self.q_table[pos[0], pos[1], action]
                let _max_index = self.argmax(next_pos[0], next_pos[1])
                let _max_val = self.q_table[next_pos[0], next_pos[1], _max_index]
                let temporal_difference = reward + self.discount_factor * _max_val
                    - self.q_table[pos[0], pos[1], action]
                let new_q_value = old_q_value + self.learning_rate * temporal_difference
                self.q_table[pos[0], pos[1], action] = new_q_value
                pos = next_pos

    fn shortest(start_row, start_col):
        # return immediately if this is an invalid starting location
        if is_terminal_state(start_row_index, start_column_index):
            return []
        else:  # if this is a 'legal' starting location
            current_row_index, current_column_index = start_row_index, start_column_index
            shortest_path = []
            shortest_path.append([current_row_index, current_column_index])
            # continue moving along the path until we reach the goal (i.e., the item packaging location)
            while not is_terminal_state(current_row_index, current_column_index):
                # get the best action to take
                action_index = get_next_action(current_row_index, current_column_index, 1.)
                # move to the next location on the path, and add the new location to the list
                current_row_index, current_column_index = get_next_location(current_row_index, current_column_index,
                                                                            action_index)
                shortest_path.append([current_row_index, current_column_index])
            return shortest_path


    fn print_results(inout self):
        print("Training finished.\n")
        print("Q Table:\n")
        for i in range(self.q_table.dim2):
            print("Action: ", i)
            self.q_table.print(i)
            print("\n")



