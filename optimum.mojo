from mmatrix import Matrix
from qstarlearning import FREE, WALL
@register_passable("trivial")
struct Node:
    var g: Int
    var h: Int
    var f: Int
    var index: Int # index in heap
    var parent: Int # index to parent 
    var posx: Int
    var posy: Int

    fn __init__(index: Int, parent: Int, position: (Int, Int)) -> Self:
        return Node { index: index,
            g: 0, h: 0, f: 0,
            parent: parent, posx: position.get[0, Int](),
            posy: position.get[1, Int]() }

    @always_inline
    fn is_none(self: Node) -> Bool:
        return self.posx == -1 and self.posy == -1

    fn __eq__(self, other: Node) -> Bool:
        return self.posx == other.posx and self.posy == other.posy

fn node_none() -> Node:
    return Node(-1, -1, (-1, -1))

fn find_cur(open_list: DynamicVector[Node]) -> Node:
    var cur: Node = node_none()
    var min_f: Int = 1_000_000
    for i in range(len(open_list)):
        if i == 0:
            cur = open_list[i]
            min_f = cur.f
        elif open_list[i].f < min_f:
            cur = open_list[i]
            min_f = cur.f
    return cur

fn remove_node(open_list: DynamicVector[Node], node: Node) -> DynamicVector[Node]:
    var new_list: DynamicVector[Node] = DynamicVector[Node]()
    for i in range(len(open_list)):
        if open_list[i] == node:
            new_list.push_back(open_list[i])
    return new_list

fn node_in(open_list: DynamicVector[Node], node: Node) -> Bool:
    for i in range(len(open_list)):
        if open_list[i] == node:
            return True
    return False

fn astar(maze: Matrix[DType.int8], start: (Int, Int), end: (Int, Int)) -> DynamicVector[(Int, Int)]:
    var start_node = Node(0, -1, start)
    var end_node = Node(0, -1, end)
    var path = DynamicVector[(Int, Int)]()
    var open_list: DynamicVector[Node] = DynamicVector[Node]()
    var closed_list: DynamicVector[Node] = DynamicVector[Node]()
    var closest: Node = node_none()
    open_list.push_back(start_node)
    
    @parameter
    fn calc_closest(node: Node):
        if closest.is_none():
            closest = node
        elif node.f < closest.f:
            closest = node

    while len(open_list):
        var cn = find_cur(open_list)
        closed_list.push_back(cn)
        if cn == end_node:
            path = DynamicVector[(Int, Int)]()
            while cn.parent != -1:
                path.push_back((cn.posx, cn.posy))
                cn = closed_list[cn.parent]
            return path
        var children = DynamicVector[Node]()

        var poss = VariadicList[(Int, Int)](
            (0, -1), (0, 1), (-1, 0), (1, 0)
        )

        for pp in range(len(poss)):
            var new_positions = poss[pp]
            var node_position = (cn.posx + new_positions.get[0, Int](), cn.posy + new_positions.get[1, Int]())
            if node_position.get[0, Int]() > (maze.dim0 - 1) or node_position.get[0, Int]() < 0 
                or node_position.get[1, Int]() > (maze.dim1 - 1) or node_position.get[1, Int]() < 0:
                continue
            if maze[node_position.get[0, Int](), node_position.get[1, Int]()] != FREE:
                continue
            var new_node = Node(len(closed_list), cn.index, node_position)
            children.push_back(new_node)
        for i in range(len(children)):
            var child = children[i]
            if node_in(closed_list, child):
                continue
            child.g = cn.g + 1
            child.h = (child.posx - end_node.posx) * (child.posx - end_node.posx) + (child.posy - end_node.posy) * (child.posy - end_node.posy)
            child.f = child.g + child.h
            calc_closest(child)
            if node_in(open_list, child) and child.g > cn.g:
                continue
            open_list.push_back(child)
    return path

