import collections
import os

import networkx as nx
import numpy as np
from gurobipy import Model, GRB, LinExpr
import pandas as pd

cut_count = 0
bound_check = 0
global bound_save


# bfs tree generation to separate integer solution
def first_depth_k_bfs(input_graph, cost, connect, L, cut_limit, model):
    global cut_count
    roots = [n for (n, attr) in cost.items() if attr == 0]
    for root in roots:
        # keep track of all visited nodes and nodes to be checked
        visited, queue = {root}, collections.deque([root])
        # this dict keeps track of levels
        levels = {root: 0}

        # this dict keeps track of predecessors
        predecessor = {root: -1}
        cut_count = 0

        # keep looping until there are no nodes still to be checked
        while queue:
            if cut_count > cut_limit:
                break
            # pop first node from the queue
            vertex = queue.popleft()
            for neighbour in input_graph[vertex]:
                # only consider neighbors with solution=0
                if neighbour not in visited and cost[neighbour] <= 1e-5:
                    new_levels = levels[vertex] + 1
                    if new_levels == 1:  # direct neighbours of root node
                        predecessor[neighbour] = vertex
                        levels[neighbour] = new_levels
                        # mark neighbours of node as visited to avoid revisiting
                        visited.add(neighbour)
                        # add neighbours of node to queue
                        queue.append(neighbour)

                    elif new_levels in range(2, L + 1):
                        predecessor[neighbour] = vertex
                        levels[neighbour] = new_levels
                        # mark neighbours of node as visited to avoid revisiting
                        visited.add(neighbour)
                        # add neighbours of node to queue
                        queue.append(neighbour)
                        i = min([root, neighbour])
                        j = max([root, neighbour])
                        if connect[(i, j, new_levels)] < 1 - 1e-5:
                            lazy_cut_lhs = LinExpr(0)
                            lazy_cut_lhs.add(model._x_delete[root])
                            while neighbour != root:
                                lazy_cut_lhs.add(model._x_delete[neighbour])
                                nxt = predecessor[neighbour]
                                neighbour = nxt
                            model.cbLazy(lazy_cut_lhs + model._u_connect[i, j, new_levels] >= 1)
                            cut_count += 1

                    else:  # new_levels >k
                        break
            else:
                continue
            break


def second_depth_k_bfs(input_graph, cost, connect, root, L, cut_limit, model):
    global cut_count
    # keep track of all visited nodes and nodes to be checked
    visited, queue = {root}, collections.deque([root])
    levels = {root: 0}  # this dict keeps track of levels
    # this dict keeps track of distance label-sum of lp-solution value of each node along the path
    dist_label = {root: cost[root]}

    # this dict keeps track of predecessors
    predecessor = {root: -1}
    cut_count = 0

    # keep looping until there are no nodes still to be checked
    while queue:
        if cut_count > cut_limit:
            return
        # pop first node from the queue
        vertex = queue.popleft()
        for neighbour in sorted(input_graph[vertex],
                                key=lambda x: input_graph.nodes[x]['LPsol']): # order the set of neighbors according to LP-relaxation solution
            # for nbr in input_graph[vertex]:
            if neighbour not in visited:  # and cost[nbr]==0:
                new_levels = levels[vertex] + 1
                if new_levels == 1:
                    predecessor[neighbour] = vertex
                    dist_label[neighbour] = dist_label[vertex] + cost[neighbour]
                    levels[neighbour] = new_levels
                    # mark neighbours of node as visited to avoid revisiting
                    visited.add(neighbour)
                    # add neighbours of node to queue
                    queue.append(neighbour)

                elif new_levels in range(2, L + 1):
                    predecessor[neighbour] = vertex
                    dist_label[neighbour] = dist_label[vertex] + cost[neighbour]
                    levels[neighbour] = new_levels
                    # mark neighbours of node as visited to avoid revisiting
                    visited.add(neighbour)
                    # add neighbours of node to queue
                    queue.append(neighbour)
                    i = min([root, neighbour])
                    j = max([root, neighbour])
                    if dist_label[neighbour] + connect[(i, j, new_levels)] < 1 - 1e-5:
                        lazy_cut_lhs = LinExpr(0)
                        lazy_cut_lhs.add(model._x_delete[root])
                        while neighbour != root:
                            lazy_cut_lhs.add(model._x_delete[neighbour])
                            nxt = predecessor[neighbour]
                            neighbour = nxt
                        model.cbLazy(lazy_cut_lhs + model._u_connect[i, j, new_levels] >= 1)
                        cut_count += 1

                else:  # new_levels >k
                    break
        else:
            continue
        break


def cut(model, where):
    global bound_save, bound_check
    cost = {}
    connect = {}

    if where == GRB.Callback.MIPSOL:  # if integer solution
        # get MIPSOL_OBJBDN = Current best objective bound
        bound_save = model.cbGet(GRB.callback.MIPSOL_OBJBND)
        for j in G.nodes():
            # retrieve values from the new MIP solution
            cost[j] = abs(model.cbGetSolution(model._x_delete[j]))
            for i in G.nodes():  # range(ind,j):
                if i < j:
                    for l in range(1, L + 1):
                        connect[(i, j, l)] = abs(model.cbGetSolution(model._u_connect[i, j, l]))

        first_depth_k_bfs(G, cost, connect, L, GRB.INFINITY)

    # if fractional solution
    elif where == GRB.Callback.MIPNODE:
        # Optimization status of current MIP node if it is OPTIMAL
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
            # get current best objective bound
            current_bound = model.cbGet(GRB.callback.MIPNODE_OBJBND)
            # get current explored node count
            node_count = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
            if bound_save == current_bound:
                bound_check += 1
                if bound_check >= 5 or node_count > 0:
                    bound_check = 0
                else:
                    for j in G.nodes():
                        cost[j] = abs(model.cbGetNodeRel(model._x_delete[j]))
                        G.nodes[j]['LPsol'] = cost[j]  # set node attributes to lp solution
                        for i in G.nodes():
                            if i < j:
                                for l in range(1, L + 1):
                                    connect[(i, j, l)] = abs(model.cbGetNodeRel(model._u_connect[i, j, l]))
                    roots = [n for (n, attr) in cost.items() if attr < 1]
                    for rt_node in roots:
                        second_depth_k_bfs(G, cost, connect, rt_node, L, 300)

            else:
                bound_save = current_bound
                for j in G.nodes():
                    cost[j] = abs(model.cbGetNodeRel(model._x_delete[j]))
                    G.nodes[j]['LPsol'] = cost[j]  # set node attributes to lp solution
                    for i in G.nodes():
                        if i < j:
                            for l in range(1, L + 1):
                                connect[(i, j, l)] = abs(model.cbGetNodeRel(model._u_connect[i, j, l]))
                roots = [n for (n, attr) in cost.items() if attr < 1]
                for rt_node in roots:
                    second_depth_k_bfs(G, cost, connect, rt_node, L, 300)


# ---------------Minimize DCNP objective-----------------
def minimize_dcnp(input_graph, L, C):
    model = Model('Minimize distance-based pairwise connectivity eg efficiency')

    # variables
    x_delete = {}
    u_connect = {}
    for j in input_graph.nodes():
        # The node degree is the number of edges adjacent to the node.
        if input_graph.degree[j] == 1:
            x_delete[j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY, name=f'x[{j}')
        else:
            x_delete[j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'x[{j}')
        for i in input_graph.nodes():
            if i < j:
                for l in range(1, L + 1):
                    u_connect[i, j, l] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                                      name="u[%d,%d,%d]" % (i, j, l))

    # objective
    obj = LinExpr(0)
    for j in input_graph.nodes():
        for i in input_graph.nodes():
            if i < j:
                obj.add(f[0] * u_connect[i, j, 1])
                for l in range(1, L):
                    obj.add(f[l] * (u_connect[i, j, l + 1] - u_connect[i, j, l]))

    # constraint on number of critical nodes
    model.addConstr(sum((x_delete[j]) for j in input_graph.nodes()) <= C, name="3.24")

    # constraints on connectivity variables u

    # constraints on (i,j) in E
    for (i, j) in input_graph.edges():
        if i < j:
            model.addConstr(u_connect[i, j, 1] + x_delete[i] + x_delete[j] >= 1,
                            name="Explicitly model the non-redundant constraints and leave the rest to BFS")
            for l in range(2, L + 1):
                model.addConstr(u_connect[i, j, 1] == u_connect[i, j, l], name="3.23_2")
        else:  # that is j<i
            model.addConstr(u_connect[j, i, 1] + x_delete[j] + x_delete[i] >= 1,
                            name="Explicitly model the non-redundant constraints and leave the rest to BFS_part 2")
            for l in range(2, L + 1):
                model.addConstr(u_connect[j, i, 1] == u_connect[j, i, l], name="3.23_2")

    # constraints on (i,j) not in E
    for j in input_graph.nodes():
        for i in input_graph.nodes():
            if i not in input_graph.neighbors(j) and i < j:
                for l in range(1, L):
                    model.addConstr(u_connect[i, j, l] <= u_connect[i, j, l + 1], name="3.22")

    model.update()
    model.setObjective(obj, GRB.MINIMIZE)
    model._x_delete = x_delete
    model._u_connect = u_connect
    model.setParam(GRB.param.Cuts, 0)
    model.setParam(GRB.param.PreCrush, 1)
    model.setParam('LazyConstraints', 1)
    model.setParam('TimeLimit', 10800)
    model.optimize(cut)
    run_time = model.Runtime
    x_delete_val = model.getAttr('x', x_delete)

    critical_nodes = [i for i in x_delete_val.keys() if x_delete_val[i] >= 1 - 1e-4]

    opt_obj = 0
    for j in input_graph.nodes():
        for i in input_graph.nodes():
            if i < j:
                opt_obj += f[0] * u_connect[i, j, 1].X
                for l in range(1, L):
                    opt_obj += f[l] * (u_connect[i, j, l + 1].X - u_connect[i, j, l].X)

    return critical_nodes, opt_obj, run_time, model


columns = ['Graph_name', 'n_nodes', 'n_edges', 'diameter', 'b', 'cost', 'final_obj', 'n_vars', 'status', 'run_time']

dir = 'data_test/'


def get_graph(path: str):
    if path.endswith('.edgelist'):
        return nx.read_edgelist(path=dir + path, nodetype=int)
    else:
        return nx.read_gml(path=dir + path, label='id')


if __name__ == "__main__":
    result = []
    for b in [0.05, 0.1, 0.01]:
        for file in os.listdir('data_test'):
            if not file.startswith('.'):
                G = get_graph(file)
                bound_check = 0
                L = nx.diameter(G)
                C = int(b * G.number_of_nodes())
                ind = 0

                # define communication efficiency function (1/d -for example)
                f = list()
                for l in range(G.number_of_nodes()):
                    f.append(1 / float(l + 1))

                # find the critical nodes
                critical_nodes, opt_obj, run_time, model = minimize_dcnp(G, L, C)
                result.append(
                    {
                        columns[0]: file.split('.')[0].upper(),
                        columns[1]: G.number_of_nodes(),
                        columns[2]: G.number_of_edges(),
                        columns[3]: L,
                        columns[4]: b,
                        columns[5]: C,
                        columns[
                            6]: f'{round(2 * 100 * opt_obj / (G.number_of_nodes() * (G.number_of_nodes() - 1)), 2)}%',
                        columns[7]: model.getAttr(GRB.Attr.NumVars),
                        columns[8]: model.getAttr(GRB.Attr.Status),
                        columns[9]: round(run_time, 3),
                    }
                )
    df = pd.DataFrame(result, columns=columns)
    df.to_csv('result_cost_dcndp_2_scalability_B_cost.csv', index=False)
