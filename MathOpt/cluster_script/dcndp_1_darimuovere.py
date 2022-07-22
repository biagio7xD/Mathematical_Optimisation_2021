import collections
import os
import networkx as nx
import pandas as pd
from gurobipy import Model, GRB, LinExpr
import numpy as np


def cut(model, where):
    global cutcount, bound_check, node_count, savebnd
    cost = {}
    connect = {}

    def first_depth_k_bfs(input_graph: nx.Graph, cost: dict, connect: dict, k_value: int, ctlimit: int):
        roots = [n for (n, attr) in cost.items() if attr == 0]
        for root in roots:
            # keep track of all visited nodes and nodes to be checked
            visited, queue = {root}, collections.deque([root])
            # this dict keeps track of levels
            levels = {root: 0}
            # this dict keeps track of predecessors
            predecessor = {root: -1}
            cutcount = 0

            # keep looping until there are no nodes still to be checked
            while queue:
                if cutcount > ctlimit:
                    break
                # pop first node from the queue
                vertex = queue.popleft()
                for neighbour in input_graph[vertex]:
                    # only consider neighbors with solution=0
                    if neighbour not in visited and cost[neighbour] <= 1e-5:
                        new_level = levels[vertex] + 1
                        # direct neighbours of root node
                        if new_level == 1:
                            predecessor[neighbour] = vertex
                            levels[neighbour] = new_level
                            # mark neighbours of node as visited to avoid revisiting
                            visited.add(neighbour)
                            # add neighbours of node to queue
                            queue.append(neighbour)

                        # new_level in range(2,k)
                        elif new_level == k_value - 1:
                            predecessor[neighbour] = vertex
                            levels[neighbour] = new_level
                            # mark neighbours of node as visited to avoid revisiting
                            visited.add(neighbour)
                            # add neighbours of node to queue
                            queue.append(neighbour)
                            i = min([root, neighbour])
                            j = max([root, neighbour])
                            if connect[(i, j)] < 1 - 1e-5:
                                model.cbLazy(
                                    model._x_delete[neighbour] + model._x_delete[vertex] +
                                    model._x_delete[root] + model._u_connect[i, j] >= 1)
                                cutcount += 1

                        elif new_level == k_value:
                            predecessor[neighbour] = vertex
                            levels[neighbour] = new_level
                            # mark neighbours of node as visited to avoid revisiting
                            visited.add(neighbour)
                            # add neighbours of node to queue
                            queue.append(neighbour)
                            i = min([root, neighbour])
                            j = max([root, neighbour])
                            if connect[(i, j)] < 1 - 1e-5:
                                model.cbLazy(
                                    model._x_delete[neighbour] + model._x_delete[vertex] +
                                    model._x_delete[predecessor[vertex]] + model._x_delete[root] + model._u_connect[
                                        i, j] >= 1)
                                cutcount += 1

                        else:  # new_level >k
                            # print("required depth reached")
                            break
                else:
                    continue
                break

    def second_depth_k_bfs(input_graph: nx.Graph, cost: dict, connect: dict, root: dict, k_value: int, ctlimit: int):
        # keep track of all visited nodes and nodes to be checked
        visited, queue = {root}, collections.deque([root])
        # this dict keeps track of levels
        levels = {root: 0}
        # this dict keeps track of distance label sum of lp-solution value of each node along the path
        dist_label = {root: cost[root]}

        # this dict keeps track of predecessor
        predecessor = {root: -1}
        cutcount = 0

        # keep looping until there are no nodes still to be checked
        while queue:
            if cutcount > ctlimit:
                # print("cutlimit reached")
                return
            # pop first node from the queue
            vertex = queue.popleft()
            # order the set of neighbors according to LP-relaxation solution
            for neighbour in sorted(input_graph[vertex], key=lambda x: input_graph.nodes[x]['LPsol']):
                # for neighbour in input_graph[vertex]:
                if neighbour not in visited:  # and cost[neighbour]==0:
                    new_level = levels[vertex] + 1
                    # levels[neighbour]= levels[vertex]+1
                    if new_level == 1:
                        predecessor[neighbour] = vertex
                        dist_label[neighbour] = dist_label[vertex] + cost[neighbour]
                        levels[neighbour] = new_level
                        # mark neighbours of node as visited to avoid revisiting
                        visited.add(neighbour)
                        # add neighbours of node to queue
                        queue.append(neighbour)

                    elif new_level == k_value - 1:  # new_level in range(2,k)
                        predecessor[neighbour] = vertex
                        dist_label[neighbour] = dist_label[vertex] + cost[neighbour]
                        levels[neighbour] = new_level
                        # mark neighbours of node as visited to avoid revisiting
                        visited.add(neighbour)
                        # add neighbours of node to queue
                        queue.append(neighbour)
                        i = min([root, neighbour])
                        j = max([root, neighbour])
                        if dist_label[neighbour] + connect[(i, j)] < 1 - 1e-5:
                            model.cbLazy(model._x_delete[neighbour] + model._x_delete[vertex] + model._x_delete[root] +
                                         model._u_connect[i, j] >= 1)
                            cutcount += 1

                    elif new_level == k_value:
                        predecessor[neighbour] = vertex
                        dist_label[neighbour] = dist_label[vertex] + cost[neighbour]
                        levels[neighbour] = new_level
                        # mark neighbours of node as visited to avoid revisiting
                        visited.add(neighbour)
                        # add neighbours of node to queue
                        queue.append(neighbour)
                        i = min([root, neighbour])
                        j = max([root, neighbour])
                        if dist_label[neighbour] + connect[(i, j)] < 1 - 1e-5:
                            model.cbLazy(
                                model._x_delete[neighbour] + model._x_delete[vertex] +
                                model._x_delete[predecessor[vertex]] +
                                model._x_delete[root] + model._u_connect[i, j] >= 1)
                            cutcount += 1

                    else:  # new_level >k
                        # print("required depth reached")
                        break
            else:
                continue
            break

    if where == GRB.Callback.MIPSOL:
        savebnd = model.cbGet(GRB.callback.MIPSOL_OBJBND)
        current_explored_node_count = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        for j in G.nodes():
            cost[j] = abs(model.cbGetSolution(model._x_delete[j]))
            for i in G.nodes():  # range(ind, j):
                if i < j:
                    connect[(i, j)] = abs(model.cbGetSolution(model._u_connect[i, j]))
        if current_explored_node_count == 0:
            first_depth_k_bfs(G, cost, connect, k, GRB.INFINITY)
        else:
            first_depth_k_bfs(G, cost, connect, k, 150)

    elif where == GRB.Callback.MIPNODE:  # if relaxed solution
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
            current_best_obj_bound = model.cbGet(GRB.callback.MIPNODE_OBJBND)
            current_explored_node_count = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
            if savebnd == current_best_obj_bound:
                bndcheck += 1
                if bndcheck >= 5 and current_explored_node_count > 0:
                    bndcheck = 0
                else:
                    for j in G.nodes():
                        cost[j] = abs(model.cbGetNodeRel(model._x_delete[j]))
                        G.nodes[j]['LPsol'] = cost[j]  # set node attributes to lp solution
                        for i in G.nodes():  # range(ind, j):
                            if i < j:
                                connect[(i, j)] = abs(model.cbGetNodeRel(model._u_connect[i, j]))

                    roots = [n for (n, attr) in cost.items() if attr < 1]
                    for rt_node in roots:  # [0:int(0.5*len(roots))]:
                        second_depth_k_bfs(G, cost, connect, rt_node, k, 150)

            else:
                savebnd = current_best_obj_bound
                for j in G.nodes():
                    cost[j] = abs(model.cbGetNodeRel(model._x_delete[j]))
                    G.nodes[j]['LPsol'] = cost[j]  # set node attributes to lp solution
                    for i in G.nodes():  # range(ind, j):
                        if i < j:
                            connect[(i, j)] = abs(model.cbGetNodeRel(model._u_connect[i, j]))

                roots = [n for (n, attr) in cost.items() if attr < 1]
                for rt_node in roots:  # [0:int(0.5*len(roots))]:
                    second_depth_k_bfs(G, cost, connect, rt_node, k, 150)


def minimize_dcnp(input_graph: nx.Graph, c: float):
    model = Model('Minimize distance -based pairwise connectivity of distance at most k ')
    # variables
    x_delete = {}
    u_connect = {}
    for j in input_graph.nodes():
        if input_graph.degree[j] == 1:
            x_delete[j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY, name=f'x[{j}]')
        else:
            x_delete[j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'x[{j}]')
        # connectivity variables
        for i in input_graph.nodes():  # range(ind, j):
            if i < j:
                u_connect[i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f'u[{i},{j}]')

    # objective
    obj = LinExpr(0)
    for j in input_graph.nodes():
        for i in input_graph.nodes():  # range(ind, j):
            if i < j:
                obj.add(u_connect[i, j])

    # constraint on number of critical nodes
    model.addConstr(sum((x_delete[j]) for j in input_graph.nodes()) <= C)

    # constraints on connectivity variables u
    # constraints on (i,j) in E
    for (i, j) in input_graph.edges():
        if i < j:
            model.addConstr(u_connect[i, j] + x_delete[i] + x_delete[j] >= 1)
        else:  # that is j<i
            model.addConstr(u_connect[j, i] + x_delete[j] + x_delete[i] >= 1)

    model.update()
    model.setObjective(obj, GRB.MINIMIZE)
    model._x_delete = x_delete
    model._u_connect = u_connect
    # model.setParam('LogToConsole', 0)
    model.setParam(GRB.param.Cuts, 0)
    model.setParam(GRB.param.PreCrush, 1)
    model.setParam('LazyConstraints', 1)
    model.setParam('TimeLimit', 3600)
    # model.write("DCNP-1PBM2bfs.lp")
    model.optimize(cut)
    run_time = model.Runtime
    xval = model.getAttr('x', x_delete)
    # model.write("DCNP-1PBM2bfs.sol")

    critical_nodes = [i for i in xval.keys() if xval[i] >= 1 - 1e-4]

    opt_obj = 0
    for j in input_graph.nodes():
        for i in input_graph.nodes():  # range(ind, j):
            if i < j:
                opt_obj += u_connect[i, j].X
    return critical_nodes, opt_obj, run_time, model


columns = ['Graph_name', 'n_nodes', 'n_edges', 'diameter', 'b', 'cost', 'final_obj', 'n_vars', 'status', 'run_time']
dir = '../data/Synthetic_networks/synthetic_graphs/'


def get_graph(path: str):
    if path.endswith('.edgelist'):
        return nx.read_edgelist(path=dir + path, nodetype=int)
    else:
        return nx.read_gml(path=dir + path, label='id')


if __name__ == "__main__":
    result = []
    B = [0.05, 0.1]
    for b in B:
        for file in os.listdir(dir):
            if not file.startswith('.'):
                print(file)
                G = get_graph(file)
                L = nx.diameter(G)  # diameter of the graph
                n = G.number_of_nodes()
                C = int(b * n)  # budget on critical nodes
                k = 3
                node_count = 0
                bound_check = 0
                critical_nodes, opt_obj, run_time, model = minimize_dcnp(G, C)
                result.append(
                    {
                        columns[0]: file.split('.')[0].upper(),
                        columns[1]: n,
                        columns[2]: G.number_of_edges(),
                        columns[3]: L,
                        columns[4]: b,
                        columns[5]: C,
                        columns[6]: f'{round(2 * 100 * opt_obj / (G.number_of_nodes() * (G.number_of_nodes() - 1)), 2)}%',
                        columns[7]: model.getAttr(GRB.Attr.NumVars),
                        columns[8]: model.getAttr(GRB.Attr.Status),
                        columns[9]: round(run_time, 3),
                    }
                )
    df = pd.DataFrame(result, columns=columns)
    df.to_csv('result_dcndp_1_synthetic_graphs.csv', index=False)
