import networkx as nx
import pandas as pd
from gurobipy import Model, GRB, LinExpr
import os

cut_count = 0
bound_check = 0
node_count = 0
global bound_save


# -----separates integer solution
def first_depth_k_bfs(graph, cost, connect, L, cut_limit, model):
    global cut_count
    roots = [n for (n, attr) in cost.items() if attr < 1 - 1e-5]
    input_graph = graph.subgraph(roots)
    for rt in roots:
        cut_count = 0
        length, path = nx.single_source_dijkstra(input_graph, rt,
                                                 cutoff=L, weight='weight')
        for v, distance in length.items():
            if rt != v:
                i = min([rt, v])
                j = max([rt, v])
                if connect[(i, j)] < f[distance - 1]:
                    model.cbLazy((f[distance - 1]
                                  * sum(model._x_delete[node] for node in path[v]))
                                 + model._u_connect[i, j] >= f[distance - 1])
                    cut_count += 1
                    if cut_count == cut_limit:
                        break


# -----separates fractional solution
def second_depth_k_bfs(graph, cost, connect, L, cut_limit, model):
    global cut_count
    roots = [n for (n, attr) in cost.items() if attr < 1 - 1e-5]
    # get a SubGraph view of the subgraph induced on nodes
    input_graph = graph.subgraph(roots)
    for rt in roots:
        cut_count = 0
        # Find shortest weighted paths and lengths from a source node.
        # Compute the shortest path length between source and all other reachable nodes for a weighted graph.
        length, path = nx.single_source_dijkstra(input_graph, rt, cutoff=L, weight='weight')
        for v, distance in length.items():
            if rt != v:
                i = min([rt, v])
                j = max([rt, v])
                if (f[distance - 1] * sum(cost[node] for node in path[v])) + connect[(i, j)] < f[distance - 1]:
                    model.cbLazy( (f[distance - 1] * sum(model._x_delete[node] for node in path[v]))
                                  + model._u_connect[i, j] >= f[distance - 1])
                    cut_count += 1
                    if cut_count == cut_limit:
                        break


def cut(model, where):
    global bound_save, bound_check, node_count
    cost = {}
    connect = {}

    # if integer solution
    if where == GRB.Callback.MIPSOL:
        bound_save = model.cbGet(GRB.callback.MIPSOL_OBJBND)
        node_count = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        for j in G.nodes():
            cost[j] = abs(model.cbGetSolution(model._x_delete[j]))
            for i in G.nodes():
                if i < j:
                    connect[(i, j)] = abs(model.cbGetSolution(model._u_connect[i, j]))
        first_depth_k_bfs(G, cost, connect, L, GRB.INFINITY, model)

    # if fractional solution
    elif where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
            current_bound = model.cbGet(GRB.callback.MIPNODE_OBJBND)
            node_count = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
            if bound_save == current_bound:
                bound_check += 1
                if bound_check >= 5 or node_count > 0:
                    bound_check = 0
                else:
                    for j in G.nodes():
                        cost[j] = abs(model.cbGetNodeRel(model._x_delete[j]))
                        for i in G.nodes():
                            if i < j:
                                connect[(i, j)] = abs(model.cbGetNodeRel(model._u_connect[i, j]))
                    second_depth_k_bfs(G, cost, connect, L, 300, model)

            else:
                bound_save = current_bound
                for j in G.nodes():
                    cost[j] = abs(model.cbGetNodeRel(model._x_delete[j]))
                    for i in G.nodes():
                        if i < j:
                            connect[(i, j)] = abs(model.cbGetNodeRel(model._u_connect[i, j]))
                second_depth_k_bfs(G, cost, connect, L, 300, model)


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
                u_connect[i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                               name=f'u[{i}, {j}]')

    # objective
    obj = LinExpr(0)
    for j in input_graph.nodes():
        for i in input_graph.nodes():
            if i < j:
                obj.add(u_connect[i, j])

    # constraint on number of critical nodes
    model.addConstr(sum((x_delete[j]) for j in input_graph.nodes()) <= C)

    # constraints on connectivity variables u
    # constraints on (i,j) in E
    for (i, j) in input_graph.edges():
        weight = input_graph.edges[i, j]['weight']
        if i < j:
            model.addConstr(u_connect[i, j] + f[weight - 1] * (x_delete[i] + x_delete[j]) >= f[weight - 1])
        else:  # that is j<i
            model.addConstr(u_connect[j, i] + f[weight - 1] * (x_delete[j] + x_delete[i]) >= f[weight - 1])

    model.update()
    model.setObjective(obj, GRB.MINIMIZE)
    model._x_delete = x_delete
    model._u_connect = u_connect
    model.setParam(GRB.param.Cuts, 0)
    model.setParam(GRB.param.PreCrush, 1)
    model.setParam('LazyConstraints', 1)
    model.setParam('TimeLimit', 3600)
    model.optimize(cut)
    run_time = model.Runtime
    x_delete_val = model.getAttr('x', x_delete)

    critical_nodes = [i for i in x_delete_val.keys() if x_delete_val[i] >= 1 - 1e-4]

    return critical_nodes, obj.getValue(), run_time, model


columns = ['Graph_name', 'n_nodes', 'n_edges', 'diameter', 'b', 'cost', 'final_obj', 'n_vars', 'status', 'run_time']

dir = '../data/Real_networks/'


def get_graph(path: str):
    if path.endswith('.edgelist'):
        return nx.read_edgelist(path=dir + path, nodetype=int)
    else:
        return nx.read_gml(path=dir + path, label='id')


if __name__ == "__main__":
    result = []
    for file in os.listdir("data_test"):
        if not file.startswith('.'):
            G = get_graph(file)
            # assign weights to edges based on edge betweenness
            edge_btw = nx.edge_betweenness_centrality(G, normalized=True)
            for e in G.edges():
                G.edges[e[0], e[1]]['weight'] = min(6, max(1, round(0.1 / edge_btw[e])))
            bound_check = 0
            L = nx.diameter(G)  # diameter of the graph
            n = G.number_of_nodes()
            C = int(0.05 * n)  # budget on critical nodes
            ind = 0
            # define distance connectivity function (eg f(d)=1/d)
            f = []
            for l in range(L + 1):
                f.append(1 / float(l + 1))

            critical_nodes, opt_obj, run_time, model = minimize_dcnp(G, L, C)
            result.append(
                {
                    columns[0]: file.split('.')[0].upper(),
                    columns[1]: n,
                    columns[2]: G.number_of_edges(),
                    columns[9]: L,
                    columns[3]: round(2 * 100 * opt_obj / (G.number_of_nodes() * (G.number_of_nodes() - 1)), 2),
                    columns[4]: C,
                    columns[5]: model.getAttr(GRB.Attr.NumVars),
                    columns[7]: model.getAttr(GRB.Attr.Status),
                    columns[8]: round(run_time, 3),
                }
            )

    df = pd.DataFrame(result, columns=columns)
    df.to_csv('result_cost_dcndp_weight_big_test.csv', index=False)
