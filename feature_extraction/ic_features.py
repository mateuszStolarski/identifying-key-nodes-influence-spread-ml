import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


def prepare_ic(G, threshold, infected_nodes, seed):
    model = ep.IndependentCascadesModel(G, seed=seed)

    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", infected_nodes)

    if threshold != 0.1:
        for e in G.edges():
            config.add_edge_configuration("threshold", e, threshold)

    model.set_initial_status(config)
    return model


def execute_ic(model):
    max_si = 1
    max_iter = 1
    counter = 0
    recovered = False

    while not recovered:
        counter += 1
        iteration = model.iteration()

        if iteration["node_count"][1] > max_si:
            max_si = iteration["node_count"][1]
            max_iter = counter
        if iteration["node_count"][1] == 0:
            recovered = True

    return max_si, max_iter, iteration["node_count"][2]


def prepare_and_execute_ic(infected_node, G, threshold=0.1, seed=None):
    model = prepare_ic(G, threshold, [int(infected_node)], seed=seed)
    max_si, max_iter, max_recovered = execute_ic(model)

    return max_si, max_iter, max_recovered
