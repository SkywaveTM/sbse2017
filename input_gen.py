import random

if __name__ == '__main__':
    graph_size = 50
    node_num = 300
    out_name = 'data/input_gen_.csv'

    nodes = random.choices([(a, b) for a in range(graph_size) for b in range(graph_size) if a > b], k=node_num)
    graph = [[0] * graph_size for _ in range(graph_size)]

    for a, b in nodes:
        graph[a][b] = 1
        graph[b][a] = 1

    with open(out_name, 'w') as f:
        f.writelines((','.join(map(str, row)) + '\n' for row in graph))
