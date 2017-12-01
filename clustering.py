import itertools
import random
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional


class SelectionEnum(Enum):
    COUPLING_HI = auto()
    COUPLING_LO = auto()
    COHESION_HI = auto()
    COHESION_LO = auto()
    RANDOM = auto()


class ConditionEnum(Enum):
    MQ_G_GT_C = auto()
    MQ_G_LT_C = auto()
    COUPLING_A_GT_B = auto()
    COUPLING_A_LT_B = auto()
    COHESION_A_GT_B = auto()
    COHESION_A_LT_B = auto()
    SIZE_A_GT_B = auto()
    SIZE_A_LT_B = auto()
    ALWAYS_TRUE = auto()
    # MF_A_GT_B
    # RANDOMLY_C


class ActionEnum(Enum):
    MERGE_A_B = auto()
    MOVE_A_TO_B = auto()
    MOVE_B_TO_A = auto()
    TEAR_A = auto()
    TEAR_B = auto()


class MutateEnum(Enum):
    SHUFFLE_BLOCKS = auto()
    ADD_BLOCK = auto()
    REMOVE_BLOCK = auto()


class Graph:
    def __init__(self, graph_path: Path):
        self.graph = None

        with graph_path.open() as f:
            self.graph = tuple([int(num) for num in line.split(',')] for line in f.readlines())

    @property
    def size(self):
        return len(self.graph)


# TODO: make immutable
class ClusterStats:
    def __init__(self):
        self.coupling = 0
        self.cohesion = 0
        self.size = 0

    @property
    def mf(self):
        return (self.cohesion * 2) / (self.cohesion * 2 + self.coupling) if self.cohesion else 0


class Clustering:
    def __init__(self, graph: Graph, cluster_repr: Optional[List[int]] = None):
        self._cluster_repr = None
        self._cohesion = None
        self._coupling = None
        self._mq = None
        self._graph = graph
        self._cluster_stats = None

        if cluster_repr is None:
            cluster_repr = list(range(graph.size))

        self.cluster_repr = cluster_repr

    @property
    def cluster_repr(self):
        return self._cluster_repr[:]

    @cluster_repr.setter
    def cluster_repr(self, value: List[int]):
        if graph.size != len(value):
            raise ValueError

        self._cluster_repr = value
        self._calc_metrics()

    @property
    def graph(self):
        return self._graph

    @property
    def cluster_ids(self) -> List[int]:
        return list(set(self.cluster_repr))

    @property
    def cohesion(self):
        return self._cohesion

    @property
    def coupling(self):
        return self._coupling

    @property
    def mq(self):
        return self._mq

    def get_cluster(self, selection_method: SelectionEnum) -> int:
        if selection_method is SelectionEnum.RANDOM:
            return random.choice(self.cluster_ids)
        elif selection_method is SelectionEnum.COUPLING_HI:
            return sorted(self._cluster_stats.items(), key=lambda x: x[1].coupling, reverse=True)[0][0]
        elif selection_method is SelectionEnum.COUPLING_LO:
            return sorted(self._cluster_stats.items(), key=lambda x: x[1].coupling)[0][0]
        elif selection_method is SelectionEnum.COHESION_HI:
            return sorted(self._cluster_stats.items(), key=lambda x: x[1].cohesion, reverse=True)[0][0]
        elif selection_method is SelectionEnum.COHESION_LO:
            return sorted(self._cluster_stats.items(), key=lambda x: x[1].cohesion)[0][0]

    def get_modules(self, cluster: int) -> List[int]:
        candidates = list(filter(lambda x: x[1] == cluster, enumerate(self._cluster_repr)))

        return [x[0] for x in candidates]

    def move_module(self, from_module: int, to_module: int):
        new_repr = self.cluster_repr

        new_repr[from_module] = new_repr[to_module]

        self.cluster_repr = new_repr

    def merge_module(self, from_module: int, to_module: int):
        pass

    def tear_module(self, target_module: int):
        pass

    def get_cluster_stat(self, cluster_id: int):
        return self._cluster_stats[cluster_id]

    def _calc_metrics(self):
        self._cluster_stats = dict((cluster_id, ClusterStats()) for cluster_id in self.cluster_ids)

        for i in range(self._graph.size):
            from_cluster = self._cluster_repr[i]
            self._cluster_stats[from_cluster].size += 1
            for j in range(self._graph.size):
                to_cluster = self._cluster_repr[j]

                graph_value = self._graph.graph[i][j]

                if graph_value == 0:
                    continue

                if from_cluster == to_cluster:
                    self._cluster_stats[from_cluster].cohesion += graph_value
                else:
                    self._cluster_stats[from_cluster].coupling += graph_value
                    self._cluster_stats[to_cluster].coupling += graph_value

        self._cohesion = sum((x.cohesion for x in self._cluster_stats.values()))
        self._coupling = sum((x.coupling for x in self._cluster_stats.values()))

        # if 0 in self._cluster_stats:
        #     print(self.cluster_repr, self._cluster_stats[0].__dict__)

        self._mq = sum((x.mf for x in self._cluster_stats.values()))

    def __str__(self):
        return ', '.join(map(str, [self.cluster_repr, self.mq]))


class ModelBlock:
    CONDITION_LAMBDA = {
        ConditionEnum.MQ_G_GT_C: lambda cl, a, b, c: cl.mq > c,
        ConditionEnum.MQ_G_LT_C: lambda cl, a, b, c: cl.mq < c,
        ConditionEnum.COUPLING_A_GT_B: lambda cl, a, b, c: cl.get_cluster_stat(a).coupling > cl.get_cluster_stat(
            b).coupling,
        ConditionEnum.COUPLING_A_LT_B: lambda cl, a, b, c: cl.get_cluster_stat(a).coupling < cl.get_cluster_stat(
            b).coupling,
        ConditionEnum.COHESION_A_GT_B: lambda cl, a, b, c: cl.get_cluster_stat(a).cohesion > cl.get_cluster_stat(
            b).cohesion,
        ConditionEnum.COHESION_A_LT_B: lambda cl, a, b, c: cl.get_cluster_stat(a).cohesion < cl.get_cluster_stat(
            b).cohesion,
        ConditionEnum.SIZE_A_GT_B: lambda cl, a, b, c: cl.get_cluster_stat(a).size > cl.get_cluster_stat(b).size,
        ConditionEnum.SIZE_A_LT_B: lambda cl, a, b, c: cl.get_cluster_stat(a).size < cl.get_cluster_stat(b).size,
        ConditionEnum.ALWAYS_TRUE: lambda cl, a, b, c: True
    }

    ACTION_LAMBDA = {
        ActionEnum.MERGE_A_B: lambda cl, a, b: cl.merge_module(a, b),
        ActionEnum.MOVE_A_TO_B: lambda cl, a, b: cl.move_module(a, b),
        ActionEnum.MOVE_B_TO_A: lambda cl, a, b: cl.move_module(b, a),
        ActionEnum.TEAR_A: lambda cl, a, b: cl.tear_module(a),
        ActionEnum.TEAR_B: lambda cl, a, b: cl.tear_module(b),
    }

    def __init__(self,
                 a_selection: Optional[SelectionEnum]=None,
                 b_selection: Optional[SelectionEnum]=None,
                 cond: Optional[ConditionEnum]=None,
                 cond_const: Optional[float]=None,
                 action: Optional[ActionEnum]=None):

        if a_selection is None:
            a_selection = random.choice(list(SelectionEnum.__members__.values()))
        if b_selection is None:
            b_selection = random.choice(list(SelectionEnum.__members__.values()))
        if cond is None:
            cond = random.choice(list(ConditionEnum.__members__.values()))
        if cond_const is None:
            cond_const = random.random()
        if action is None:
            action = random.choice(list(ActionEnum.__members__.values()))

        self.a_selection = a_selection
        self.b_selection = b_selection
        self.cond = cond
        self.cond_const = cond_const
        self.action = action

    @property
    def props(self):
        return self.a_selection, self.b_selection, self.cond, self.cond_const, self.action

    def apply(self, clustering: Clustering):
        a_cluster = clustering.get_cluster(self.a_selection)
        b_cluster = clustering.get_cluster(self.b_selection)

        # print(a_cluster, clustering.get_cluster_stat(a_cluster))
        # print(b_cluster, clustering.get_cluster_stat(b_cluster))

        a_module = random.choice(clustering.get_modules(a_cluster))
        b_module = random.choice(clustering.get_modules(b_cluster))

        if self.CONDITION_LAMBDA[self.cond](clustering, a_cluster, b_cluster, self.cond_const):
            self.ACTION_LAMBDA[self.action](clustering, a_module, b_module)

    def __str__(self):
        return ', '.join(map(str, self.props))


class Model:
    def __init__(self, blocks: List[ModelBlock], max_iter=200, max_skip=5, no_skip=False):
        self._no_skip = no_skip
        self._max_skip = max_skip
        self._max_iter = max_iter
        self._blocks = blocks

    @property
    def blocks(self):
        return self._blocks[:]

    def generate_cluster(self, graph: Graph) -> Clustering:
        cluster = Clustering(graph)

        skipped = 0
        iter = 0

        while iter < self._max_iter:
            new_cluster = Clustering(cluster.graph, cluster.cluster_repr)

            # print(new_cluster.cluster_repr)

            for block in self._blocks:
                iter += 1
                block.apply(new_cluster)

            if self._no_skip or new_cluster.mq > cluster.mq:
                cluster = new_cluster
                skipped = 0
            else:
                skipped += 1

                if skipped > self._max_skip:
                    break

        return cluster


class ModelGenerator:
    def __init__(self, max_iter=200, max_skip=5, no_skip=True):
        self._max_iter = max_iter
        self._max_skip = max_skip
        self._no_skip = no_skip

    def generate_model(self, blocks: List[ModelBlock]) -> Model:
        return Model(blocks, self._max_iter, self._max_skip, self._no_skip)


class GeneticAlgorithm:
    def __init__(self, model_gen: ModelGenerator):
        self._model_gen = model_gen
        pass

    def run(self, graph: Graph, max_pop=20, max_gen=30, max_blocks=5) -> List[Model]:
        generation = [self._model_gen.generate_model([ModelBlock()]) for _ in range(max_pop)]

        gen_count = 0

        while gen_count <= max_gen:
            crossed = itertools.chain(*[self._cross(*random.sample(generation, k=2)) for _ in range(max_pop)])
            mutated = filter(None, [self._mutate(random.choice(generation)) for _ in range(max_pop)])
            merged = filter(lambda x: len(x.blocks) <= max_blocks, itertools.chain(generation, crossed, mutated))

            evals = [(model, model.generate_cluster(graph)) for model in merged]
            evals = sorted(evals, key=lambda x: x[1].mq, reverse=True)
            generation = [x[0] for x in evals[:max_pop]]

            print(gen_count, evals[0][1].mq)

            gen_count += 1

        return generation

    def _cross(self, a: Model, b: Model) -> List[Model]:
        a_index = random.randrange(len(a.blocks)) + 1
        b_index = random.randrange(len(b.blocks)) + 1

        new_blocks = (a.blocks[a_index:] + b.blocks[:b_index],
                      b.blocks[b_index:] + a.blocks[:a_index])

        return [self._model_gen.generate_model(new_block) for new_block in new_blocks]

    def _mutate(self, a: Model, mutate_type: Optional[MutateEnum]=None) -> Optional[Model]:
        if mutate_type is None:
            mutate_type = random.choice(list(MutateEnum.__members__.values()))

        new_blocks = a.blocks

        if mutate_type is MutateEnum.SHUFFLE_BLOCKS:
            if len(new_blocks) < 2:
                return None

            random.shuffle(new_blocks)
        elif mutate_type is MutateEnum.ADD_BLOCK:
            new_blocks.insert(random.randrange(len(new_blocks)), ModelBlock())
        elif mutate_type is MutateEnum.REMOVE_BLOCK:
            if len(new_blocks) < 2:
                return None

            new_blocks.remove(new_blocks[random.randrange(len(new_blocks))])

        return self._model_gen.generate_model(new_blocks)


if __name__ == '__main__':
    graph = Graph(Path('data/input_gen1.csv'))
    ga = GeneticAlgorithm()

    models = ga.run(graph, max_pop=10, max_gen=20, max_blocks=10)
    print('results')

    print(max((model.generate_cluster(graph).mq for model in models)))
