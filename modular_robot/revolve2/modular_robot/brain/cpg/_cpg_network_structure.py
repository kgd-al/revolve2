from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Cpg:
    """Identifies a cpg to be used in a cpg network structure."""

    index: int

    def __lt__(self, other):
        return self.index < other.index


@dataclass(frozen=True, init=False)
class CpgPair:
    """A pair of CPGs that assures that the first cpg always has the lowest index."""

    # lowest is automatically set to be the lowest state index of the two
    cpg_index_lowest: Cpg
    cpg_index_highest: Cpg

    def __init__(self, cpg_1: Cpg, cpg_2: Cpg) -> None:
        """
        Initialize this object.

        The order of the provided CPGs is irrelevant.

        :param cpg_1: One of the CPGs part of the pair.
        :param cpg_2: The other CPG part of the pair.
        """
        # hacky but normal variable setting not possible with frozen enabled
        # https://stackoverflow.com/questions/57893902/how-can-i-set-an-attribute-in-a-frozen-dataclass-custom-init-method
        if cpg_1.index < cpg_2.index:
            object.__setattr__(self, "cpg_index_lowest", cpg_1)
            object.__setattr__(self, "cpg_index_highest", cpg_2)
        else:
            object.__setattr__(self, "cpg_index_lowest", cpg_2)
            object.__setattr__(self, "cpg_index_highest", cpg_1)

    def __lt__(self, other):
        if self.cpg_index_lowest != other.cpg_index_lowest:
            return self.cpg_index_lowest < other.cpg_index_lowest
        return self.cpg_index_highest < other.cpg_index_highest


class CpgNetworkStructure:
    """
    Describes the structure of a CPG network.

    Can generate parameters for a CPG network, such as the initial state.
    """

    cpgs: list[Cpg]
    connections: set[CpgPair]

    def __init__(self, cpgs: list[Cpg], connections: set[CpgPair]) -> None:
        """
        Initialize this object.

        :param cpgs: The CPGs used in the structure.
        :param connections: The connections between CPGs.
        """
        assert isinstance(connections, set)

        self.cpgs = cpgs
        self.connections = connections

    @staticmethod
    def make_cpgs(num_cpgs: int) -> list[Cpg]:
        """
        Create a list of CPGs.

        :param num_cpgs: The number of CPGs to create.
        :returns: The created list of CPGs.
        """
        return [Cpg(index) for index in range(num_cpgs)]

    def make_connection_weights_matrix(
        self,
        internal_connection_weights: dict[Cpg, float],
        external_connection_weights: dict[CpgPair, float],
    ) -> npt.NDArray[np.float_]:
        """
        Create a weight matrix from internal and external weights.

        :param internal_connection_weights: The internal weights.
        :param external_connection_weights: The external weights.
        :returns: The created matrix.
        """
        state_size = self.num_cpgs * 2

        assert set(internal_connection_weights.keys()) == set(self.cpgs)
        assert set(external_connection_weights.keys()) == self.connections

        weight_matrix = np.zeros((state_size, state_size))

        for cpg, weight in internal_connection_weights.items():
            weight_matrix[cpg.index][self.num_cpgs + cpg.index] = weight
            weight_matrix[self.num_cpgs + cpg.index][cpg.index] = -weight

        for cpg_pair, weight in external_connection_weights.items():
            weight_matrix[cpg_pair.cpg_index_lowest.index][
                cpg_pair.cpg_index_highest.index
            ] = weight
            weight_matrix[cpg_pair.cpg_index_highest.index][
                cpg_pair.cpg_index_lowest.index
            ] = -weight

        return weight_matrix

    @property
    def num_connections(self) -> int:
        """
        Get the number of connections in the structure.

        :returns: The number of connections.
        """
        return len(self.cpgs) + len(self.connections)

    def make_connection_weights_matrix_from_params(
        self, params: list[float]
    ) -> npt.NDArray[np.float_]:
        """
        Create a connection weights matrix from a list if connections.

        :param params: The connections to create the matrix from.
        :returns: The created matrix.
        """
        assert len(params) == self.num_connections

        # a = ",".join(f"{c.index}" for c in self.cpgs)
        # b = ",".join(f"{pair.cpg_index_lowest.index} -> {pair.cpg_index_highest.index}"
        #              for pair in self.connections)
        # print(f"[kgd-debug] make matrix: {a}; {b}")

        internal_connection_weights = {
            cpg: weight for cpg, weight in zip(sorted(self.cpgs), params[: self.num_cpgs])
        }

        external_connection_weights = {
            pair: weight
            for pair, weight in zip(sorted(self.connections), params[self.num_cpgs :])
        }

        return self.make_connection_weights_matrix(
            internal_connection_weights, external_connection_weights
        )

    @property
    def num_states(self) -> int:
        """
        Get the number of states in a cpg network of this structure.

        This would be twice the number of CPGs.

        :returns: The number of states.
        """
        return len(self.cpgs) * 2

    def make_uniform_state(self, value: float) -> npt.NDArray[np.float_]:
        """
        Make a state array by repeating the same value.

        Will match the required number of states in this structure.

        :param value: The value to use for all states
        :returns: The array of states.
        """
        _half = self.num_states // 2
        mask = np.hstack([np.full(_half, 1), np.full(_half, -1)])

        return mask * value

    @property
    def num_cpgs(self) -> int:
        """
        Get the number of CPGs in the structure.

        :returns: The number of CPGs.
        """
        return len(self.cpgs)

    @property
    def output_indices(self) -> list[int]:
        """
        Get the index in the state array for each cpg, matching the order the CPGs were provided in.

        :returns: The indices.
        """
        return [i for i in range(self.num_cpgs)]
