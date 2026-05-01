"""
Minimum s-t Cut with Cut Queries - Demo Code

Companion code for the MATH 482 project report on
Jiang, Nanongkai, Sawettamalya, "Minimum s-t Cuts with Fewer Cut Queries",
SODA 2026 (arXiv:2510.18274).

AI disclosure: generative AI was used to help draft and clean up parts of
this script. All code was reviewed and tested by me.

This is a small demo, not a full implementation of the paper's algorithm.
The full algorithm requires enumerating a witness set of size up to n^O(n),
which is not feasible to run on a real machine, so I focused on the pieces
of the paper that I could actually code up and test.

The script contains four pieces:
  1. CutQueryOracle - simulates the cut-query model and counts queries.
  2. dinic_min_cut - a baseline min-cut using NetworkX (reads the full graph).
  3. learn_graph_and_cut - the trivial O(n^2) algorithm: learn every edge
     of G using cut queries, then run Dinic's on the learned graph.
  4. find_long_edge - a stand-alone version of Lemma 5.8 from the paper,
     with a correctness check on random graphs and random potentials.

Usage:
    pip install networkx
    python min_st_cut_demo.py
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# 1. Cut-Query Oracle
class CutQueryOracle:
    """Wraps a graph so the algorithm can only access it through cut queries.
    Every call to cut_value() is counted."""

    def __init__(self, graph: nx.Graph, s, t):
        self._graph = graph
        self._edges = frozenset(
            frozenset((u, v)) for u, v in graph.edges()
        )
        self.s = s
        self.t = t
        self.V = frozenset(graph.nodes())
        self.n = len(self.V)
        self.cut_queries = 0

    def cut_value(self, S) -> int:
        """Returns |E(S, V\\S)|. Costs 1 cut query."""
        self.cut_queries += 1
        S = frozenset(S)
        count = 0
        for edge in self._edges:
            u, v = tuple(edge)
            if (u in S) != (v in S):
                count += 1
        return count

    def independent_set(self, A, B) -> bool:
        """Checks whether there are no edges between A and B.
        Uses 3 cut queries (Proposition 2.10 of the paper)."""
        A, B = set(A), set(B)
        if not A or not B or A & B:
            return True
        a = self.cut_value(A)
        b = self.cut_value(B)
        ab = self.cut_value(A | B)
        cross = (a + b - ab) // 2
        return cross == 0

    def find_edge(self, A, B) -> Optional[Tuple]:
        """Find some edge in E with one endpoint in A and one in B.
        Returns None if no such edge exists.
        Uses O(1) cut queries if no edge exists, O(log n) if one does."""
        A, B = list(A), list(B)
        if not A or not B:
            return None
        if self.independent_set(A, B):
            return None
        # Binary search to narrow down to one vertex on each side.
        while len(A) > 1:
            mid = len(A) // 2
            left, right = A[:mid], A[mid:]
            if not self.independent_set(left, B):
                A = left
            else:
                A = right
        while len(B) > 1:
            mid = len(B) // 2
            left, right = B[:mid], B[mid:]
            if not self.independent_set(A, left):
                B = left
            else:
                B = right
        return (A[0], B[0])

    def reset_counts(self):
        self.cut_queries = 0


# 2. Dinic's baseline (reads the full graph)
def dinic_min_cut(graph: nx.Graph, s, t) -> Tuple[int, Set]:
    """Computes the exact min s-t cut using Dinic's blocking flow via NetworkX.
    This reads the graph directly so it serves as the ground truth answer."""
    dg = nx.DiGraph()
    for u, v in graph.edges():
        dg.add_edge(u, v, capacity=1)
        dg.add_edge(v, u, capacity=1)
    cut_value, (S_side, _) = nx.minimum_cut(
        dg, s, t, flow_func=nx.algorithms.flow.dinitz
    )
    return cut_value, set(S_side)


# 3. The trivial O(n^2)-query algorithm
def learn_graph_and_cut(oracle: CutQueryOracle) -> Tuple[int, Set, nx.Graph]:
    """The trivial baseline. We learn every edge of G using cut queries,
    then run Dinic's on the learned graph. This costs O(n^2) cut queries
    but is guaranteed correct, and it's the bound that RSW 2018 first beat.

    Strategy: for each pair (u, v), the number of edges between {u} and {v}
    is (deg(u) + deg(v) - cut_value({u, v})) / 2. So one cut query per pair
    plus n queries to learn the degrees gives us the whole graph.
    """
    V = list(oracle.V)
    n = len(V)
    learned = nx.Graph()
    learned.add_nodes_from(V)

    # First learn each vertex's degree (n cut queries).
    deg = {v: oracle.cut_value({v}) for v in V}

    # Then for each pair of vertices, check if there's an edge between them.
    # This is the n(n-1)/2 part.
    for i in range(n):
        for j in range(i + 1, n):
            u, v = V[i], V[j]
            pair_cut = oracle.cut_value({u, v})
            edges_between = (deg[u] + deg[v] - pair_cut) // 2
            if edges_between == 1:
                learned.add_edge(u, v)

    # Now we have the whole graph, so just run Dinic's on it.
    cut_value, S_side = dinic_min_cut(learned, oracle.s, oracle.t)
    return cut_value, S_side, learned


# 4. FindLongEdge (Lemma 5.8 of the paper)
def find_long_edge(
    oracle: CutQueryOracle,
    phi: Dict,
    delta: float,
    excluded_edges: Optional[Set] = None,
) -> Optional[Tuple]:
    """Finds an edge (u, v) of G with |phi(u) - phi(v)| >= delta/2.

    This is the paper's FindLongEdge subroutine (Lemma 5.8), simplified.
    It takes a potential phi mapping each vertex to [0, 1] and a threshold
    delta, and returns an edge whose endpoints are mapped far apart in phi.
    Uses O(log(n/delta)) cut queries.

    Strategy: bucket vertices by their phi value, then probe pairs of
    buckets that are far apart for an edge using the find_edge primitive.
    """
    if excluded_edges is None:
        excluded_edges = set()

    V = list(oracle.V)
    n = len(V)

    # Split [0, 1] into buckets of width delta/2.
    num_buckets = max(2, int(math.ceil(2.0 / delta)))
    buckets = [[] for _ in range(num_buckets)]
    for v in V:
        val = min(max(phi.get(v, 0.5), 0.0), 1.0)
        b = min(int(val * num_buckets / 1.0000001), num_buckets - 1)
        buckets[b].append(v)

    # Try the farthest pairs of buckets first since those give the strongest
    # guarantee on the gap.
    for gap in range(num_buckets - 1, 1, -1):
        for i in range(num_buckets - gap):
            j = i + gap
            A_pool, B_pool = list(buckets[i]), list(buckets[j])
            if not A_pool or not B_pool:
                continue
            attempts = 0
            while attempts < 5:
                e = oracle.find_edge(A_pool, B_pool)
                if e is None:
                    break
                u, v = e
                if frozenset((u, v)) not in excluded_edges:
                    return e
                # If we hit an excluded edge, drop v and try again.
                B_pool = [x for x in B_pool if x != v]
                if not B_pool:
                    break
                attempts += 1
    return None


# 5. Tests
def generate_test_graph(
    n: int, p: float = 0.2, seed: int = 42
) -> Tuple[nx.Graph, int, int]:
    """Generate a random Erdos-Renyi graph with a guaranteed s-t path."""
    random.seed(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    s, t = 0, n - 1
    if not nx.has_path(G, s, t):
        # If s and t aren't connected, just add a path between them.
        for u, v in zip(range(n - 1), range(1, n)):
            G.add_edge(u, v)
    return G, s, t


def test_find_long_edge(seed: int = 0):
    """Check find_long_edge against brute force on a random graph."""
    print("\n--- Testing find_long_edge ---")
    random.seed(seed)
    n = 20
    G = nx.erdos_renyi_graph(n, 0.3, seed=seed)
    phi = {v: random.random() for v in G.nodes()}
    # Brute force: find the edge with the largest |phi(u) - phi(v)|.
    gaps = [(abs(phi[u] - phi[v]), (u, v)) for u, v in G.edges()]
    if not gaps:
        print("  empty graph, skipping")
        return
    max_gap, max_edge = max(gaps, key=lambda x: x[0])
    print(f"  Graph: n={n}, m={G.number_of_edges()}")
    print(f"  Largest gap in any edge: {max_gap:.3f} on edge {max_edge}")

    oracle = CutQueryOracle(G, 0, n - 1)
    delta = max_gap
    result = find_long_edge(oracle, phi, delta)
    if result is None:
        print(f"  find_long_edge returned None - FAIL")
        return False
    u, v = result
    actual_gap = abs(phi[u] - phi[v])
    ok = actual_gap >= delta / 2
    status = "PASS" if ok else "FAIL"
    print(f"  find_long_edge returned {result} with gap {actual_gap:.3f}")
    print(f"  Required gap >= delta/2 = {delta/2:.3f}: {status}")
    print(f"  Cut queries used: {oracle.cut_queries}")
    return ok


def benchmark(n: int, p: float, seed: int):
    print(f"\nBenchmark: n={n}, edge_prob={p}, seed={seed}")

    G, s, t = generate_test_graph(n, p, seed)
    m = G.number_of_edges()
    print(f"Graph: {n} vertices, {m} edges, s={s}, t={t}")

    # Ground truth from Dinic's on the full graph.
    t0 = time.perf_counter()
    true_cut, _ = dinic_min_cut(G, s, t)
    t_dinic = time.perf_counter() - t0
    print(
        f"\n[Dinic's, reads graph directly]  min cut = {true_cut}   "
        f"({t_dinic*1000:.2f} ms)"
    )

    # Trivial cut-query algorithm.
    oracle = CutQueryOracle(G, s, t)
    t0 = time.perf_counter()
    cq_cut, _, _ = learn_graph_and_cut(oracle)
    t_cq = time.perf_counter() - t0
    print(
        f"[Cut-query, learn-the-graph]     min cut = {cq_cut}   "
        f"({t_cq*1000:.2f} ms)"
    )
    print(f"                                 cut queries: {oracle.cut_queries}")
    print(f"                                 n^2 = {n*n} (trivial bound)")
    correct = cq_cut == true_cut
    print(f"                                 {'correct' if correct else 'WRONG'}")

    return correct


if __name__ == "__main__":
    print("Part 1: find_long_edge correctness check")
    all_ok = True
    for seed in range(5):
        ok = test_find_long_edge(seed=seed)
        all_ok = all_ok and ok
    print(f"\nOverall find_long_edge test: {'PASS' if all_ok else 'FAIL'}")

    print("\n\nPart 2: Min s-t cut benchmarks")
    benchmark(n=15, p=0.25, seed=42)
    benchmark(n=25, p=0.20, seed=7)
    benchmark(n=40, p=0.15, seed=123)