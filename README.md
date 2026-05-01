Minimum s-t Cut with Cut Queries — Demo Code
Companion code for a MATH 482 (Combinatorial Algorithms) project at CSU Northridge on the paper:

Jiang, Nanongkai, and Sawettamalya. Minimum s–t Cuts with Fewer Cut Queries. SODA 2026. arXiv:2510.18274

What this is
A small Python script (about 300 lines) that demonstrates the cut-query model and implements the pieces of the paper that are tractable to code and test directly.
This is not a full implementation of the paper's Õ(n^(8/5)) algorithm. The full algorithm requires enumerating a witness set of size up to n^O(n), which is not feasible to run on a real machine.
What's in the script
The script contains four components:

CutQueryOracle — wraps a NetworkX graph and exposes only one primitive, cut_value(S), while counting every call. This enforces the cut-query model: any algorithm built on top of the oracle has no way to "see" the graph other than by asking about cut sizes.
dinic_min_cut — a Dinic's blocking-flow baseline using NetworkX. Reads the graph directly and produces the true minimum cut as ground truth.
learn_graph_and_cut — the trivial O(n²)-query reconstruction algorithm. Uses the identity |E ∩ ({u} × {v})| = (deg(u) + deg(v) − cut({u, v})) / 2 to detect each edge, then runs Dinic's on the reconstructed graph. This is the baseline that RSW 2018 first improved upon.
find_long_edge — a stand-alone implementation of FindLongEdge (Lemma 5.8 of the paper). Takes a potential φ : V → [0, 1] and threshold δ and returns an edge with |φ(u) − φ(v)| ≥ δ/2.

How to run
pip install networkx
python min_st_cut_demo.py
The script runs two parts:

Part 1: correctness check on find_long_edge across five random seeds
Part 2: benchmarks comparing Dinic's vs. the trivial cut-query algorithm at n = 15, 25, and 40 (hard coded sizes) 

Test data is randomly generated Erdős–Rényi graphs with fixed seeds for reproducibility.
Sample output
On n = 40, p = 0.15:

Dinic's (full graph access): finds min cut in under 2 ms
Trivial cut-query algorithm: 820 cut queries (against the n² = 1600 ceiling), correct min cut

find_long_edge consistently uses 15–21 cut queries per call across the five test seeds.
AI disclosure
Generative AI was used to help draft and clean up parts of this script. All code was reviewed and tested by me.
