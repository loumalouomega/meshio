---
title: Proximity graphs
description: Radius and k-nearest-neighbour graphs over a mesh's points, with a periodic minimum-image mode, world edges alongside the mesh edges, and bistride multiscale coarsening.
---

# Proximity graphs

[`edge_index`](ml.md#the-mesh-as-a-graph-edge_index) builds a graph out of a mesh's *connectivity* — the cell edges, or the facet-sharing cell dual. For a particle method there is no connectivity to build one from: a smoothed-particle or discrete-element state is a cloud of positions, and what makes two particles interact is the interaction radius, not a shared element. A graph network trained on such a state needs its neighbourhoods recomputed from geometry at every step, and that is what `proximity_graph` is.

It also covers the two structures a *mesh* graph needs that connectivity cannot supply. **World edges** are a proximity graph laid alongside the mesh edges, so a model can see two surfaces that touch without being connected — contact, self-collision, a fluid meeting a wall. A **bistride hierarchy** is the multiscale pooling a deep graph network needs so information crosses the mesh in a few message-passing steps instead of one element per step.

Everything here is pure Python over a bucket grid, the same spatial hash [`interpolate`](interpolate.md) and [`merge`](merge.md) already use, vectorized here because a proximity graph queries every point rather than a few. There is no SciPy, no KD-tree and no cell-list library; the C++ core, the WASM build and every binding are untouched.

```python
import meshioplusplus as mio

cloud = mio.read("particles.vtu")

edges = mio.proximity_graph(cloud, radius=0.015)                     # (2, E) int64
edges = mio.proximity_graph(cloud, method="knn", max_neighbors=16)   # near-constant degree
edges = mio.proximity_graph(cloud, radius=0.015, box_size=[2.0, 2.0, 2.0])   # periodic

attributes = mio.edge_vectors(cloud.points, edges)   # (E, 4): displacement, then its norm
```

## Two rules, and when each is right

`method="radius"` links every pair closer than `radius`. It is the physical rule — an interaction cutoff is a distance — and its degree varies with local density, which is what a model should see when the density is part of the physics. A radius a few mean spacings wide is the intended operating point; one ten times that asks for a thousand times the edges.

`method="knn"` links each point to its `max_neighbors` nearest and then symmetrizes, so an edge exists whenever *either* endpoint chose the other. Degree is then near-constant, and a point in a sparse region still has neighbours where a radius rule would leave it isolated. `max_neighbors` is clamped to `N - 1`.

Both are exact. Neither approximates to go faster, and neither is seeded: the answer is a function of the positions alone.

## The output is the same object `edge_index` returns

`(2, E)` C-contiguous int64, both directions of every edge, lexicographically sorted by (source, target), de-duplicated, with no self edges. `undirected=False` emits each edge once with `source < target` instead — half the array, and the form to write to disk.

That layout is not merely similar to [`edge_index`](ml.md#the-mesh-as-a-graph-edge_index)'s, it is *the same code*: both call one internal canonicalization, so a mesh graph and a proximity graph over the same pair set are byte-identical. A model that consumes one consumes the other.

`kind="cell"` puts the vertices at the block-major cell centroids instead of the mesh points, exactly as it does for `edge_index`, so a cell-located sample stays coherent.

## Periodic boxes

`box_size` — one value, or one per axis — turns on the **minimum image** convention: a pair is linked by the shortest of its images across the box's faces, so two particles either side of a boundary are neighbours and the vector between them is short. This is the difference between a periodic simulation's graph and a wrong one.

```python
points = [[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]]

mio.proximity_graph(points, radius=0.05)                  # no edges: they are 0.98 apart
mio.proximity_graph(points, radius=0.05, box_size=1.0)    # one edge: they are 0.02 apart
mio.edge_vectors(points, edges, box_size=1.0)             # dx = 0.02, never 0.98
```

Positions outside the box are taken modulo it, so an unwrapped trajectory needs no preparation. The same convention is applied to the search and to `edge_vectors`, from one shared line.

A radius larger than half the smallest box side is **refused by name** rather than approximated. Past that, a pair has two images at the same distance and "the" minimum image does not exist; returning one of them would be a silent choice with no defensible basis.

## What it costs

Both searches are `O(N + E)` in the bucket grid, and `E` is what dominates: a radius graph's edge count grows as the cube of the radius, so the honest unit is edges per second rather than points. Measured on one core over a uniform cloud at unit density:

| points | method | edges | time |
| --- | --- | --- | --- |
| 50 000 | radius | 3.3 M | 1.3 s |
| 50 000 | knn (k=16) | 0.9 M | 4.5 s |
| 200 000 | radius | 13.4 M | 6.4 s |
| 200 000 | knn (k=16) | 3.6 M | 23.0 s |

`knn` is dearer per edge because it must *rank* every candidate rather than threshold it. At or below a couple of thousand points a plain `O(N²)` route runs instead — the same answer by a shorter road, and the oracle the bucket path is tested against on every run of the suite.

The bucket size is an implementation detail that **cannot** change the answer, which is why it can be retuned on measurement alone; a test pins that output is identical across a sixteen-fold range of it.

## Edge features

`edge_vectors(points, edge_index, box_size=None)` returns `(E, d + 1)`: the displacement `points[row] - points[col]` — source minus destination, the stable upstream convention — followed by its Euclidean length. It is the same function [`graph_sample`](physicsnemo.md#graph-samples) already used for mesh edges, so the two agree by construction rather than by transcription.

## Bistride multiscale hierarchies

A message-passing network moves information one element per layer. On a mesh with a hundred thousand elements across, a pressure signal that must reach the far side needs either a hundred thousand layers or a coarser graph to travel on. `bistride_hierarchy` builds the coarser graphs, following Cao et al.'s BSMS construction:

```python
hierarchy = mio.bistride_hierarchy(edges, mesh.points, num_levels=3)

hierarchy.edges      # 4 edge sets, each (2, E_i) in its OWN level's numbering
hierarchy.ids        # 3 arrays: which rows of level i survive into level i + 1
hierarchy.schema     # {"version", "num_levels", "num_nodes": [N_0 .. N_3]}
```

Each level two-colours every connected component by breadth-first depth from the node nearest its centroid, keeps the smaller colour class, and joins the survivors wherever they were within two hops. A path of `N` nodes becomes a path of about `N / 2`, so a message crosses the mesh in logarithmically many steps.

Two details are load-bearing. The breadth-first search is seeded at the node nearest the component's **centroid**, so the two classes are the geometric checkerboard rather than an artefact of the node numbering. And the two-hop join is over the adjacency **plus self-loops**: without them a survivor loses the neighbours that were dropped instead of inheriting their reach, and the coarse graph falls apart rather than coarsening. A test pins that against the definition, on a fixture chosen so the two formulations disagree.

`edges` has one more entry than `ids`, deliberately: every level has edges, every *transition* has a pooling map. A level that would collapse below two nodes raises by name — that is a request for too many levels, not a graph to hand a model.

## In a training sample

[`graph_sample`](physicsnemo.md#graph-samples) takes both structures as dictionaries in this page's vocabulary:

```python
import meshioplusplus.physicsnemo as mpn

# A particle state: positions, no connectivity. Proximity REPLACES the mesh edges.
sample = mpn.graph_sample(cloud, fields=["v"], proximity={"method": "radius", "radius": 0.015})

# A mesh, plus contact edges. World edges are ADDED beside the mesh edges.
sample = mpn.graph_sample(mesh, fields=["u"], world_edges={"method": "knn", "max_neighbors": 8})
sample.arrays["edge_index"], sample.arrays["world_edge_index"]
```

The two edge sets are kept as **separate arrays** rather than concatenated. PyTorch Geometric increments any attribute whose name contains `index` when it batches, so two sets batch correctly for free, whereas a concatenated one could not be split apart again afterwards. A model wanting the single edge set HybridMeshGraphNet expects concatenates them itself, **mesh edges first** — that model splits its edge features positionally, so the order is part of its contract, not a preference.

How the edges were built is recorded in the sample's schema, and `graph_sample_version` moved to 3 to say so. A schema stored under an earlier release now compares unequal, which is the drift guard doing its job: a checkpoint trained on a proximity graph must not be replayed on a mesh one.

`edge_stats` takes the same two arguments, and **must be given the same ones training will use**. It rebuilds the edges to gather its normalization statistics, and statistics gathered over a different graph normalize the wrong thing with nothing downstream to report the discrepancy. Both go through one shared builder so they cannot drift.

## In a training spec

A [`TrainSpec`](physicsnemo.md#the-training-spec) carries the neighbourhood in its `Graph` block:

```json
{
  "Graph": {
    "Proximity": {"Method": "radius", "Radius": 0.015, "BoxSize": [2.0, 2.0, 2.0]}
  }
}
```

`Method` is `"radius"` or `"knn"`; the other method's key is refused rather than ignored, as is a malformed one — when the document is read, not an epoch later. The block is written back only when it is set, so a spec that does not use the feature is byte-for-byte what it was before.

There is deliberately **no** `Graph.WorldEdges` and no bistride block. No shipped model family reads a second edge set or a hierarchy, and a key a run would silently ignore is exactly what the spec's strict unknown-key refusal exists to prevent. Both are available from Python, which is where a model that consumes them lives today.

## CLI and MCP

```bash
meshioplusplus proximity-graph particles.vtu graph.vtu --radius 0.015
meshioplusplus proximity-graph mesh.vtu graph.vtu --knn 8 --box 2.0 --kind cell
```

The verb writes the graph as `line` cells over the same positions, with a `degree` point array, and prints the vertex and edge counts, the degree spread and how many vertices came out isolated — the numbers that say whether the radius was well chosen. The `proximity_graph` MCP tool takes the same arguments and returns those numbers as its report.
