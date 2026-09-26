# HW1: Path Planning with A* and RRT*

This assignment implements two complementary path-planning methods:

- **A\*** searches a discretized map using `f(n) = g(n) + h(n)` with Euclidean-distance heuristic.
- **RRT\*** incrementally builds a collision-free tree, selects lower-cost parents, and rewires nearby nodes to improve path cost.

## My implementation

- [`a_star_implementation.py`](src/a_star_implementation.py): open-set selection, neighbor expansion, cost update, and parent-based path reconstruction.
- [`rrt_star_implementation.py`](src/rrt_star_implementation.py): random sampling, nearest-node extension, collision checks, best-parent selection, rewiring, and goal connection.

The files depend on the course planning framework, which is intentionally not copied into this portfolio repository.

## Report

[Open PDF](https://anyi-lee.github.io/robotic-navigation-and-exploration/hw1-path-planning/report/hw1-report.pdf)
