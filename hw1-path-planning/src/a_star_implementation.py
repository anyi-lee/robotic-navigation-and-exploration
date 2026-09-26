import cv2
import numpy as np

from path_planning import *
from path_planning.a_star_planner import AStarPlanner


class AStarImplementation(AStarPlanner):
    def preloop(self):
        self.queue: set[PathNode] = set()
        self.closed_set: set[PathNode] = set()
        self.g: dict[PathNode, float] = {}
        self.h: dict[PathNode, float] = {}
        self.visited_nodes: set[PathNode] = set()

        self.start_node.cost = 0.0
        self.g[self.start_node] = 0.0
        self.h[self.start_node] = calculate_node_distance(
            self.start_node,
            self.goal_node
        )
        self.queue.add(self.start_node)
        self.visited_nodes.add(self.start_node)

    def step(self):
        if not self.queue:
            self.is_done.set()
            return

        current_node = min(
            self.queue,
            key=lambda node: self.g[node] + self.h[node]
        )
        self.queue.remove(current_node)
        self.closed_set.add(current_node)

        if calculate_node_distance(current_node, self.goal_node) <= self.goal_threshold:
            self.goal_node.parent = current_node
            self.goal_node.cost = current_node.cost + calculate_node_distance(
                current_node,
                self.goal_node
            )
            self.visited_nodes.add(self.goal_node)
            self.is_done.set()
            return

        neighbor_nodes = self.get_neighbor_nodes(current_node)

        for neighbor_node in neighbor_nodes:
            if neighbor_node in self.closed_set:
                continue

            tentative_g = self.g[current_node] + calculate_node_distance(
                current_node,
                neighbor_node
            )

            if neighbor_node not in self.g or tentative_g < self.g[neighbor_node]:
                neighbor_node.parent = current_node
                neighbor_node.cost = tentative_g
                self.g[neighbor_node] = tentative_g
                self.h[neighbor_node] = calculate_node_distance(
                    neighbor_node,
                    self.goal_node
                )
                self.queue.add(neighbor_node)
                self.visited_nodes.add(neighbor_node)

    def postloop(self):
        if self.goal_node.parent is None:
            return ([self.start_node], self.visited_nodes)

        return (
            collect_path(self.goal_node),
            self.visited_nodes
        )