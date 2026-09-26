import cv2
import numpy as np

from path_planning import *
from path_planning.rrt_star_planner import RRTStarPlanner
from path_planning.primitives import PixelCoordinates
from path_planning.planner_utils import (
    check_inside_map,
    check_collision_free,
    calculate_node_distance,
    collect_path,
)


class RRTStarImplementation(RRTStarPlanner):
    def preloop(self):
        self.visited_nodes: set[PathNode] = set()
        self.visited_nodes.add(self.start_node)
        self.start_node.cost = 0.0

    def _nearest_node(self, target_node: PathNode) -> PathNode:
        return min(
            self.visited_nodes,
            key=lambda node: calculate_node_distance(node, target_node)
        )

    def _steer(self, source_node: PathNode, target_node: PathNode) -> PathNode:
        distance = calculate_node_distance(source_node, target_node)
        if distance == 0:
            return PathNode(
                coordinates=source_node.coordinates,
                parent=source_node,
                cost=source_node.cost
            )

        if distance <= self.step_size:
            new_coordinates = target_node.coordinates
        else:
            dx = target_node.coordinates.x - source_node.coordinates.x
            dy = target_node.coordinates.y - source_node.coordinates.y
            scale = self.step_size / distance
            new_x = source_node.coordinates.x + dx * scale
            new_y = source_node.coordinates.y + dy * scale
            new_coordinates = PixelCoordinates(new_x, new_y)

        new_node = PathNode(coordinates=new_coordinates)
        return new_node

    def _near_nodes(self, new_node: PathNode) -> list[PathNode]:
        near_nodes = []
        for node in self.visited_nodes:
            if calculate_node_distance(node, new_node) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def step(self):
        random_node = self.sample_random_node()

        nearest_node = self._nearest_node(random_node)
        new_node = self._steer(nearest_node, random_node)

        if new_node.coordinates == nearest_node.coordinates:
            return

        if not check_inside_map(self.occupancy_map, new_node):
            return

        if not check_collision_free(self.occupancy_map, nearest_node, new_node):
            return

        near_nodes = self._near_nodes(new_node)

        best_parent = nearest_node
        best_cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)

        for near_node in near_nodes:
            if not check_collision_free(self.occupancy_map, near_node, new_node):
                continue
            candidate_cost = near_node.cost + calculate_node_distance(near_node, new_node)
            if candidate_cost < best_cost:
                best_parent = near_node
                best_cost = candidate_cost

        new_node.parent = best_parent
        new_node.cost = best_cost
        self.visited_nodes.add(new_node)

        for near_node in near_nodes:
            if near_node == self.start_node:
                continue
            if not check_collision_free(self.occupancy_map, new_node, near_node):
                continue

            rewired_cost = new_node.cost + calculate_node_distance(new_node, near_node)
            if rewired_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = rewired_cost

        if calculate_node_distance(new_node, self.goal_node) <= self.goal_threshold:
            if check_collision_free(self.occupancy_map, new_node, self.goal_node):
                self.goal_node.parent = new_node
                self.goal_node.cost = new_node.cost + calculate_node_distance(
                    new_node,
                    self.goal_node
                )
                self.visited_nodes.add(self.goal_node)
                self.is_done.set()

    def postloop(self):
        if self.goal_node.parent is None:
            return ([self.start_node], self.visited_nodes)

        return (
            collect_path(self.goal_node),
            self.visited_nodes
        )