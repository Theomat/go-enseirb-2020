from typing import Tuple, Optional

import numpy as np
import torch

vget_visits = np.vectorize(lambda x: x.visits, otypes=[np.float])
vget_base_incertitudes = np.vectorize(lambda x: x.incertitude)
vget_action_values = np.vectorize(lambda x: x.current_action_value)


class Edge:
    def __init__(self, action: int, parent, prior: float = 0, child=None):
        self.visits: int = 0
        self.prior: float = prior
        self.sum_action_values: float = 0
        self.current_action_value: float = 0
        self.incertitude: float = prior

        self.parent = parent
        self.child = child
        self.action: int = action

    def backup(self, value: float, child_depth: int):
        self.sum_action_values += value
        self.visits += 1
        self.current_action_value = self.sum_action_values / self.visits
        self.incertitude = self.prior / (1 + self.visits)
        if self.parent:
            self.parent.backup(value, child_depth)

    def free(self):
        del self.parent
        del self.child


class Node:
    def __init__(self, state, inbound: Edge):
        self.state = state
        self.inbound: Edge = inbound
        self.children: Optional[np.ndarray] = None
        self.depth: int = 0
        self._max_child_depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def select(self, cpuct: float = 1.0):
        """
        Returns a Node
        """
        if self.is_leaf:
            return self
        if self.inbound:
            # I have as much visits as the sum of the visits of my children
            total_visits: float = np.sqrt(self.inbound.visits)
        else:
            total_visits: float = np.sqrt(np.sum(vget_visits(self.children)))
        values: np.ndarray = vget_base_incertitudes(self.children) * total_visits * cpuct
        values += vget_action_values(self.children)
        index: int = np.argmax(values)
        selected_edge: Edge = self.children[index]
        return selected_edge.child.select(cpuct)

    def expand(self, actions, states, priors, value):
        self.children: np.ndarray = np.zeros(len(actions), dtype=Edge)
        index: int = 0
        for action, state in zip(actions, states):
            # Little trick: PASS is -1 which maps to index 81, nice !
            edge: Edge = Edge(action, parent=self, prior=priors[action])
            edge.child = Node(state, edge)
            self.children[index] = edge
            index += 1
        self.backup(value, 1)

    def backup(self, value: float, child_depth: int = 0):
        if child_depth > self._max_child_depth:
            self._max_child_depth = child_depth
            self.depth = 1 + self._max_child_depth
        if self.inbound:
            self.inbound.backup(value, self.depth)

    def play(self, temperature: float = 1.0) -> Tuple[Edge, torch.FloatTensor]:
        array: np.ndarray = torch.zeros(82, dtype=torch.float32)
        if temperature <= 0:
            # If temperature is 0 then choose the node with the most visits
            # having a temperature of 10**(-5) creates overflow very easily
            visits: np.ndarray = vget_visits(self.children)
            chosen: np.ndarray = np.where(visits == np.max(visits))[0]
            for i in range(chosen.shape[0]):
                index: int = chosen[i]
                child: Edge = self.children[index]
                array[child.action] = 1 / chosen.shape[0]
            chosen_child_index: int = np.random.choice(chosen)
            return self.children[chosen_child_index], array
        else:
            probabilities: np.ndarray = vget_visits(self.children)
            np.power(probabilities, 1 / temperature, out=probabilities)
            probabilities /= np.sum(probabilities)

            index: int = np.random.choice(np.arange(probabilities.shape[0]), p=probabilities)
            for i in range(self.children.shape[0]):
                child: Edge = self.children[i]
                array[child.action] = probabilities[i]

            return self.children[index], array

    def add_dirichlet_noise(self, alpha: float = .03, epsilon: float = .25):
        if self.is_leaf:
            return

        noise: np.ndarray = np.random.dirichlet([alpha] * self.children.shape[0])
        for i in range(self.children.shape[0]):
            child: Edge = self.children[i]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def should_resign(self, v_resign: float) -> bool:
        if self.inbound and self.inbound.current_action_value < v_resign:
            if np.max(vget_action_values(self.children)) < v_resign:
                return True
        return False

    def free_except(self, keep):
        """
        Delete everything to be GC except the specified Node **keep**.
        """
        if self.inbound:
            self.inbound.free()
        if not self.is_leaf:
            for i in range(self.children.shape[0]):
                child = self.children[i]
                if child.child != keep:
                    child.child.free_except(None)
                else:
                    child.parent = None
        del self.state
        del self.inbound
        del self.children
