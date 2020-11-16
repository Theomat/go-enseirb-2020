from typing import Tuple

import numpy as np

vget_visits = np.vectorize(lambda x: x.visits)
vget_base_incertitudes = np.vectorize(lambda x: x.base_incertitude())
vget_action_values = np.vectorize(lambda x: x.current_action_value)


class Edge:
    def __init__(self, action: int, parent, prior: float = 0, child=None):
        self.visits = 0
        self.prior = prior
        self.sum_action_values = 0
        self.current_action_value = 0

        self.parent = parent
        self.child = child
        self.action = action

    def backup(self, value: float):
        self.sum_action_values += value
        self.visits += 1
        self.current_action_value = self.sum_action_values / self.visits
        if self.parent:
            self.parent.backup(value)

    def base_incertitude(self):
        return self.prior / (1 + self.visits)

    def free(self):
        del self.parent
        if self.child:
            self.child.free_except(None)
        del self.child


class Node:
    def __init__(self, state, inbound: Edge):
        self.state = state
        self.inbound: Edge = inbound
        self.children: np.ndarray = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def select(self, cpuct: float = 1.0):
        """
        Returns a Node
        """
        if self.is_leaf:
            return self
        total_visits = np.sqrt(np.sum(vget_visits(self.children)))
        values = vget_base_incertitudes(self.children) * total_visits * cpuct
        values += vget_action_values(self.children)
        index = np.argmax(values)
        return self.children[index].child.select(cpuct)

    def expand(self, actions, states, priors, value):
        self.children = np.zeros(len(actions), dtype=Edge)
        index = 0
        for action, state in zip(actions, states):
            edge = Edge(action, parent=self, prior=priors[action])
            edge.child = Node(edge)
            self.children[index] = edge
            index += 1
        self.backup(value)

    def backup(self, value: float):
        if self.inbound:
            self.inbound.backup(value)

    def play(self, temperature: float = 1.0) -> Tuple[Edge, np.ndarray]:
        probabilities = vget_visits(self.children)
        np.power(probabilities, 1 / temperature, out=probabilities)
        probabilities /= np.sum(probabilities)
        index = np.random.choice(np.randrange(probabilities.shape[0]), size=1, replace=False, p=probabilities)
        array = np.zeros(82, dtype=np.float)
        for i in range(self.children.shape[0]):
            child = self.children[i]
            if child.action == -1:
                array[81] = probabilities[i]
            else:
                array[child.action] = probabilities[i]
        return self.children[index], array

    def add_dirichlet_noise(self, alpha: float = .03, epsilon: float = .25):
        noise = np.random.dirichlet(alpha, self.children.shape[0])
        for i in range(self.children.shape[0]):
            child = self.children[i]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def should_resign(self, v_resign: float) -> bool:
        if self.inbound.current_action_value < v_resign:
            if np.max(vget_action_values(self.children)) < v_resign:
                return True
        return False

    def free_except(self, keep):
        """
        Delete everythign to be GC except the specified Node **keep**.
        """
        del self.state
        del self.children
        if self.inbound:
            self.inbound.free()
        for i in range(self.children.shape[0]):
            child = self.children[i]
            if child.child != keep:
                child.free()
            else:
                child.parent = None
