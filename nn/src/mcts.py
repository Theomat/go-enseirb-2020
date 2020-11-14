from queue import PriorityQueue

from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class Edge:
    def __init__(self, action, parent, prior: float = 0):
        self.visits = 0
        self.prior = prior
        self.action_values = 0
        self.parent = parent
        self.child = None
        self.action = action
        self.closed = False

    def incertitude(self):
        if self.closed:
            return 0
        return self.prior / (1 + self.visits)

    def priority(self):
        return - (self.action_values + self.incertitude())


class Node:
    def __init__(self, inbound: Edge):
        self.inbound: Edge = inbound
        self.children = PriorityQueue()

    @property
    def is_leaf(self):
        return self.children.empty()

    @property
    def score(self):
        if self.inbound:
            return -self.inbound.priority()
        return .5

    def select_move(self, legal_moves):
        edge = self.children.get().item
        while edge.action not in legal_moves:
            edge = self.children.get().item
        edge.child.inbound = None
        edge.parent = None
        return edge.child, edge.action, edge.action_values, edge.incertitude()

    def select(self, actions=None):
        if self.is_leaf:
            return self, actions or []
        if actions is None:
            actions = []
        selected_edge = self.children.get().item
        actions.append(selected_edge.action)
        return selected_edge.child.select(actions)

    def expand(self, legal_actions, priors):
        for i, action in enumerate(legal_actions):
            edge = Edge(action, self, prior=priors[i])
            edge.child = Node(edge)
            self.children.put(PrioritizedItem(item=edge, priority=edge.priority()))

    def update(self, value, closed=False):
        edge = self.inbound
        if not edge:
            return
        edge.action_values = (edge.action_values * edge.visits + value) / (edge.visits + 1)
        edge.visits += 1
        edge.closed = closed
        if closed:
            edge.action_values = value
        node = edge.parent
        if node:
            node.children.put(PrioritizedItem(priority=edge.priority(), item=edge))
            node.update(value)

    def move_to(self, action):
        while not self.children.empty():
            edge = self.children.get().item
            if edge.action == action:
                print("Moved along action:", action)
                print("Score:", self.score)
                node = edge.child
                node.inbound = None
                edge.parent = None
                edge.child = None
                del self.children
                return node
        return None
