class Edge:
    def __init__(self, action, parent, turn, prior: float = 0):
        self.visits = 0
        self.prior = prior
        self.action_values = 0
        self.parent = parent
        self.child = None
        self.action = action
        self.closed = False
        self.coeff = -1 if turn else 1
        # turn == 1 => max || turn = 0 => min

    def incertitude(self):
        if self.closed:
            return 0
        return self.prior / (1 + self.visits)

    def priority(self):
        if self.closed:
            return -self.coeff * 10
        return self.coeff * (self.action_values + self.incertitude())


DEBUG = True


class Node:
    def __init__(self, inbound: Edge):
        self.inbound: Edge = inbound
        self.children = []
        self._best = None
        self._best_prio = 999

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def score(self):
        if self.inbound:
            return self.inbound.action_values
        return -1

    def __get_best(self):
        if self._best:
            return self._best
        self._best_prio = 9999
        for edge in self.children:
            prio = edge.priority()
            if prio < self._best_prio:
                self._best = edge
                self._best_prio = prio
        return self._best

    def select_move(self):
        best_edge = self.__get_best()
        if DEBUG:
            for edge in self.children:
                print(f"\t{edge.action} has value: {edge.action_values:.4f} ~ {edge.incertitude():.4f}",
                      f"(prior={edge.prior:.4f}, visits={edge.visits} closed={edge.closed})")

        best_edge.child.inbound = None
        best_edge.parent = None
        del self.children
        del self.inbound
        return best_edge.child, best_edge.action, best_edge.action_values, best_edge.incertitude()

    def select(self, actions=None):
        if self.is_leaf:
            return self, actions or []
        if actions is None:
            actions = []
        best_edge = self.__get_best()
        actions.append(best_edge.action)
        return best_edge.child.select(actions)

    def expand(self, legal_actions, priors, turn):
        for i, action in enumerate(legal_actions):
            edge = Edge(action, self, turn=turn, prior=priors[i])
            edge.child = Node(edge)
            self.children.append(edge)

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
            node.update(value)
            node._best = None

    def move_to(self, action):
        for edge in self.children:
            if edge.action == action:
                node = edge.child
                inbound = node.inbound
                if DEBUG:
                    print("Moved along action:", action)
                    print(f"Score:{inbound.action_values:.4f} ~ {inbound.incertitude():.4f}",
                          f"(prior={inbound.prior:.4f}, visits={inbound.visits})")
                node.inbound = None
                edge.parent = None
                edge.child = None
                del self.children
                return node
        return None
