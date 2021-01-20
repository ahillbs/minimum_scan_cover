from typing import Union, Optional, List, Dict, Tuple, Set, Any

class DependencyNode():
    def __init__(self, value, depends_on: Optional[List['DependencyNode']] = None):
        self.__depends_on = []
        self.__dependency_for = []
        if depends_on:
            for dependency in depends_on:
                self.add_dependency(dependency)
        self.value = value
    
    @property
    def depends_on(self):
        return self.__depends_on

    @property
    def dependency_for(self):
        return self.__dependency_for
    
    def add_dependency(self, dependency_node: Union['DependencyNode', List['DependencyNode']]):
        if not isinstance(dependency_node, list):
            dependency_node = [dependency_node]
        for node in dependency_node:
            self.__depends_on.append(node)
            node.__dependency_for.append(self)

    def remove_dependency(self, dependency_node: Union['DependencyNode', List['DependencyNode']]):
        if not isinstance(dependency_node, list):
            dependency_node = [dependency_node]
        for node in dependency_node:
            self.__depends_on.remove(node)
            node.__dependency_for.remove(self)

    def __eq__(self, other: 'DependencyNode'):
        if (isinstance(other, DependencyNode) and
                self.value == other.value and
                len(self.dependency_for) == len(other.dependency_for) and
                len(self.depends_on) == len(other.depends_on)):

            self_dep_on_values = [dep.value for dep in self.depends_on]
            self_dep_for_values = [dep.value for dep in self.dependency_for]

            other_dep_on_values = [dep.value for dep in other.depends_on]
            other_dep_for_values = [dep.value for dep in other.dependency_for]

            for val in self_dep_for_values:
                if val not in other_dep_for_values:
                    return False
            for val in self_dep_on_values:
                if val not in other_dep_on_values:
                    return False
            for val in other_dep_for_values:
                if val not in self_dep_for_values:
                    return False
            for val in other_dep_on_values:
                if val not in self_dep_on_values:
                    return False

            return True
        
        return False

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        string = str(self.value)
        if self.depends_on:
            string += " Depends on: "
            for dependency in self.depends_on:
                string += str(dependency.value) + ", "
        return string[:-2] if self.depends_on else string

NodeUnion = Union[List[DependencyNode], Tuple[DependencyNode], Set[DependencyNode]]

class CircularDependencyException(Exception):
    def __init__(self, circle: NodeUnion, *args):
        self.circle_nodes = circle
        super().__init__(self, *args)


class DisconnectedDependencyGraphException(Exception):
    def __init__(self, disconnected_nodes: NodeUnion, *args):
        self.disconnected_nodes = disconnected_nodes
        super().__init__(self, *args)

DependencyGraph = Dict[Any, DependencyNode]
SolveOrder = List[Any]

def calculate_order(dependency_graph: DependencyGraph,
                    calculate_circle_dep=False,
                    with_disconnect_error=False
                    ) -> SolveOrder:
    # First get all no dependecy nodes
    if with_disconnect_error:
        no_dependency_nodes = _get_no_dep_nodes(dependency_graph)
    else:
        no_dependency_nodes = [dependency_graph[node] for node in dependency_graph if not dependency_graph[node].depends_on]

    # Now order by going through the dependency nodes
    edge_order = _resolve_dependencies(no_dependency_nodes)
    if len(edge_order) != len(dependency_graph):
        cycle = calculate_cycle(dependency_graph) if calculate_circle_dep else []
        raise CircularDependencyException(cycle, "Circular dependency found")
    else:
        return edge_order

def calculate_cycle(dependency_graph: DependencyGraph) -> List[DependencyNode]:
    # Get all dep nodes that still have dependencies
    if not isinstance(dependency_graph, dict):
        dependency_graph = {node.value: node for node in dependency_graph}

    dep_nodes = {
        dependency_graph[node]
        for node in dependency_graph if dependency_graph[node].depends_on
    }
    bfs_graph = {
        dependency_graph[i]: DependencyNode(dependency_graph[i])
        for i in dependency_graph
        }
    # Go through the graph BFS style
    while dep_nodes:
        seen = set()
        outer_node = dep_nodes.pop()#next(iter(dep_nodes))
        queued = [bfs_graph[outer_node]]
        while queued:
            node = queued[0]
            queued.remove(node)
            #dep_nodes.remove(node.value)
            seen.add(node.value)
            for dep_node in node.value.depends_on:
                if dep_node not in seen:
                    if not bfs_graph[dep_node].depends_on:
                        bfs_graph[dep_node].add_dependency(node)
                    queued.append(bfs_graph[dep_node])
                else: # Found a cycle
                    path = [node]
                    while path[-1].value != dep_node and path[-1].depends_on:
                        if len(path[-1].depends_on) != 1:
                            assert len(path[-1].depends_on) == 1, "DEBUG: Should only depend on one node!"
                        path.append(path[-1].depends_on[0])
                    if len(path[-1].depends_on) == 1 or path[-1].value == dep_node:
                        value_cycle = [n.value for n in path]
                        return [val for val in reversed(value_cycle)]
        for k in bfs_graph:
            for node in bfs_graph[k].depends_on:
                bfs_graph[k].remove_dependency(node)
    return []

def _resolve_dependencies(no_dependency_nodes: DependencyGraph) -> List[Any]:
    edge_order = []
    while no_dependency_nodes:
        node = no_dependency_nodes.pop()
        edge_order.append(node.value)
        # We need to copy the dependency_for list before deleting entries
        for dep_node in [dep for dep in node.dependency_for]:
            dep_node.remove_dependency(node)
            # Add dependency node to no dependency list if free of dependencies
            if not dep_node.depends_on:
                no_dependency_nodes.append(dep_node)
    return edge_order

def _get_no_dep_nodes(dependency_graph: DependencyGraph) -> List[DependencyNode]:
    no_dependency_nodes = []
    queued = {next(iter(dependency_graph.values()))}
    seen = set()
    while queued:
        node = queued.pop()
        if not node.depends_on:
            no_dependency_nodes.append(node)
        seen.add(node)
        for dep_node in node.depends_on + node.dependency_for:
            if dep_node not in queued and dep_node not in seen:
                queued.add(dep_node)
    if not len(seen) == len(dependency_graph):
        raise DisconnectedDependencyGraphException(seen, "Dependency graph is not connected")
    return no_dependency_nodes
