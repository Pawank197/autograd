from graphviz import Digraph
from tensor import Tensor

def trace(root: Tensor):
    """Walk the graph upstream from `root`, collecting nodes and edges."""
    nodes, edges = set(), set()
    def build(v: Tensor):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root: Tensor, format='svg', rankdir='LR') -> Digraph:
    """
    Render the autograd graph of `root` with graphviz.
    - format: 'png' | 'svg' | ...
    - rankdir: 'LR' (left→right) or 'TB' (top→bottom)
    """
    assert rankdir in ('LR','TB')
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    # for each Tensor node
    for n in nodes:
        # show op name, data and grad
        label = "{{%s | data %.4f | grad %.4f}}" % (n._op or n.label or 'input', float(n.data.flatten()[0]), float(n.grad.flatten()[0]))
        dot.node(str(id(n)), label=label, shape='record')
        # if this node was produced by an op, draw an op‐node in between
        if n._op:
            op_id = str(id(n)) + n._op
            dot.node(op_id, label=n._op)
            dot.edge(op_id, str(id(n)))

    # connect children→parent edges via the op‐nodes
    for child, parent in edges:
        if parent._op:
            dot.edge(str(id(child)), str(id(parent)) + parent._op)
        else:
            dot.edge(str(id(child)), str(id(parent)))

    return dot