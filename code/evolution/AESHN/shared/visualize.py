from typing import List, Tuple, Dict

import graphviz
import plotly.graph_objs as pltlygo
import plotly.io as pltlyio
from neat.graphs import feed_forward_layers

from AESHN.shared.modular_network_it import Connection
from AESHN.shared.substrate_factory import DGCBSubstrate
import itertools

try:
    import cPickle as pickle
except:
    import pickle


def get_cppn_input_node_names() -> List[str]:
    # return ["x1", "y1", "z1", "x2", "y2", "z2", "cl", "bias"]
    return ["x1", "y1", "z1", "x2", "y2", "z2", "b_x", "b_y", "b_z", "cl", "bl", "bias"]


def get_cppn_output_node_names(params):
    if params["experiment"]["dgcb"]:
        if params["morph_params"]["fix_brain"]:
            return ["w", "bias", "tau", "lr", "A", "B", "C", "D", "M", "NB", "FB", "SB", "SBD", "FJ", "CJ", "LAX",
                    "HAX", "AYL", "AZL"]
        else:
            return ["w", "bias", "tau", "lr", "A", "B", "C", "D", "M", "NB", "BB", "FB", "SB", "SBD", "FJ", "CJ", "LAX",
                    "HAX", "AYL", "AZL"]
    else:
        if params["morph_params"]["fix_brain"]:
            return ["w", "lr", "A", "B", "C", "D", "M", "FB", "CJ", "LAX",
                    "HAX", "AYL", "AZL"]
        else:
            return ["w", "lr", "A", "B", "C", "D", "M", "NB", "BB", "FB", "SB", "SBD", "FJ", "CJ", "LAX", "HAX", "AYL",
                    "AZL"]


def draw_net(net, filename=None):
    """
    Uses graphviz to draw a CPPN with arbitrary topology.
    """
    input_node_names = get_cppn_input_node_names()
    output_node_names = get_cppn_output_node_names(net.params)
    node_names = {}
    node_colors = {}

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    for k, label in zip(net.input_nodes, input_node_names):
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, label=label, _attributes=input_attrs)

    for k, label in zip(net.output_nodes, output_node_names):
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled',
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, label=label, _attributes=node_attrs)

    for k, label in net.hidden_nodes_and_act_f:
        name = node_names.get(k, str(k))
        node_atrrs = {
            'style': 'filled',
            'fillcolor': node_colors.get(k, 'white')
        }
        dot.node(name, label=label, _attributes=node_atrrs)

    connections = list()
    for node, act_func, bias, response, inodes, iweights in net.node_evals:
        for i, w in zip(inodes, iweights):
            connections.append((i, node))
            input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(input, str(input))
            style = 'solid'
            color = 'green' if w > 0.0 else 'red'
            width = str(0.1 + abs(w / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)

    # Morph paths
    inodes = {i for i, name in zip(net.input_nodes, input_node_names) if name in ["b_x", "b_y", "b_z", "bl", "bias"]}
    onodes = {i for i, name in zip(net.output_nodes, output_node_names) if
              name in ["FB", "CJ", "LAX", "HAX", "AYL", "AZL"]}
    draw_specific_cppn_paths(net, inodes, onodes, connections, filename + "_morph")

    # Brain paths
    inodes = {i for i, name in zip(net.input_nodes, input_node_names) if
              name in ["x1", "y1", "z1", "x2", "y2", "z2", "cl", "bias"]}
    onodes = {i for i, name in zip(net.output_nodes, output_node_names) if
              name in ["w", "lr", "A", "B", "C", "D", "M"]}
    draw_specific_cppn_paths(net, inodes, onodes, connections, filename + "_brain")

    return dot


def draw_specific_cppn_paths(cppn, input_nodes, output_nodes, connections, filename=None):
    # Get hidden nodes connecting the input_nodes with output_nodes
    c = set(output_nodes)
    hidden_nodes = set()
    while True:
        c = set(a for (a, b) in connections if b in c)
        c -= input_nodes
        if c:
            hidden_nodes = hidden_nodes.union(c)
        else:
            break
    hidden_nodes -= set(cppn.input_nodes)

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}
    node_names = {}
    node_colors = {}
    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    for k, label in zip(cppn.input_nodes, get_cppn_input_node_names()):
        if k in input_nodes:
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled',
                           'shape': 'box',
                           'fillcolor': node_colors.get(k, 'lightgray')}
            dot.node(name, label=label, _attributes=input_attrs)

    for k, label in zip(cppn.output_nodes, get_cppn_output_node_names(cppn.params)):
        if k in output_nodes:
            name = node_names.get(k, str(k))
            node_attrs = {'style': 'filled',
                          'fillcolor': node_colors.get(k, 'lightblue')}
            dot.node(name, label=label, _attributes=node_attrs)

    for k, label in cppn.hidden_nodes_and_act_f:
        if k in hidden_nodes:
            name = node_names.get(k, str(k))
            node_atrrs = {
                'style': 'filled',
                'fillcolor': node_colors.get(k, 'white')
            }
            dot.node(name, label=label, _attributes=node_atrrs)

    for node, act_func, bias, response, inodes, iweights in cppn.node_evals:
        if node in input_nodes or node in output_nodes or node in hidden_nodes:
            for i, w in zip(inodes, iweights):
                if i in input_nodes or i in output_nodes or i in hidden_nodes:
                    input, output = node, i
                    a = node_names.get(output, str(output))
                    b = node_names.get(input, str(input))
                    style = 'solid'
                    color = 'green' if w > 0.0 else 'red'
                    width = str(0.1 + abs(w / 5.0))
                    dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)


# Node colors
INPUT_NODE_COLOR = 'green'  # "#01d000"  # green
OUTPUT_NODE_COLOR = 'red'  # "d00000"  # red
RECURRENT_NODE_COLOR = 'blue'  # "#0001d0"  # blue
MOD_NODE_COLOR = "#eeaaff"  # light pink
NORMAL_NODE_COLOR = "#959595"  # gray

# Edge colors
POSITIVE_EDGE_COLOR = INPUT_NODE_COLOR
NEGATIVE_EDGE_COLOR = OUTPUT_NODE_COLOR


def draw_es(coords_to_id: Dict[Tuple[float, float, float], int], connections: List[Connection],
            substrate: DGCBSubstrate, filename: str, morph):
    """
    Uses plotly to draw the neural networks created by AESHN.
    This also visualises the agent's morphology.
    """
    n_colors = []
    e_colors, e_m_colors = [], []

    recurrent_nodes = set()
    mod_nodes = set(coords_to_id.keys())

    xn, yn, zn = [], [], []
    xe, ye, ze = [], [], []
    xe_m, ye_m, ze_m = [], [], []

    traces = list()

    # Get edges and set mod / recurrent nodes
    for c in connections:
        x = [c.x1, c.x2, None]
        y = [c.y1, c.y2, None]
        z = [c.z1, c.z2, None]

        if c.weight > 0:
            color = POSITIVE_EDGE_COLOR
        else:
            color = NEGATIVE_EDGE_COLOR

        if c.modulatory:
            xe_m.extend(x)
            ye_m.extend(y)
            ze_m.extend(z)
            e_m_colors.append(color)
        else:
            if c.x1 == c.x2 and c.y1 == c.y2 and c.z1 == c.z2:
                recurrent_nodes.add((c.x1, c.y1, c.z1))
            else:
                mod_nodes.discard((c.x1, c.y1, c.z1))

                xe.extend(x)
                ye.extend(y)
                ze.extend(z)

                e_colors.append(color)

    for coord, idx in coords_to_id.items():
        if coord in substrate.input_coordinates:
            color = INPUT_NODE_COLOR
            mod_nodes.discard(coord)
        elif coord in substrate.output_coordinates:
            color = OUTPUT_NODE_COLOR
            mod_nodes.discard(coord)
        elif coord in mod_nodes:
            color = MOD_NODE_COLOR
        elif coord in recurrent_nodes:
            color = RECURRENT_NODE_COLOR

        else:
            color = NORMAL_NODE_COLOR

        n_colors.append(color)
        x, y, z = coord

        xn.append(x)
        yn.append(y)
        zn.append(z)

    # Standard connections
    traces.append(pltlygo.Scatter3d(
        x=xe,
        y=ye,
        z=ze,
        line=dict(color=e_colors, width=1),
        hovertext='text',
        hoverinfo='none'
    ))

    # Modular connections
    traces.append(pltlygo.Scatter3d(
        x=xe_m,
        y=ye_m,
        z=ze_m,
        mode='lines',
        line=dict(color=e_m_colors, dash="dot", width=1),
        hoverinfo='none'  # todo: show weight
    ))

    # Nodes
    traces.append(pltlygo.Scatter3d(
        x=xn,
        y=yn,
        z=zn,
        mode='markers',
        marker=dict(
            symbol='circle',
            size=6,
            color=n_colors,
            line=dict(color='rgb(50,50,50)', width=0.5)
        )
    ))

    if morph is not None:
        # Draw blocks
        #   Get vertices
        all_vertices = set()
        xme, yme, zme = list(), list(), list()

        for bx, by, bz in morph.block_locations:
            vertices = list()
            for dx in [-morph.sub_ws / 2, morph.sub_ws / 2]:
                x = bx + dx
                for dy in [-morph.sub_hs / 2, morph.sub_hs / 2]:
                    y = by + dy
                    for dz in [-morph.sub_ds / 2, morph.sub_ds / 2]:
                        z = bz + dz
                        vertices.append((x, y, z))

            # ugly but okay
            possible_edges = itertools.combinations(vertices, 2)
            for v1, v2 in possible_edges:
                distances = iter([dis - 0.001 < abs(x - y) < dis + 0.001 for x, y, dis in zip(v1, v2, morph.sub_s)])
                if any(distances) and not any(distances):
                    xme.extend([v1[0], v2[0], None])
                    yme.extend([v1[1], v2[1], None])
                    zme.extend([v1[2], v2[2], None])

            all_vertices = all_vertices.union(vertices)
        xmv, ymv, zmv = zip(*all_vertices)

        traces.append(
            pltlygo.Scatter3d(
                x=xmv,
                y=ymv,
                z=zmv,
                mode='markers',
                marker=dict(
                    symbol='square',
                    size=6,
                    color='black'
                )
            )
        )

        traces.append(pltlygo.Scatter3d(
            x=xme,
            y=yme,
            z=zme,
            mode='lines',
            line=dict(color='black'),
            hoverinfo='none'
        ))

        #   todo: Draw edges between vertices

    # General plot settings
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    mod_nodes_count = len(mod_nodes)
    rec_nodes_count = len(recurrent_nodes)
    normal_nodes_count = len(coords_to_id) - mod_nodes_count - rec_nodes_count
    connections_count = len(connections)

    layout = pltlygo.Layout(
        title=f"AESHN Brain | "
              f"{normal_nodes_count} normal nodes | "
              f"{mod_nodes_count} mod nodes | "
              f"{rec_nodes_count} rec nodes | "
              f"{connections_count} connections",
        width=1200,
        height=1200,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
    )

    fig = pltlygo.Figure(data=traces, layout=layout)
    camera = {
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {"x": 0, "y": 0, "z": -2}
    }
    fig.update_layout(scene_camera=camera)

    pltlyio.write_html(fig, file=filename + '.html')


def draw_trajectory(position_samples_2d: List[Tuple[float, float]],
                    target_locations_2d: List[Tuple[float, float]], filename: str):
    """
    Uses plotly to draw the given 2D trajectory and target location sequence.
    """
    traces = [
        [pltlygo.Scatter(x=[xx], y=[yy], mode='markers',
                         marker=dict(symbol='circle', color=RECURRENT_NODE_COLOR, size=10)),
         pltlygo.Scatter(x=[tx], y=[ty], mode='markers',
                         marker=dict(symbol='circle', color=OUTPUT_NODE_COLOR, size=10))
         ]
        for (xx, yy), (tx, ty) in zip(position_samples_2d, target_locations_2d)]

    frames = [pltlygo.Frame(data=trace, name=str(i)) for i, trace in
              enumerate(traces)]

    slider_steps = [
        {"args": [
            [i],
            {"frame": {"duration": 1, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 1}}
        ],
            "method": "animate"}
        for i in range(len(frames))]

    layout = pltlygo.Layout(
        xaxis=dict(range=[-100, 100], autorange=False, zeroline=False),
        yaxis=dict(range=[-100, 100], autorange=False, zeroline=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                {
                    "args": [None, {"frame": {"duration": 1, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 1,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 1, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": slider_steps
        }]
    )

    fig = pltlygo.Figure(
        data=traces[0],
        layout=layout,
        frames=frames
    )

    pltlyio.write_html(fig, file=filename + '.html')


# Draw quad tree
def draw_qt(root):
    pass
    """ DEPRECATED
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    def get_leaves(node):
        children = [c for c in node.cs if c is not None]
        if not children:
            return [node]
        else:
            leaves = list()
            for c in children:
                leaves.extend(get_leaves(c))

            return leaves

    leaves = get_leaves(root)

    # Add point per leave
    # Add rectangle per leave

    fig, ax = plt.subplots(1)
    fig.set_size_inches(18.5, 10.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(color='gray', linestyle='-', linewidth=1)
    ax.scatter([p.x for p in leaves], [p.y for p in leaves], c='r', s=2)
    rects = [patches.Rectangle((p.x - p.width, p.y - p.width), 2 * p.width, 2 * p.width, 0, linewidth=3, edgecolor='r',
                               facecolor='none') for p in leaves]
    for rect in rects:
        ax.add_patch(rect)
    plt.show()
    """
