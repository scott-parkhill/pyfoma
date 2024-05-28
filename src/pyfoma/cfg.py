#!/usr/bin/env python

"""Context-Free Grammar tools"""

import graphviz

def draw_cfg(cfg_string: str, style='tree'):
    """
    Draws a CFG string as a tree with graphviz.
    :param cfg_string: The CFG string to render
    :param style: The style to render the CFG in. Choose from 'tree', 'boxes'.
    :return: A `graphviz.Graph` object.
    """

    START_END_SYMBOLS = {"(": ")", "[": "]"}

    graph = graphviz.Graph(comment=cfg_string)
    all_nodes = []

    def parse_string(graph, initial_index=0, start_marker=None) -> tuple[int, str]:
        """
        Parses the string starting at the specified index, until a closing parenthesis or the end of the string is found.
        :param initial_index: The index in the string to start at
        :param start_marker: If provided, defines the type of marker that the group uses to mark the start. Must be either ( or [
        :return: (new_index, subtree_root) where new_index is the next un-parsed index, and `subtree_root` is the root of the subtree
        """
        index = initial_index
        current_parent = None  # The parent of the current subtree
        current_children = []  # The children ids of the current parent
        current_node = None  # A new node that is being read in

        use_clusters = style == 'boxes'
        hide_box_attrs = {'peripheries': '0', 'margin': '2'}

        with graph.subgraph(name=("cluster_" if use_clusters else "") + str(initial_index),
                               graph_attr=hide_box_attrs if style == 'tree' else {}) as subgraph:

            while index < len(cfg_string):
                parsed_char = cfg_string[index]

                if parsed_char.isspace() or parsed_char in START_END_SYMBOLS.values():
                    # If we are currently reading a node, we've finished. Add the node to the graph.
                    if current_node:
                        node_id = str(len(all_nodes))
                        all_nodes.append(current_node)

                        if not current_parent:
                            # If there's no parent yet, this must be the parent
                            subgraph.node(node_id, current_node, shape='plain', group=node_id)
                            current_parent = node_id
                        else:
                            # Must be a daughter
                            subgraph.node(node_id, current_node, shape='plain')
                            current_children.append(node_id)
                        current_node = None

                    if parsed_char in START_END_SYMBOLS.values():
                        # We found the end. This *must* be the end of the current subtree,
                        # because any other end symbols should have already been parsed.
                        if START_END_SYMBOLS[start_marker] != parsed_char:
                            # Parenthesis mismatch
                            raise SyntaxError(
                                f"Parenthesis mismatch. A matching parenthesis {START_END_SYMBOLS[start_marker]} must be provided for the parenthesis at position {initial_index}",
                                ("", 1, index, cfg_string))

                        # Add edges for all the daughters
                        for child in current_children:
                            subgraph.edge(current_parent, child)

                            # If there's only one child, set the group on that node so we get a vertical edge
                            if len(current_children) == 1:
                                subgraph.node(child, all_nodes[int(child)], group=current_parent)

                        return index, current_parent
                elif parsed_char in START_END_SYMBOLS.keys():
                    # We start a new group which continues until a close parenthesis
                    (index, subroot_id) = parse_string(subgraph, index + 1, start_marker=parsed_char)
                    if current_parent:
                        current_children.append(subroot_id)
                else:
                    # This must be either the root of the current subtree, or a leaf daughter
                    # Read the string until we see a space
                    if current_node:
                        current_node += parsed_char
                    else:
                        current_node = parsed_char
                index += 1

            if start_marker:
                # If we got this far and never saw an end marker, we're missing parenthesis
                raise SyntaxError(
                    f"Parenthesis mismatch. A matching parenthesis {START_END_SYMBOLS[start_marker]} must be provided for the parenthesis at position {initial_index}",
                    ("", 1, index, cfg_string))

    parse_string(graph)
    graph.attr(ranksep='0.3', splines='false')
    if style == 'boxes':
        graph.edge_attr['style'] = 'invis'
    return graph
