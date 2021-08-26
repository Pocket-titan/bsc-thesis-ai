import numpy as np


def generate_attribute(n_items, eps=0.1, children_per_parent=2, initial_nodes=1):
    """
    Generate a single attribute vector of length `n_items` through a process of branching diffusion.
    """
    tree = [[np.random.choice([-1, 1]) for _ in range(initial_nodes)]]

    while len(tree[-1]) < n_items:
        parents = tree[-1]
        new_generation = []

        for parent in parents:
            children = [
                np.random.choice([parent, -1 * parent], p=[1 - eps, eps])
                for _ in range(children_per_parent)
            ]
            new_generation.extend(children)

        tree.append(new_generation)

    return tree[-1][:n_items]


def generate_data(
    n_items,
    n_attributes,
    eps=0.1,
    levels=None,
    children_per_parent=2,
    initial_nodes=1,
):
    """
    Generate a dataset of shape (n_items, n_attributes).
    """
    if n_items > 2 ** n_attributes:
        raise Exception(
            f"Can't make {n_items} unique items with {n_attributes}, best I can do is {2**n_attributes}."
        )

    if levels is not None:
        children_per_parent = np.ceil((n_items / initial_nodes) ** (1 / (levels - 1))).astype(int)

    attributes = []
    while len(attributes) < n_attributes:
        attributes.append(
            attribute := generate_attribute(
                n_items,
                eps=eps,
                children_per_parent=children_per_parent,
                initial_nodes=initial_nodes,
            )
        )

        if len(attributes) == n_attributes:
            items = np.array(attributes).transpose()
            n_unique_items = len(np.unique([",".join([str(x) for x in row]) for row in items]))

            # There's definitely better ways to do this, but whatever
            if n_unique_items < n_items:
                indices = np.random.choice(range(len(attributes)), size=n_items - n_unique_items)
                attributes = list(np.delete(attributes, indices, axis=0))

    attributes = np.array(attributes)
    attributes[attributes == -1] = 0
    return attributes.transpose()
