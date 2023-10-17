def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]
