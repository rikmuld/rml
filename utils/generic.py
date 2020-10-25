
def do_many(f, *args):
    for arg in args:
        f(*arg)


def build_idx_map(la):
    dct = {i: [] for i in set(la)}

    for idx, i in enumerate(la):
        dct[i].append(idx)
    
    return dct