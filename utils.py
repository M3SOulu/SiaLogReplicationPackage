from tensorflow.keras.metrics import Precision, Recall, Metric


def iter_shuffler(iterator, buffer_size):
    import random
    buffer = []
    for item in iterator:
        if len(buffer) < buffer_size:
            buffer.append(item)
        else:
            yield buffer.pop(random.randrange(buffer_size))
            buffer.append(item)
    random.shuffle(buffer)
    yield from buffer


def abbreviate(string: str) -> str:
    return "".join([c for c in string if c.isupper()])


def dump_dict(dictionary, indents=0):
    import numpy as np
    string = ""
    for k, v in dictionary.items():
        if type(v) == dict:
            string += ("\t" * indents) + f"{k}:\n{dump_dict(v, indents + 1)}"
        elif type(v) in {float, np.float16, np.float32, np.float64}:
            if 0 < v < 1:
                v *= 100
            string += ("\t" * indents) + f"{k}: {v:.2f}\n"
        else:
            string += ("\t" * indents) + f"{k}: {v}\n"

    return string


def average_dict(dict_list=None, *args):
    import numpy as np
    if dict_list is None:
        dict_list = args
    avg_dict = {}
    for k, v in dict_list[0].items():
        if type(v) == dict:
            avg_dict[k] = average_dict([d[k] for d in dict_list])
        elif type(v) in {float, np.float16, np.float32, np.float64, int, np.int}:
            avg_dict[k] = sum(d[k] for d in dict_list) / len(dict_list)
        else:
            raise TypeError(f"Type {type(v)} is not supported in average_dict")

    return avg_dict


def rand_bool(p_true):
    import numpy as np
    return np.random.rand() < p_true
