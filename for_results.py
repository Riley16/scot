import numpy as np

# average lists
def avg_lists(lists):
    global_sum = 0
    lens = 0
    for l in lists:
        global_sum += sum(l)
        lens += len(l)
    print(global_sum / lens)

def var_lists(lists):
    variances = []
    for l in lists:
        variances.append(np.var(l))
    print(sum(variances) / len(lists))

if __name__ == "__main__":
    lists = [
        [10, 8, 6, 6, 5, 7, 7, 5, 9],
        [10, 8, 6, 6, 5, 7, 7, 5, 9]
        ]
    avg_lists(lists)
    var_lists(lists)
