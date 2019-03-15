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
        [34, 16, 14, 19, 76, 21, 18, 18, 79, 29, 25, 28, 24, 25, 73, 79, 21, 16, 75, 15]
        ]
    avg_lists(lists)
    var_lists(lists)
