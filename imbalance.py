from algorithm_blockwise_nonnegative_least_squares import pertask_utilization_NNLS
from algorithm_greedy_mean_only import pertask_utilization_greedy

def temporal_imbalance():
    contribs_nnls = pertask_utilization_NNLS()
    contribs_greedy = pertask_utilization_greedy()

    for wi, nodes in contribs_nnls:
        

    for wi, nodes in contribs_greedy:
        print(wi, nodes)

if __name__ == "__main__":
    