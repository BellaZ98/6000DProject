import numpy as np
import time
import random
import multiprocessing
import gc


def split_list(input_list, division_count):
    length_input_list = len(input_list)
    if division_count <= 0 or length_input_list == 0:
        return []
    if division_count > length_input_list:
        return []
    elif division_count == length_input_list:
        return [[element] for element in input_list]
    else:
        segment_size = length_input_list // division_count
        remainder = length_input_list % division_count
        result_list = []
        for index in range(0, (division_count - 1) * segment_size, segment_size):
            result_list.append(input_list[index:index + segment_size])
        result_list.append(input_list[(division_count - 1) * segment_size:])
        return result_list


def calculate_multi_ranking(tasks, dist_matrix, dist_matrix_transposed, k_values, additional_args):
    left_to_right_accuracy, right_to_left_accuracy = np.array([0.] * len(k_values)), np.array([0.] * len(k_values))
    l2r_mean, r2l_mean, l2r_mrr, r2l_mrr = 0., 0., 0., 0.
    for i, task in enumerate(tasks):
        reference = task
        sorted_indices = dist_matrix[i, :].argsort()
        rank_position = np.where(sorted_indices == reference)[0][0]
        l2r_mean += (rank_position + 1)
        l2r_mrr += 1.0 / (rank_position + 1)
        for j, k in enumerate(k_values):
            if rank_position < k:
                left_to_right_accuracy[j] += 1

    # Additional calculations can be added here

    return left_to_right_accuracy, right_to_left_accuracy, l2r_mean, r2l_mean, l2r_mrr, r2l_mrr
