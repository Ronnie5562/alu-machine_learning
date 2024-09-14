#!/usr/bin/env python3
"""_summary_
Contains a func add_arrays(arr1, arr2) that adds elements in same position.
"""


def add_arrays(arr1, arr2):
    """_summary_

    Args:
        arr1 (int): _description_
        arr2 (int): _description_

    Returns:
        list: A new list
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
