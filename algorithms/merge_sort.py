

from typing import List


def merge_sort(nums: List[int]) -> List[int]:
    if len(nums) == 1:
        return nums
    length = len(nums)
    l = merge_sort(nums[0:length//2])
    r = merge_sort(nums[length//2:])

    return merge(l, r)


def merge(l1: List[int], r1: List[int]) -> List[int]:
    result = []
    i, j = 0, 0
    while True:
        if i == len(l1):
            result.extend(r1[j:])
            break

        if j == len(r1):
            result.extend(l1[i:])
            break

        if l1[i] <= r1[j]:
            result.append(l1[i])
            i += 1
        else:
            result.append(r1[j])
            j += 1

    return result


merge_sort([4,5,2,1,3,6,7])
