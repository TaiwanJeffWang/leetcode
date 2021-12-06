
from typing import List

index = 0


def binary_sort(nums: List[int], target: int):
    m = len(nums) // 2

    if nums[m] == target:
        return m

    if nums[m] >= target:
        return binary_sort(nums[:m], target)
    else:
        return binary_sort(nums[m+1:], target)


x = binary_sort([2,3,4,5,6,7,8,], 5)
print(x)