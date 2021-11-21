# o(n^2)

from typing import List


def bubble_sort(nums: list) -> List:
    length = len(nums)
    for i in range(length):
        for j in range(i, length):
            if nums[i] > nums[j]:
                nums[j], nums[i] = nums[i], nums[j]
    return nums


x = bubble_sort([4, 3, 2, 1, 0])
print(x)

