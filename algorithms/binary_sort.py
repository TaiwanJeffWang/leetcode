
from typing import List
#從有序列中找到 元素是否存在
#一開始就對半切 再決定要從左半邊切 還是右半邊切

def binary_sort_2(nums: List[int], k: int):
    high = len(nums) - 1
    low = 0

    while low <= high:

        mid = (low+high) // 2

        if nums[mid] == k:
            return mid

        if nums[mid] > k:
            high = mid - 1
        else:
            low = mid + 1




y = binary_sort_2([2,3,4,5,6,7,8,9], 5)
print(y)
















