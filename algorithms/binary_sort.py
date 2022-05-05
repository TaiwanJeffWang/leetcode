
from typing import List
#從有序列中找到 元素是否存在
#一開始就對半切 再決定要從左半邊切 還是右半邊切

def binary_search(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
    return -1


y = binary_search([2,3,4,5,6,7,8,9], 5)
print(y)
















