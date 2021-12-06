from typing import List

#先選擇第一個位置做一個基準點後 用雙指針 一個從頭找 一個從尾巴找
#左邊找比基準點小的值 右邊找比基準點大的值 找到後互換
#當左邊與右邊相撞時 跟基準點互換位置 nlon


def quick_sort(nums: List[int], left: int, right: int):
    key = nums[left]
    l = left
    r = right

    while l < r:

        while nums[l] < key and l < r:
            l += 1

        while nums[r] > key and l < r:
            r -= 1

        if l < r:
            nums[l], nums[r] = nums[r], nums[l]

    if 

    return nums


quick_sort([10,2,3,6,4,5,1,7,9,8])