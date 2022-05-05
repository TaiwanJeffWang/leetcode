from typing import List

#先選擇第一個位置做一個基準點後 用雙指針 一個從頭找 一個從尾巴找
#左邊找比基準點小的值 右邊找比基準點大的值 找到後互換
#當左邊與右邊相撞時 跟基準點互換位置 best and avg case nlon , worst case n^2 , space: logn

def quick_sort(data, left, right):
    if left > right:
        return

    i = left
    j = right
    key = data[left]

    while i != j:

        while data[j] >= key and i < j:   # 從右邊開始找，找比基準點小的值
            j -= 1
        while data[i] <= key and i < j:  # 從左邊開始找，找比基準點大的值
            i += 1
        if i < j:                        # 當左右代理人沒有相遇時，互換值
            data[i], data[j] = data[j], data[i]

    data[left] = data[i]
    data[i] = key

    quick_sort(data, left, i-1)
    quick_sort(data, i+1, right)


# 沒有在管pivot 在哪一個位置 只要左邊大於pivot 右邊小於pivot就交換
# 優化版
def sortArray(nums: List[int]) -> List[int]:
    if not nums:
        return nums
    quickSort(0, len(nums) - 1, nums)
    return nums

def quickSort(start, end, nums):
    if start >= end:
        return

    left = start
    right = end
    pivot = nums[(left + right) // 2]

    while left <= right:
        while left < right and nums[left] < pivot:
            left += 1
        while left < right and nums[right] > pivot:
            right -= 1

        if left <= right:
            if left < right:
                nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    quickSort(start, right, nums)
    quickSort(left, end, nums)


data = [3,4,9,1,7,0,5,2,6,8]
sortArray(data)
print(data)

