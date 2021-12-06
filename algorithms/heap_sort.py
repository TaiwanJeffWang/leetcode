from typing import List


# the heap sort has somes step:
# 1. adjust nums to  max_heap or min_heap
# 2-1. max heap: nums[i] >= nums[2i] and nums[i] >= nums[2i=1] , but it doesn't matter the value between [2i] and [2i+1]
# 2-2. min heap: nums[i] <= nums[2i] and nums[i] <= nums[2i=1] , but it doesn't matter the value between [2i] and [2i+1]
# when max heap or min heap is done
# swap the firsy value and last value , exclude the new last value , and re-construct the heap again
# the  time capacity is o(nlogn)則要繼續

#goor ref :https://www.geeksforgeeks.org/heap-sort/
def heap_sort(nums: List[int]) -> List[int]:

    n = len(nums)//2
    for s in range(n-1, -1, -1):
        adjust_heap(nums, s, len(nums))

    # Max Heap的特徵是「第一個node具有最大值」，如果要將資料「由小到大」排序，步驟如下：
    # 把「第一個node」和「最後一個node」互換位置。
    # 假裝heap的「最後一個node」從此消失不見。
    # 對「第一個node」進行adjust_heap。
    for i in range(len(nums)-1, -1, -1):
        nums[i], nums[0] = nums[0], nums[i]

        adjust_heap(nums, 0, i)

    return nums


# The main function to sort an array of given size
def adjust_heap(nums: List[int], s: int, m: int):
    if s > m:
        return

    large = s
    l = 2 * s + 1 #(左子樹)
    r = 2 * s + 2 #(右子樹)

    if l < m and nums[l] > nums[large]:
        large = l

    if r < m and nums[r] > nums[large]:
        large = r

    if large != s:  #(如果已經有交換過了 則要繼續往下面做檢查 是否下面的根大於子樹)
        nums[large], nums[s] = nums[s], nums[large]

        adjust_heap(nums, large, m)




x =heap_sort([50,10,90,30,70,40,80,60,20])