from typing import List


# the heap sort has somes step:
# 1. adjust nums to  max_heap or min_heap
# 2-1. max heap: nums[i] >= nums[2i] and nums[i] >= nums[2i=1] , but it doesn't matter the value between [2i] and [2i+1]
# 2-2. min heap: nums[i] <= nums[2i] and nums[i] <= nums[2i=1] , but it doesn't matter the value between [2i] and [2i+1] => 在這個狀態下heap[k] <= heap[2*k+1] 且 heap[k] <= heap[2*k+2]
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
def adjust_heap(nums: List[int], index: int, length: int):
    if index > length:
        return

    large = index
    l = 2 * index + 1 #(左子樹)
    r = 2 * index + 2 #(右子樹)

    if l < length and nums[l] > nums[large]:
        large = l

    if r < length and nums[r] > nums[large]:
        large = r

    if large != index:  #(如果已經有交換過了 則要繼續往下面做檢查 是否下面的根大於子樹)
        nums[large], nums[index] = nums[index], nums[large]

        adjust_heap(nums, large, length)


def heap_sort_2(nums):
    # step 1: create a max heap or min heap, we should start with len(nums)/2
    # step 2: sort   0 1 2
    n = (len(nums) // 2) - 1
    for i in range(n, -1, -1):
        adjust_max_heap(nums, i, len(nums))

    # Max Heap的特徵是「第一個node具有最大值」，如果要將資料「由小到大」排序，步驟如下：
    # 把「第一個node」和「最後一個node」互換位置。
    # 假裝heap的「最後一個node」從此消失不見。
    # 對「第一個node」進行adjust_heap。
    for i in range(len(nums)-1, -1, -1):
        nums[i], nums[0] = nums[0], nums[i]

        adjust_max_heap(nums, 0, i)

    return nums

def adjust_max_heap(nums, index, length):  #一定要把length傳進去, 因為在第二步驟調整剩餘的陣列時需要判斷
    large = index
    left = 2*index + 1
    right = 2*index + 2

    if index > length:
        return

    if left < length and nums[left] > nums[large]:
        large = left

    if right < length and nums[right] > nums[large]:
        large = right

    if large != index:
        nums[large], nums[index] = nums[index], nums[large]
        adjust_max_heap(nums, large, length)

a = [50, 10, 90, 30, 70, 40, 80, 60, 20]
heap_sort_2(a)
print(a)