from typing import List

#玩鋪克牌 排序法 ,

def insertion_sort(nums: List) -> list:
    leng = len(nums)
    #先固定第一個位置
    for i in range(1, leng):
        #因為之後位置會被取代掉 所以先把值拿出
        key = nums[i]
        #接著從 i 往前比！
        j = i - 1
        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1

        nums[j + 1] = key

    return nums

insertion_sort([9, 1, 4, 3, 5, 7, 2])
