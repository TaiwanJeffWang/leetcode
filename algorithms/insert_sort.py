from typing import List

#玩鋪克牌 排序法 ,
def insert_sort(arr: List[int]):
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key

    return arr


insert_sort([3,1,4,1,7,6,25,5])