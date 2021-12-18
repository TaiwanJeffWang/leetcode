from typing import List

#先選擇第一個位置做一個基準點後 用雙指針 一個從頭找 一個從尾巴找
#左邊找比基準點小的值 右邊找比基準點大的值 找到後互換
#當左邊與右邊相撞時 跟基準點互換位置 nlon

def quick_sort(data, left, right):
    if left > right:
        return

    i = left
    j = right
    key = data[left]

    while i != j:

        while data[j] > key and i < j:   # 從右邊開始找，找比基準點小的值
            j -= 1
        while data[i] <= key and i < j:  # 從左邊開始找，找比基準點大的值
            i += 1
        if i < j:                        # 當左右代理人沒有相遇時，互換值
            data[i], data[j] = data[j], data[i]

    data[left] = data[i]
    data[i] = key

    quick_sort(data, left, i-1)
    quick_sort(data, i+1, right)


data = [89, 34, 23, 78, 67, 100, 66, 29, 79, 55, 78, 88, 92, 96, 96, 23]
quick_sort(data, 0,len(data)-1)
print(data)


