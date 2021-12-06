


from typing import List


def simple_selection_sort(nums: List[int]):
    leng = len(nums)

    for i in range(leng):
        temp = i
        for j in range(i+1, leng):
            if nums[j] < nums[temp]:
                temp = j

        if nums[i] != nums[temp]:
            nums[i], nums[temp] = nums[temp], nums[i]

    return nums



simple_selection_sort([3,2,1,5,7,4,5,7,8,3,23,99,87,421])