def merge_sort(nums):
    if len(nums) == 1:
        return nums

    middle = len(nums) // 2
    l = merge_sort(nums[:middle])
    r = merge_sort(nums[middle:])

    return merge_2_list(l, r)


def merge_2_list(left, right):
    i = 0
    j = 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    while i < len(left):
        result.append(left[i])
        i += 1

    while j < len(right):
        result.append(right[j])
        j += 1

    return result


def heap_sort(nums):
    m = len(nums) // 2

    def max_heap(nums, index, length):
        if index > length:
            return

        large = index
        left = 2*index + 1
        right = 2*index + 2

        if left < length and nums[left] > nums[large]:
            large = left

        if right < length and nums[right] > nums[large]:
            large = right

        if large != index:
            nums[large], nums[index] = nums[index], nums[large]
            max_heap(nums, large, length)

    # max heap
    for i in range(m, -1, -1):
        max_heap(nums, i, len(nums))

    #sort
    for i in range(len(nums) - 1, -1, -1):
        #beacause the nums[0] is max and we want to order by aesc so change nums[0] and last index
        nums[0], nums[i] = nums[i], nums[0]
        max_heap(nums, 0, i)
