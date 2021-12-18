from os import close, sendfile
from typing import Dict, List, Optional
import math
from collections import deque
import heapq


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if root is None:
            return

        if root.val < val:
            if root.right is None:
                root.right = TreeNode(val)
            else:
                self.insertIntoBST(root.right, val)
        else:
            if root.left is None:
                root.left = TreeNode(val)
            else:
                self.insertIntoBST(root.left, val)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = []
        trav = root
        prev = -float('inf')
        while trav or stack:
            if trav:
                stack.append(trav)
                trav = trav.left
            else:
                u = stack.pop()
                if u:
                    if u.val <= prev:
                        return False
                    prev = u.val
                trav = u.right
        return True

    def create_binary_tree(self, nums: List[int]):
        result = TreeNode(nums[0])

        for i in range(1, len(nums)):
            self.insertIntoBST(result, nums[i])

        return result

    # def create_bst(self, nums: List[int], k: int, root: TreeNode):
    #     root.val = nums[k]

    #     if k+1 < len(nums):
    #         if nums[k+1] > root.val:
    #             root.right = TreeNode()
    #             self.create_bst(nums, k+1, root.right)
    #         else:
    #             root.left = TreeNode()
    #             self.create_bst(nums, k+1, root.left)


a = Solution()
x = a.create_binary_tree([62, 88, 58, 47, 35, 73, 51, 99, 37, 93])
print(x)
