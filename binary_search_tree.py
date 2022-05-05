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

    def delete_binary_tree(self, root: TreeNode, k: int)->TreeNode:
        #如果沒有左子樹也沒有右子樹的話 直接移除
        #如果只有左子樹 或是 只有右子樹的話 移除後直接接上去
        #如果有左子樹跟右子樹， 找離該移除點最近的值填上(左子樹中最大直 或是右子樹中最小值)

        if root.val == k:
            if root.left is None and root.right is None:
                root = None
            elif root.left and root.right is None:
                root = root.left
            elif root.left is None and root.right:
                root = root.right
            else:
                #有左子樹也有右子樹：
                max_node = self.findＭax(root.left)
                root.val = max_node.val
                root.left = self.delete_binary_tree(root.left, max_node.val)

        else:
            if root.val < k:
                root.right = self.delete_binary_tree(root.right, k)
            else:
                root.left = self.delete_binary_tree(root.left, k)

        return root

    def findＭax(self, root: TreeNode) -> TreeNode:
        while root.right:
            root = root.right

        return root



a = Solution()
x = a.create_binary_tree([62, 88, 58, 47, 35, 73, 51, 99, 37, 93])
c = a.isValidBST(x)
b = a.delete_binary_tree(x, 93)
c = a.isValidBST(b)
print(x)
