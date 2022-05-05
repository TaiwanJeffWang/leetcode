
import collections
from os import close, remove, sendfile
from typing import Collection, Dict, List, Optional, OrderedDict
import math
from collections import Counter, defaultdict, deque
import heapq
import sys
import bisect
from functools import cmp_to_key
import itertools
import copy
import uuid

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:

    #763. Partition Labels
    def partitionLabels_1(self, s: str) -> List[int]:
        # Input: s = "ababcbacadefegdehijhklij"
        # Output: [9,7,8]
        # 當中間的所有字母在後面都沒有出現的時候 就可以形成一個斷點
        result = []
        max_depth = 0
        min = 0
        length = len(s)
        for i in range(length):
            print(s[i])
            max_depth = max(s.rfind(s[i]), max_depth)  #使用這種方法 代表每次都要找尋一次字串 所以會是 time: n^2
            if i == max_depth:
                result.append(max_depth+1-min)
                min = max_depth+1

        return result


    #763. Partition Labels hash table
    #使用table ,一開始就把最後一個位置找到 接著根據區間操作 time:n
    def partitionLabels_2(self, s: str) -> List[int]:
        last_index = {}
        for i, ch in enumerate(s):
            last_index[ch] = i

        start = end = 0
        ans = []
        for i, ch in enumerate(s):
            if last_index[ch] > end:
                end = last_index[ch]
            if i == end:
                ans.append(end-start+1)
                start = end + 1
        return ans

    #17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        d={'2':['a','b','c'],
           '3':['d','e','f'],
           '4':['g','h','i'],
           '5':['j','k','l'],
           '6':['m','n','o'],
           '7':['p','q','r','s'],
           '8':['t','u','v'],
           '9':['w','x','y','z']}

        if digits == "":
            return []

        result = []
        self.get_letter_combination(d, digits, "", result,  start_index=0)
        return result

    def get_letter_combination(self, d: dict, digits: str, temp: str, result: List, start_index: int):

        if len(temp) == len(digits):
            result.append(temp)
            return

        for w in digits[start_index]:
            for word in d[w]:
                if word in temp:
                    continue

                temp += word
                self.get_letter_combination(d, digits=digits, temp=temp, result=result, start_index=start_index+1)
                temp = temp[:-1]

    #77. Combinations
    def combine(self, n: int, k: int) -> List[List[int]]:

        #back tracking , range (1,n)
        result = []
        self.get_combine(n, k, [], result, 1)
        return result

    def get_combine(self, n: int, k: int, temp: List, result: List, start_index: int):

        if len(temp) == k:
            result.append(list(temp))
            return

        for i in range(start_index, n+1):
            if i in temp:
                continue

            temp.append(i)
            self.get_combine(n, k, temp, result=result, start_index=i+1)
            temp.pop()

    #347. Top K Frequent Elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        tb = {}
        for i in nums:
            if i in tb:
                tb[i] = tb[i]+1
            else:
                tb[i] = 1

        w = sorted(tb.items(), key=lambda x: x[1], reverse=True)

        result = []

        for i in range(k):
           result.append(w[i][0])

        return result

    

    #136. Single Number
    def singleNumber(self, nums: List[int]) -> int:
        tb= {}
        for item in nums:
            if item in tb:
                tb.pop(item)
            else:
                tb[item] = 1

        w = list(tb)
        return w[0]

    #739. Daily Temperatures using stack
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        result = [0] * len(temperatures)


        for i in range(len(temperatures)):
            while len(stack) > 0:
                if temperatures[i] > temperatures[stack[-1]]:
                    result[stack[-1]] = i- stack[-1]
                    stack.pop()
                else:
                    break

            stack.append(i)

        return result



    #617. Merge Two Binary Trees
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        # return self.get_merge_trees(root1, root2)
        if root1 is None:
            return root2
        if root2 is None:
            return root1

        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)

        return root1

    # def get_merge_trees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]):
    #     if root1 is None and root2 is None:
    #         return

    #     if root1 is not None and root2 is not None:
    #         result = TreeNode(val=root1.val + root2.val)
    #         result.left = self.get_merge_trees(root1.left, root2.left)
    #         result.right = self.get_merge_trees(root1.right, root2.right)

    #     elif root1 is not None:
    #         result = TreeNode(val=root1.val)
    #         result.left = self.get_merge_trees(root1.left, None)
    #         result.right = self.get_merge_trees(root1.right, None)

    #     elif root2 is not None:
    #         result = TreeNode(val=root2.val)
    #         result.left = self.get_merge_trees(None, root2.left)
    #         result.right = self.get_merge_trees(None, root2.right)

    #     return result



    #104  Maximum Depth of Binary Tree

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        q = self.get_maxDepth(root.left, 0)
        r = self.get_maxDepth(root.right, 0)
        return max(q, r)+1

    def get_maxDepth(self, root: TreeNode, result: int):
        if root is None:
            return result
        else:
            result += 1

        return max(self.get_maxDepth(root.left, result), self.get_maxDepth(root.right, result))

    #46. Permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        self.get_permute(nums, result, [])
        return result

    def get_permute(self, nums: list[int], result: List, temp: List):
        if len(temp) == len(nums):
            result.append(list(temp))
            return

        for i in nums:
            if i in temp:
                continue
            else:
                temp.append(i)
                self.get_permute(nums, result, temp)
                temp.pop()

        return

    #226. Invert Binary Tree
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return

        temp = root.left
        root.left = root.right
        root.right = temp

        self.invertTree(root.left)
        self.invertTree(root.right)

    #94. Binary Tree Inorder Traversal 145 Binary Tree postorder Traversal
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # middle left middle right:Inorder Traversal
        #left right middle :post order traversal
        result = []
        self.get_inorderTraversal(root, result)
        #self.get_postorderTraversal(root, result)
        return result

    def get_inorderTraversal(self, root: Optional[TreeNode], result: List):
        if root is None:
            return

        self.get_inorderTraversal(root.left, result)
        result.append(root.val)
        self.get_inorderTraversal(root.right, result)


    def get_postorderTraversal(self, root: Optional[TreeNode], result: List):
        if root is None:
            return

        self.get_postorderTraversal(root.left, result)
        self.get_postorderTraversal(root.right, result)
        result.append(root.val)

    #78. Subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        self.get_subsets(nums, result, [], 0)
        return result

    def get_subsets(self, nums: List[int], result: List, temp: List, start_index: int):
        result.append(list(temp))

        for i in range(start_index, len(nums)):
            if nums[i] in temp:
                return
            temp.append(nums[i])
            self.get_subsets(nums, result, temp, i+1)  #要記得如果使用start_index時 是目前走到的index＋1!!!!!!
            temp.pop()

    #230. Kth Smallest Element in a BST
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        qw = self.inorderTraversal_1(root, k)
        #bst 使用inorder 後就是有小排序到大 而且 左子樹一定小於根 右子樹一定大於根 所以當排序到第Ｋ個就可以停止了
        return qw

    #94. Binary Tree Inorder Traversal 145 Binary Tree postorder Traversal
    def inorderTraversal＿1(self, root: Optional[TreeNode], k: int):
        result = []
        self.get_inorderTraversal＿1(root, result, k)
        return result[k-1]

    def get_inorderTraversal＿1(self, root: Optional[TreeNode], result: List, k: int):
        if len(result) >= k or root is None:
            return
        else:
            self.get_inorderTraversal＿1(root.left, result, k)
            result.append(root.val)
            self.get_inorderTraversal＿1(root.right, result, k)

    def majorityElement(self, nums: List[int]) -> int:
        length = len(nums)/2
        result = {}

        for i in nums:
            if i in result:
                result[i] += 1
            else:
                result[i] = 1

            if result[i] > length:
                return i

    #283. Move Zeroes
    def moveZeroes(self, nums: List[int]) -> None:
        pos = 0 
        for i in range(len(nums)):
            if nums[i]:
                nums[pos] = nums[i]
                pos+=1

        for i in range(pos,len(nums)):
            nums[i] = 0

    def moveZeroes＿1(self, nums: List[int]) -> None:
        #[1,2,0,0,3,12]
        index = 0
        for i in range(len(nums)):
            if nums[i]:
                nums[index] = nums[i]
                index += 1

        for i in range(index, len(nums)):
            nums[i] = 0

    #238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 1,2,3,4
        length = len(nums)

        #left
        left = [1] * length
        for i in range(1, length):
            left[i]=(nums[i-1] * left[i-1])

        #right
        right = [1] * length
        for i in range(length-2, -1, -1):
            right[i] = nums[i+1] * right[i+1]

        #ans
        ans = [1] * length
        for i in range(length):
            ans[i] = left[i] * right[i]

        return ans

    #22. Generate Parentheses
    def generateParenthesis(self, n: int) -> List[str]:
        #剩餘數量 ')' count >  '('
        result = []
        self.get_generateParenthesis(n, "",0, 0, [])
        return result

    def get_generateParenthesis(self, n: int, curr: str, open_count: int, close_count: int, result: List):
        if len(curr) == n*2:
            result.append(curr)
        else:
            if open_count < n:
                self.get_generateParenthesis(n ,curr+"(", open_count+1, close_count, result)
            if close_count < open_count:
                self.get_generateParenthesis(n ,curr+")", open_count, close_count+1, result)

    #20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        open = ['(','[','{']
        stack = []

        for i in s:
            if i in open:
                stack.append(i)
            else:
                if len(stack) == 0:
                    return False
                if i == ")" and stack[-1] != "(":
                        return False
                elif i == "]" and stack[-1] != "[":
                    return False
                elif i == "}" and stack[-1] != "{":
                    return False
                else:
                    stack.pop()

        if len(stack) != 0:
            return False

        return True

    #49. Group Anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = dict()
        for word in strs:
            temp = []
            for letter in word:
                temp.append(letter)

            t = ''.join(sorted(temp))

            if result.get(t):
                result[t] = result[t].append(word)
                result[t] = w
            else:
                result[t] = [word]

        return list(result.values())

    #199. Binary Tree Right Side View
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        dic = {}

        self.get_rightSideView(root,0,dic)

        result = []
        for i in dic.values():
            result.append(i[0])

        return result

    def get_rightSideView(self, root: TreeNode,layer: int, dic: dict[int,list]):
        if root is  None:
            return
        if dic.get(layer) is None:
            dic[layer] = [root.val]

        self.get_rightSideView(root.right,layer+1,dic)
        self.get_rightSideView(root.left,layer+1,dic)

    #287. Find the Duplicate Number
    def findDuplicate(self, nums: List[int]) -> int:
        slow = 0
        fast = 0
        while True:
            fast = nums[fast]
            fast = nums[fast]
            slow = nums[slow]

            if slow == fast:
                break
        ori = 0
        while True:
            ori = nums[ori]
            slow = nums[slow]

            if ori == slow:
                return ori

        return -1

    #141. Linked List Cycle  ex. 3 2 0 -4  , -4 back to 2
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        try:
            slow = head
            fast = head.next.next

            while slow != fast:
                slow = slow.next
                fast = fast.next.next
        except:
            return None

        return slow

    #142. Linked List Cycle II
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None

        while head != slow:
            slow = slow.next
            head = head.next
        return head


    def get_bfs_tree(self,head: TreeNode):
        queue = [head]
        result = []
        while len(queue)>0:
            item = queue[0]
            del queue[0]

            result.append(item.val)

            if item.left is not None:
                queue.append(item.left)

            if item.right is not None:
                queue.append(item.right)

    #64. Minimum Path Sum
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        for i in range(m):
            for j in range(n):
                if i==0 and j == 0:
                    continue
                elif i == 0:
                   grid[0][j] = grid[0][j-1]+grid[0][j]
                elif j ==0:
                    grid[i][0] = grid[i-1][0]+grid[i][0]
                else:
                    grid[i][j] = min(grid[i][j-1],grid[i-1][j])+grid[i][j]

        return grid[m-1][n-1]

    #62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        grid = [[1]*m]*n

        for i in range(1,n):
            for j in range(1,m):
                    grid[i][j]= grid[i][j-1] +grid[i-1][j]

        return grid[-1][-1]

    #114. Flatten Binary Tree to Linked List
    def flatten(self, root: Optional[TreeNode]) -> None:
        if root is None:
            return  []

        prev = root
        stack = []

        if root.right is not None:
                stack.append(root.right)

        if root.left is not None:
                stack.append(root.left)

        while stack:
            item = stack.pop()
            prev.right = item
            prev.left = item

            if prev.right is not None:
                stack.append(prev.right)

            if prev.left is not None:
                stack.append(prev.left)
            prev = item


    def reverse_vowels(self,s: str):
        lis =list(s)
        vowels = ['a','e','i','o','u']
        i = 0
        j = len(s)-1
        while i <= j:
            while i < j and i !=j:
                if lis[i] in vowels:
                    break
                else:
                    i+=1
            while j > i and j!=i:
                if lis[j] in vowels:
                    break
                else:
                    j-=1
            if i!=j and i < j:
                lis[i] , lis[j] = lis[j] ,lis[i]
                i+=1
                j-=1

        return ''.join(lis)

    #費氏數列 fibonacci  1 1 2 3 5
    def fibonacci(self,i: int):
        if i < 2:
            if i == 0:
                return 0
            else:
                return 1

        return self.fibonacci(i-1) + self.fibonacci(i-2)

    #881. Boats to Save People
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        times = 0
        sort = sorted(people)
        start_index= 0

        for j in range(len(sort)-1,-1,-1):
            if sort[j] == limit:
                times+=1
                continue

            if j == start_index:
                times+=1
                break

            if limit - sort[j]- sort[start_index] >= 0:
                start_index+=1

            times+=1

            if j == start_index:
                break

        return times

    #941. Valid Mountain Array
    def validMountainArray(self, arr: List[int]) -> bool:
        if arr is None or len(arr) ==1 :
            return False
        i = 1
        while i < len(arr) and arr[i] > arr[i-1]:
            i+=1

        if i == len(arr) or arr[i] == arr[i-1]:
            return False

        while i < len(arr) and arr[i] < arr[i-1]:
            i+=1

        if arr[i] == arr[i-1]:
            return False

        return True

    #11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        #貪樊法 , 將比較低的木板往前移動 即能找到最大值
        result = 0
        length = len(height)
        i = 0
        j = length -1
        while i < j:
            lh = height[i]
            rh = height[j]

            area = (j-i) * min(lh,rh)

            if area > result:
                result = area

            if lh >= rh:
                j-=1
            else:
                i+=1
        return result

    #121. Best Time to Buy and Sell Stock
    #7,1,5,3,6,4
    def maxProfit(self, prices: List[int]) -> int:
        leng = len(prices)
        profit = 0
        min_buy = prices[0]

        for i in range(1,leng):
            if prices[i] < min_buy:
                min_buy = prices[i]

            if prices[i] - min_buy > profit:
                profit = prices[i]-min_buy

        return profit

    #53. Maximum Subarray
    #[-2,1,-3,4,-1,2,1,-5,4]
    def maxSubArray(self, nums: List[int]) -> int:
        sum = 0
        maxSum = nums[0]

        for num in nums:
            sum = sum + num
            if num > sum:
                sum = num

            if sum > maxSum:
                maxSum = sum

        return maxSum

    #2016. Maximum Difference Between Increasing Elements
    def maximumDifference(self, nums: List[int]) -> int:
        max = -1
        min =nums[0]

        for i in range(len(nums)):
            if nums[i] <  min:
                min = nums[i]

            temp = nums[i]- min
            if temp > max:
                max = temp

        return max

    #152. Maximum Product Subarray
    def maxProduct(self, nums: List[int]) -> int:
        max = nums[0]
        current = 1
        # 2 3 -1 5 6 2
        for num in nums:
            temp = num * current
            if temp >= max:
                max = temp
                current = temp
        return max

    #845. Longest Mountain in Array
    def longestMountain(self, arr: List[int]) -> int:
        if len(arr) < 3:
            return 0
        longest = 0
        index = 1
        length = len(arr)
        while index < length:
            up = 0
            while index < length:
                if arr[index] <= arr[index-1]:
                    break
                index+=1
                up+=1

            down = 0
            while(index < length):
                if arr[index] >= arr[index-1]:
                    break
                index+=1
                down+=1

            if index < length and arr[index] == arr[index-1]:
                index+=1

            if up > 0 and down >0:
                longest = max(longest,up+down+1)

        return longest

    #394. Decode String
    def decodeString(self, s: str) -> str:
    # 3[a2[c]]   =  acc acc acc
        open = []
        close = []
        nums = []
        result =[]
        for i in len(s):
          # find nums
            if s[i].isnumeric():
                temp = s[i]
                for j in range(i,len(s)):
                    if s[j].isnumeric():
                      temp+=s[j]
                    else:
                        nums.append(int(temp))
            else:
                if s[i] == "[":
                    open.append(i)
                    continue
                if s[i] == "]":
                    close.append(i)
                    continue
                result.append(s[i])

        i = len(open)-1
        j = 0

        while i > 0:
            temp = ""
            for k in range(open[i]+1,close[j]):
                temp+=s[k]

            result.append(nums.pop()*temp)
            i-=1
            j+=1



    #560. Subarray Sum Equals K
    # 3, 4, 7, 2, -3, 1, 4, 2 , k =7
    #pre_sum + dic
    def subarraySum(self, nums: List[int], k: int) -> int:
        dic = {0:1}
        count = 0
        temp = 0
        for i in range(0, len(nums)):
            temp += nums[i]  #現在的總和是多少
            diff = temp - k               #差值
            if diff in dic:             #尋找差值中出現的次數，不用擔心是否會多算 因為現在的結果是 前面累積來的
                count += dic[diff]

            if temp in dic:
                dic[temp] = dic[temp] +1
            else:
                dic[temp] = 1

        #{0, 1, 3, 6, 10, 15}
        return count

    #112. Path Sum
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False

        if root.left is None and root.right is None:
            if (targetSum-root.val) == 0:
                return True
            else:
                return False

        l = self.hasPathSum(root.left,targetSum-root.val)
        r = self.hasPathSum(root.right,targetSum-root.val)

        return l or r

    def getHasPathSum(self, root: Optional[TreeNode], targetSum: int):
        if root is None:
            return False

        if root.left is None and root.right is None:
            if (targetSum-root.val) == 0:
                return True
            else:
                return False

        l = self.getHasPathSum(root.left,targetSum-root.val)
        r = self.getHasPathSum(root.right,targetSum-root.val)

        return l or r


    #113. Path Sum II
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result = []
        self.getPathSum(root,targetSum,[],result)
        return result

    def getPathSum(self, root: Optional[TreeNode], targetSum: int,temp:List[int], result: List):
        if root is None:
            return

        if root.left is None and root.right is None:
            if targetSum-root.val == 0:
                temp.append(root.val)
                result.append(list(temp))
                temp.pop()

            return


        temp.append(root.val)
        self.getPathSum(root.left, targetSum-root.val, temp, result)
        self.getPathSum(root.right, targetSum-root.val, temp, result)
        temp.pop()
        count =0

    #437. Path Sum III
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        #not from the root so I should record every sum of nodes
        # dfs
        # 當要找尋連續串列中的連續子串列，使用 prefix_sum
        # 利用紀錄已出現的sum 找出subsequence
        self.count = 0
        temp = 0
        prefix_sum = {0:1}

        self.dfs(root, targetSum, prefix_sum, temp)

        return self.count

    def dfs(self, root, targetSum, prefix_sum, temp):
        if root is None:
            return

        temp += root.val
        diff = temp - targetSum

        if diff in prefix_sum:
            self.count += prefix_sum[diff]

        if temp in prefix_sum:
            prefix_sum[temp] += 1
        else:
            prefix_sum[temp] = 1

        self.dfs(root.left, targetSum, prefix_sum, temp)
        self.dfs(root.right, targetSum, prefix_sum, temp)
        prefix_sum[temp] -=1
        temp -= root.val

    #994. Rotting Oranges
    def orangesRotting(self, grid: List[List[int]]) -> int:
        #find 2 and 1 first
        m = len(grid)
        n = len(grid[0])
        rottens = deque()
        freshs = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 2:
                    rottens.appendleft([i,j])
                if grid[i][j] == 1:
                    freshs.append([i,j])

        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        times = 0
        q = deque()
        while freshs:
            rotten = rottens.popleft()
            for di in dirs:
                x = rotten[0]+di[0]
                y = rotten[1]+di[1]

                if x >= 0 and y >= 0:
                    temp = [x,y]
                if x >= 0 and y < 0:
                    temp = [x,rotten[1]]
                if x < 0 and y >= 0:
                    temp = [rotten[0],y]

                if temp in freshs:
                    freshs.remove(temp)
                    q.append(temp)

            if not rottens:
                times+=1
                rottens = deque(q)

        return times

    #record every later nodes
    def bfs(self,root: TreeNode):
        if root is None:
            return 0
        
        max_depth = 1
        res = {}
        queue = deque()
        queue.append((1,root))
                
        while queue:
            layer , item = queue.popleft()
            
            if res.get(layer):
                value = res[layer]
                value.append(item.val)
                res[layer] = value.append(item.val)
            else:
                res[layer] = [item.val]
            
            if item.left:
                queue.append((layer+1,item.left))
          
            if item.right:
                queue.append((layer+1,item.right)) 

            max_depth = max(max_depth,layer)      
        
        return max_depth

    #1302. Deepest Leaves Sum
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        dic = {}
        layer = 1
        self.getDeepestLeavesSum(root, dic ,layer)
        return dic[self.max_layer]
    
    max_layer = 1
    def getDeepestLeavesSum(self, root: Optional[TreeNode], dic: dict, layer) -> int:
        if root is None:
            return 
        
        if dic.get(layer):
            value = dic[layer]
            value += root.val
            dic[layer] = value
        else:
            dic[layer] = [root.val]

        self.getDeepestLeavesSum(root.left, dic, layer+1)
        self.getDeepestLeavesSum(root.right, dic, layer+1)

        if layer > self.max_layer:
           self.max_layer = layer
        
    #1769. Minimum Number of Operations to Move All Balls to Each Box
    def minOperations(self, boxes: str) -> List[int]:
        n = len(boxes)
        from_end = [0] * n
        from_end[-1] = int(boxes[-1])
        from_start =[int(boxes[0])]
        from_start_2 =[int(boxes[0])]

        result = [0] * n
        
        for i in range(1,n):
            from_start.append(from_start[i-1]+int(boxes[i]))
            
        for i in range(n-2,-1,-1):
            from_end[i] = from_end[i+1]+int(boxes[i])

        for i in range(n):
            result[i] = sum(from_start[:i]) + sum(from_end[i+1:])

        return result
    
    #654. Maximum Binary Tree            
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        m = max(nums)
        i = nums.index(m)
        l = nums[:i]
        r = nums[i+1:]

        tree = TreeNode(val=m)
        if l: 
            tree.left = self.constructMaximumBinaryTree(l)
        if r:
            tree.right = self.constructMaximumBinaryTree(r)

        return tree

    #1235. Maximum Profit in Job Scheduling 
    #把所有沒有重疊的地方相加找最大
    def jobScheduling_1(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        leng = len(startTime)
        dic = []
        for i in range(leng):
            dic.append(([startTime[i],endTime[i],i]))
        
        pass

    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        start = 0
        leng = len(startTime)
        max_profit = 0

        for i in range(leng):
            temp_profit = profit[i]
            ori_end_time = endTime[i]
            run_end_time = endTime[i]
            for j in range(i+1,leng):
                start_time = startTime[j]
                if start_time >= ori_end_time:
                    temp_profit += profit[j]
                if temp_profit > max_profit:
                    max_profit = temp_profit

        return max_profit

    def searchInsert(self, nums: List[int], target: int) -> int:
        i = 0
        j = len(nums) - 1

        while i <= j :
            if nums[i] > target:
                return i
            if nums[j] < target:
                return j + 1

            mid = (j + i) // 2
            if nums[mid] == target:
                return mid

            if nums[mid] < target:
                i = mid + 1

            if nums[mid] > target:
                j = mid -1

        return i

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        #做兩次 binary search 找頭跟尾
        def search_for(nums, target, left = True):
            i = 0
            j = len(nums) 
            while i < j:
                mid = (i+j) / 2
                if nums[mid] == target:
                    if left:
                        j = mid 
                    else:
                        i = mid + 1
                elif nums[mid] < target:
                    i = mid + 1
                else:
                    j = mid 
            return i
        left = search_for(nums, target, True)
        right = search_for(nums, target, False)
        if not nums:
            return[-1,-1]
        elif 0 <= left < len(nums) and nums[left] == target:
            return [left, right-1]
        else:
            return [-1,-1]

    def shipWithinDays(self, weights: List[int], days: int) -> int:
        left = max(weights)
        right = sum(weights)

        while left < right:
            mid = ( right + left) // 2

            current_weight = 0
            need_days = 1
            for w in weights:
                current_weight += w
                if current_weight > mid:
                    need_days +=1
                    current_weight = w

            if need_days > days:
                left = mid + 1
            else:
                right = mid

        return left


    def validPalindrome(self, s: str) -> bool:
        i = 0
        j = len(s) - 1

        def check_palindrome(i ,j):
            while i < j:
                if s[i] == s[j]:
                    i += 1
                    j -= 1
                else:
                    return False
            return True

        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                left = check_palindrome(i+1 , j)
                right = check_palindrome(i, j-1)
                return left or right

        return True 

    def nthUglyNumber(self, n):
        heap = [1]
        counter = 1
        while counter <= n:
            number = heapq.heappop(heap)
            counter += 1
        
            if number % 5 == 0:
                heapq.heappush(heap, 5 * number)
                continue
        
            if number % 3 == 0:
                heapq.heappush(heap, 3 * number)
                heapq.heappush(heap, 5 * number)
                continue
        
            heapq.heappush(heap, 2 * number)
            heapq.heappush(heap, 3 * number)
            heapq.heappush(heap, 5 * number)
        
        return number

    def reorganizeString(self, s: str) -> str:
        _len = len(s)
        count = collections.Counter(s)
        c, f = count.most_common(1)[0]
        if 2 * f - 1 > _len:
            return ''
        
        que = [(-v, c) for c, v in count.items()]
        
        heapq.heapify(que)

        res = ""
        while _len:
            cnt = min(_len, 2) #每次取兩個剩餘最多的字母 
            temp = list()
            for i in range(cnt):
                if not que:
                    return ""
                v, c = heapq.heappop(que)
                res += c

                if v + 1 !=0:
                    temp.append((v+1, c))

                _len -= 1

            for x in temp:
                heapq.heappush(que, x)

        return res

    #253 Meeting Rooms II
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        heap = []
        count = 0
        for i in intervals:
            if i:
                heap.append(i)

        heapq.heapify(heap)
        heapq.h
        # i =heapq.heappop(heap)
        end_times = []

        while heap:
            meeting = heapq.heappop(heap)
            start_time = meeting[0]
            end_time = meeting[1]
            end_times.append(end_time)

            heapq.heapify(end_times)

            if start_time < end_times[0]:
                count += 1
            else:
                del end_times[0]


        return count


    def numBusesToDestination(self, routes, S, T):
        #[[1,2,7],[3,6,7]]
        to_route = defaultdict(set)
        # announce a dic , but the default is a set
        # set can use "add" and "remove"
        for i, route in enumerate(routes):
            for station in route:
                to_route[station].add(i)

        bfs = [(S,0)]
        seen = [S]

        for station, bus in bfs:
            if station == T:
                return bus
            for i in to_route[station]:  #跟該車站有連結的bus index = i
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus+1))
                        seen.append(j)

                routes[i] = []  # seen route
        return -1


    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None and q == None:
            return True
        elif p == None and q != None:
            return False
        elif p != None and q == None:
            return False
        else:
            if q.val != p.val:
                return False
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        queue = collections.deque([(root, 1)])

        while queue:
            w = queue.popleft()
            item = w[0]
            layer = w[1]
            if len(result) < layer:
                result.append([])

            result[layer-1].append(item.val)

            if item.left:
                queue.append([item.left, layer+1])
            if item.right:
                queue.append([item.right, layer+1])

        return result

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        queue = collections.deque([root])
        res = []
        flag = True #left->right
        while queue:
            tmp = []
            for _ in range(len(queue)):
                if flag:
                    node = queue.popleft()
                    tmp.append(node.val)
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
                else:
                    node = queue.pop()
                    tmp.append(node.val)
                    if node.right:
                        queue.appendleft(node.right)
                    if node.left:
                        queue.appendleft(node.left)
            flag = not flag
            res.append(tmp)
        return res


    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        result = collections.deque([])
        queue = collections.deque([root])

        while queue:
            temp = []
            for _ in range(len(queue)):
                item = queue.popleft()
                temp.append(item.val)
                if item.left:
                    queue.append(item.left)
                if item.right:
                    queue.append(item.right)

            result.appendleft(list(temp))

        return result

    min_depth = 0
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return []
        # using bfs is better

        queue = collections.deque([(root, 1)])

        while queue:
            item, layer = queue.popleft()
            if item.left is None and item.right is None:
                return layer

            if item.left:
                queue.append([item.left, layer+1])
            if item.right:
                queue.append([item.right, layer+1])

    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        if right == 0 and nums[0] == target:
            return 0

        while left < right :
            mid = (left+right) // 2

            if target == nums[mid]:
                return mid

            if target < nums[mid]:
                right = mid

            if target >= nums[mid]:
                left = mid+1
        return -1

    def singleNonDuplicate(self, nums: List[int]) -> int:
        b, e = 0, len(nums) - 1
        while b < e:
            m = (b + e) // 2
            if m % 2 == 1 and nums[m - 1] == nums[m]:
                b = m + 1
            elif  m % 2 == 0 and nums[m + 1] == nums[m]:
                b = m + 2
            else:
                e = m
        return nums[b]

    def letterCasePermutation(self, s: str) -> List[str]:
        result = []
        self.getLetterCasePermutation(s, 0, "", result)

        return result

    def getLetterCasePermutation(self, s: str ,index: int, temp: str, result):
        if len(temp) == len(s):
            result.append(temp)
            return

        for i in range(index, len(s)):
            if str.isalpha(s[i]):
                self.getLetterCasePermutation(s, i+1, temp+s[i].lower(), result)
                self.getLetterCasePermutation(s, i+1, temp+s[i].upper(), result)
            else:
                temp+=s[i]
                self.getLetterCasePermutation(s, i+1, temp, result)

    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1

        if len(nums) == 1:
            if nums[0] == target:
                return 0
            else:
                return -1

        left = 0
        right = len(nums) - 1
        pivot = 0
        if nums[left] < nums[right]: # 如果一開始就發現 沒有pivot
            pivot = 0
        else:
            while left <= right:
                pivot = (left + right) // 2
                if nums[pivot] > nums[pivot + 1]:
                    pivot += 1
                    break
                else:
                    if nums[pivot] > nums[left]:
                        left = pivot + 1
                    else:
                        right = pivot - 1

        if target <= nums[-1]:
            ans = self.binary_search(nums[pivot:], target)
            if ans == -1:
                return -1
            return pivot + ans
        else:
            return self.binary_search(nums[:pivot], target)


    def binary_search(self, nums, target):
        left = 0
        right = len(nums)-1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid
        return -1

    def countNegatives(self, grid: List[List[int]]) -> int:
        count = 0
        for g in grid:
            c = len(g)- self.getNegatives(g)
            count += c

        return count

    def getNegatives(self, arr: List[int]):
        left = 0
        right = len(arr)-1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] > 0:
                left = mid + 1
            elif arr[mid] == 0:
                return mid + 1
            else:
                right = mid -1

        return left

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for m in matrix:
            left = 0
            right = len(m) - 1
            if m[left] <= target and target <= m[right]:
                r = self.binary_search(m, target)
                if r:
                    return True
            else:
                continue

        return False


    def binary_search(self, nums: List[int], target) -> bool:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid -1
            else:
                return True

        return False

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        leng = len(nums)
        dic = {}
        for i in range(leng):
            temp = target - nums[i]

            if temp in dic:
                return [dic[temp], i]

            if not dic.get(nums[i]):
                dic[nums[i]] = i

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        #graph dfs
        result = []

        self.getPath(graph, 0, result, [0])
        return result

    def getPath(self, graph ,index, result, path):
        if path[-1] == len(graph) - 1:
            result.append(list(path))
            return

        for j in graph[index]:
            self.getPath(graph, j, result, path+[j])

    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        # 1 to n
        # out degree:trust someone
        # in degree: someone been trust
        #judge: out degree is 0, in degree is n-1
        indegree = [0] * (n + 1)
        outdegree = [0] * (n + 1)
        for tru in trust:
            out = tru[0]
            in_ = tru[1]
            outdegree[out] = outdegree[out] + 1
            indegree[in_] = indegree[in_] + 1

        for i in range(1, n + 1):
            if outdegree[i] == 0 and indegree[i] == n-1:
                return i
        return -1

    def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        graph = defaultdict(list)
        for u, v in edges: # 雙向連結
            graph[u].append(v)
            graph[v].append(u)

        visited = set()
        path = []
        def dfs(cur):
            visited.add(cur)
            path.append(cur)

            if cur == end:
                print(path)
                return True

            for neibor in graph[cur]:
                if neibor not in visited:
                    if dfs(neibor):
                        return True

            path.pop()  # 如果走到死路的話 先刪除
            return False

        r = dfs(start)
        return r



    def validPath_bfs(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        grpah = defaultdict(list)
        for u, v in edges:
            grpah[u].append(v)
            grpah[v].append(u)

        queue = deque([start])
        visited = set()
        while queue:
            cur = queue.popleft()
            visited.add(cur)
            if cur == end:
                return True

            dis = grpah[cur]
            for d in dis:
                if d not in visited:
                    queue.append(d)

        return False



    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(i):
            seen[i] = 1
            for j in range(n):
                if isConnected[i][j] == 1 and seen[j] == 0:
                    dfs(j)

        n = len(isConnected)
        res = 0
        seen = [0]*n
        for i in range(n):
            if seen[i] == 0:
                dfs(i)
                res += 1
        return res

    def solution(self, a):
        dic = collections.defaultdict(int)
        result = []
        max = 0
        for item in a:
            num = str(item)
            for n in num:
                dic[n] +=1
                if dic[n] > max:
                    result = []
                    max = dic[n]
                if dic[n] == max:
                    result.append(n)
        result.sort()
        return result


    def jump(self, nums: List[int]) -> int:
        count = 0
        farst_position = 0
        current_position = 0
        for i in range(len(nums)-1):
            farst_position = max(farst_position, i + nums[i])

            if i == current_position:
                current_position = farst_position
                count+=1
        return count

    def canJump(self, nums: List[int]) -> bool:
        r = farthest = 0
        while r <= farthest:
            if farthest >= len(nums) - 1:
                return True
            farthest = max(farthest, r + nums[r])
            r += 1
        return False

    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        trucks = 0
        boxTypes.sort()
        boxTypes.sort(key=lambda x: -x[1])

        total_units = 0
        for truck, units in boxTypes:
            if trucks >= truckSize:
                break
            last_spaces = truckSize - trucks
            if last_spaces >= truck:
                trucks += truck
                total_units += truck * units
            else:
                trucks += last_spaces
                total_units += last_spaces * units

        return total_units

    def largestPerimeter(self, nums: List[int]) -> int:
        # 兩邊之合 > 第三邊 , sotr reverse , take max
        nums.sort(reverse = True)
        max_length = 0
        i = 2
        while i < len(nums):
            a = nums[i - 2]
            b = nums[i - 1]
            c = nums[i]

            if b + c > a and a + b + c > max_length:
                max_length = a + b + c
            i +=1
        return max_length

    def largestNumber(self, nums: List[int]) -> str:

        def cmp_func(x, y):
            """Sorted by value of concatenated string increasingly."""
            if x + y > y + x:
                return 1
            elif x == y:
                return 0
            else:
                return -1

        # Build nums contains all numbers in the String format.
        nums = [str(num) for num in nums]

        # Sort nums by cmp_func decreasingly.
        nums.sort(key = cmp_to_key(cmp_func), reverse = True)

        # Remove leading 0s, if empty return '0'.
        return ''.join(nums).lstrip('0') or '0'


    def minAddToMakeValid(self, s: str) -> int:
        temp = []

        for i in s:
            if i == "(":
                temp.append("(")
            else:  #")"
                if temp and temp[-1] == "(":
                    temp = temp[:-1]
                else:
                    temp.append(")")
        return len(temp)

    def connectSticks(self, sticks: List[int]) -> int:
        # find min of two
        result = 0
        heapq.heapify(sticks)
        i = 2
        while len(sticks) > 1:
            a = heapq.heappop(sticks)
            b = heapq.heappop(sticks)
            heapq.heappush(sticks, a+b)
            result += a + b

        return result

    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        # find min count remove
        dic = collections.defaultdict(int)
        for a in arr:
            dic[a] += 1

        sort_dic = sorted(dic.items(), key = lambda x:x[1])
        remove_count = 0
        for _, value in sort_dic:
            if k == 0:
                break
            if k >= value:
                k -= value
                remove_count += 1
            else:
                k -= k

        return len(sort_dic) - remove_count

    def maxProfit(self, prices: List[int]) -> int:
        leng = len(prices)
        profit = 0
        min_buy = prices[0]

        for i in range(1,leng):
            if prices[i] < min_buy:
                min_buy = prices[i]


            p1 = prices[i]-min_buy
            if p1 <= 0:
                continue

            p2 = self.maxProfit(prices[i:])
            profit = max(p1+p2, profit)

        return profit

    def lengthOfLIS(self, nums: List[int]) -> int:
        track = []
        for n in nums:
            if not track or n > track[-1]:
                track.append(n)
                continue
            ind = bisect.bisect_left(track, n)
            track[ind] = n
        return len(track)

    def increasingTriplet(self, nums: List[int]) -> bool:
        d = []
        for x in nums:
            i = bisect.bisect_left(d, x)  #會回傳從 d 中要插入的 index
            if i == len(d): # 當index與長度相同時 代表是要插入到最後一個位置（有序）
                d.append(x)
                if len(d) == 3:
                    return True
            else:
                d[i] = x  #當回傳的index 小於 d 長度時代表

        return False

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * len(cost)
        dp[0] = cost[0]
        dp[1] = cost[1]

        for i in range(2, len(cost)):
            a1 = dp[i-2]
            a2 = dp[i-1]
            dp[i] = min(a1, a2) + cost[i]

        return min(dp[-1], dp[-2])

    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0

        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1

        if i == len(s):
            return True

        return False

    def minDistance(self, word1: str, word2: str) -> int:
        len1 = len(word1)
        len2 = len(word2)

        #dp = [[0] * (len + 1)] * (len2 + 1) 這是錯的！！！！！ 會是參考型別的複製！！！！
        dp =  [[ 0 for _ in range(len2+1) ] for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i

        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[len1][len2]

    def stoneGame(self, piles: List[int]) -> bool:
        pass

    def PredictTheWinner(self, nums: List[int]) -> bool:
        dp = [[ [0] for _ in range(len(nums))] for _ in range(len(nums))]

        return self.Winner(0, len(nums) - 1, nums, dp) >= 0

    def Winner(self, s, e, nums, dp):
        if s == e:
            return nums[s]

        if dp[s][e] > 0:
            return dp[s][e]

        a = self.Winner(s + 1, e, nums, dp)
        b = self.Winner(s, e - 1, nums, dp)

        dp[s][e] = max(nums[s] - a, nums[e] - b)

        return dp[s][e]

    def partition(self, s: str) -> List[List[str]]:
        result = []
        dp = [[ None for _ in range(len(s) + 1)] for _ in range(len(s) + 1)]
        self.getPartition(s, 0, len(s), result, [], dp)
        return result

    def getPartition(self, s, start, N, result , path, dp):
        if start == N:
            result.append(list(path))
            return

        for end in range(start + 1, N + 1):
            if dp[start][end]:
                self.getPartition(s, end, N, result, path + [s[start:end]], dp)
            else:
                t = s[start:end]
                r = t[::-1]
                if t == r:
                    dp[start][end] = True
                    self.getPartition(s, end, N, result, path + [t], dp)  #使用path + [t] 是為了不pop 否則要加一行 path.pop()

    def canPermutePalindrome(self, s: str) -> bool:
        leng = len(s)
        dic = defaultdict(int)

        for i in range(leng):
            if s[i] not in dic:
                dic[s[i]] = 1
            else:
                del dic[s[i]]

        if leng % 2 == 0 and len(dic) == 0:
            return True

        if leng % 2 == 1 and len(dic) == 1 :
            if list(dic.values())[0] == 1:
                return True

        return False

    def isAnagram(self, s: str, t: str) -> bool:
        dic = defaultdict(int)
        for i in range(len(s)):
            if s[i] in dic:
                dic[s[i]] += 1
            else:
                dic[s[i]] = 1
        
        for j in range(len(t)):
            if t[j] not in dic:
                return False
            else:
                dic[t[j]] -= 1
                if dic[t[j]] < 1:
                    return False
                
        for v in dic.values():
            if v != 0:
                return False
            
        return True


    # def longestPalindrome(self, s):
    #     dic = collections.Counter(s)
    #     res = 0
    #     isOdd = False
    #     for key, value in dic.items():
    #         if value % 2 == 0:
    #             res += value
    #         else:
    #             if isOdd:
    #                 res += value - 1
    #             else:
    #                 isOdd = True
    #                 res += value
    #     return res

    def longestPalindrome(self, s: str) -> str:
        leng = 0
        start = 0
        end = 0
        for i in range(len(s)):
            cur, st, e = max(self.getLen(i, i, s), self.getLen(i, i + 1, s))

            if cur > leng:
                leng = cur
                start = st
                end = e


        return s[start:end]
    # 如果是回文的話 左邊 －１ 右邊 ＋ １ 也相等的話也會是回文
    def getLen(self, l, r, s):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1

        return r - l -1, l + 1 ,r

    def generatePalindromes(self, s: str) -> List[str]:
        result = []
        self.getePalindromes(s, "", result, [])

        return result

    def getePalindromes(self, s, temp, result, paths):
        if len(temp) == len(s):
            r = temp[::-1]
            if temp == r and temp not in result:
                result.append(temp)
            return

        for i in range(len(s)):
            if i in paths:
                continue

            self.getePalindromes(s, temp + s[i], result, paths + [i])


    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        result = []
        c = collections.Counter(nums)
        self.getPermute(nums, [], result, c)
        return result

    def getPermute(self, nums, temp, result, c):
        if len(nums) == len(temp):
            result.append(list(temp))
            return
 
        for i in c:
            if c[i] > 0:
                c[i] -= 1
                temp.append(i)
                self.getPermute(nums, temp, result, c)
                c[i] += 1
                temp.pop()

    def generatePalindromes(self, s: str) -> List[str]:
        counter = collections.Counter(s)
        odds = []
        res = []
        n = len(s)
        for key, value in counter.items():
            if value % 2 != 0:
                odds.append(key)

        if len(odds) > 1:
            return []

        if len(odds) == 1:
            counter[odds[0]] -= 1
            self.help(counter, odds[0], res, n)
        else:
            self.help(counter, "", res, n)

        return res

    def help(self, counter, temp, res, n):
        if len(temp) == n:
            res.append(temp)
            return

        for key, value in counter.items():
            counter[key] -= 2
            self.help(counter, key+temp+temp, res, n)
            counter[key] += 2


    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            next = head.next #先把下一個位置取出來備用
            head.next = prev
            prev = head
            head = next

        return prev



    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        current_node = head
        end = None
        left_count = 1
        while left_count != left:
                current_node = current_node.next
                left_count += 1

        end = self.reverseList(current_node, right - left + 1)

        current_node = end
        head.next = current_node
        return head

    def reverseList(self, head, target):
        prev = None
        count = 1
        while count <= target:
            next = head.next
            head.next = prev
            prev = head
            head = next
            count += 1

        r = prev
        c = r

        while c.next:
            c = c.next
        c.next = head

        return r

    def substr_finder(self,s: str, alphabet: set[str]) -> str:
        max_str = ""
        now_compare = ""
        temp = ""

        for i in range(len(s)):
            if s[i] not in alphabet:
                now_compare = ""
                temp = ""
                continue

            if now_compare == s[i]:
                temp += s[i]
            else:
                now_compare = s[i]
                temp = s[i]

            if len(temp) > len(max_str):
                    max_str = temp

        return max_str
    def merge(self, lists: List[List[int]]) -> List[int]:
        from heapq import merge
        for i  in range(1, len(lists)):
           lists[0] = list(merge(lists[0], lists[i]))

        return lists[0]

    def solution(self,A):
        sum = 0
        n = len(A)
        m = len(A[0])
        if n == 0 or m == 0:
            return 0

        B = [[ 1 for _ in range(m)] for _ in range(n)]


        for i in range(n):
            for j in range(m):
                if B[i][j] > 0:
                    self.checkNearIsIslane(A, B, i, j, n, m)
                    sum += 1

        return sum

    def checkNearIsIslane(self, A, B, i, j, n, m):
        if B[i][j] == -1:
            return
        B[i][j] = -1

        if i + 1 < n:
            if A[i+1][j] == A[i][j]:
                self.checkNearIsIslane(A, B, i + 1, j, n, m)

        if i - 1 >= 0:
            if A[i - 1][j] == A[i][j]:
                self.checkNearIsIslane(A, B, i - 1, j, n, m)

        if j + 1 < m:
            if A[i][j + 1] == A[i][j]:
                self.checkNearIsIslane(A, B, i, j + 1, n, m)

        if j - 1 >= 0:
            if A[i][j - 1] == A[i][j]:
                self.checkNearIsIslane(A, B, i, j - 1, n, m)

    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        paths = defaultdict(list)
        for path in edges:
            ori = path[0]
            dis = path[1]
            paths[ori].append(dis)
            paths[dis].append(ori)

        count = 0
        node = 0
        visited = set()

        for node in range(n):
            if node not in visited:
                count += 1
                visited.add(node)
                self.dfs(node, paths, visited)

        return count

    def dfs(self, node, paths, visted):
        for neighbor in paths[node]:
            if neighbor not in visted:
                visted.add(neighbor)
                self.dfs(neighbor, paths, visted)

    def numIslands(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])

        visited = [[ False for _ in range(m)] for _ in range(n)]
        count = 0
        for i in range(n):
            for j in range(m):
                if not visited[i][j] and grid[i][j] == '1':
                    count += 1
                    self.dfs(grid, visited, i, j, n, m)

        return count

    def dfs(self, grid, visited, i, j, n , m):
        if not visited[i][j]:
            visited[i][j] = True
            if grid[i][j] == '1':
                if i > 0:
                    self.dfs(grid, visited, i - 1, j ,n ,m)
                if i + 1 < n:
                    self.dfs(grid, visited, i + 1, j ,n ,m)
                if j  > 0:
                    self.dfs(grid, visited, i, j - 1, n, m)
                if j + 1 < m:
                    self.dfs(grid, visited, i, j + 1, n, m)

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # single graph
        # visitedCourses
        def dfs(i):
            color[i] = 1 #表示開始繞行i以後的子孫
            if i in graph:
                for j in graph[i]:
                    if color[j] == 0:
                        if not dfs(j):
                            return False
                    elif color[j] == 1: #如果中的途中遇見等於１的表示其為環狀結構
                        return False
            color[i] = 2 #標記成2的意思是以i為啟點的subpath已經完成且並無環狀,代表i的子孫們已經走過
            return True

        graph = {}
        for pair in prerequisites:
            if pair[1] in graph:
                graph[pair[1]].add(pair[0])
            else:
                graph[pair[1]] = set([pair[0]])

        color = [0]*numCourses

        for i in range(numCourses):
            if color[i] == 0:
                if not dfs(i):
                    return False

        return True

    def exist(self, board, word):
        n = len(board)
        m = len(board[0])

        visited = [[ False for _ in range(m)] for _ in range(n)]

        def dfs(i , j, k):
            if board[i][j] != word[k]:
                return False
            else:
                if k == len(word) - 1:
                    return True

                visited[i][j] = True

                if i > 0 and not visited[i-1][j] and dfs(i-1, j ,k + 1):
                    return True

                if i < n - 1 and not visited[i + 1][j] and dfs(i + 1, j ,k + 1):
                    return True

                if j > 0 and not visited[i][j - 1] and dfs(i, j - 1, k + 1):
                    return True

                if j < m - 1 and not visited[i][j + 1] and dfs(i, j + 1, k + 1):
                    return True

            visited[i][j] = False

            return False

        for i in range(n):
            for j in range(m):
                if dfs(i, j ,0):
                    return True

        return False

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if s == "":
            return True
        dic = defaultdict(bool)

        def wordBreak(r):
            if r == "":
                return True
            if r in dic:
                return dic[r]

            for i in range(1, len(r) + 1):
                temp = r[:i]
                if (temp in dic and dic[temp]):
                    return True
                elif temp in wordDict:
                    dic[temp] = True
                    if wordBreak(r[i:]):
                        return True
                else:
                    dic[temp] = False

            return False

        return wordBreak(s)

    def lengthOfLastWord(self, s: str) -> int:
        li = s.split(' ')
        result = 0
        for latter in li:
            if latter == ' ' or latter == '':
                continue
            result = len(latter)
        return result

    def plusOne(self, digits: List[int]) -> List[int]:
        s = ""
        for i in digits:
            s += str(i)
        l = int(s) + 1
        r = list(str(l))
        return r

    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        result = ListNode(0)
        result.next = head
        prev = result
        curr = head
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next

        return result.next

    def titleToNumber(self, columnTitle: str) -> int:
        # 26**2*i + 26**1*i + 26**0*i
        leng = len(columnTitle)
        total = 0
        alpha_map = {chr(i + 65): i + 1 for i in range(26)}

        for i in range(leng):
            cur_char = columnTitle[leng - 1 - i]
            total += (alpha_map[cur_char] * (26 ** i))

        return total

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        result = []
        self.dfs(root,result,"")
        return result

    def dfs(self, root: Optional[TreeNode], result, temp):
        if root is None:
            return

        temp += str(root.val)
        if root.left is None and root.right is None:
            result.append(temp)
            return

        temp += "->"
        self.dfs(root.left, result, temp)
        self.dfs(root.right, result, temp)

    def reverseWords(self, s: str) -> str:
        result = []
        w = s.split()
        for l in w:
            r = l[::-1]
            result.append(r)
        k = ' '.join(result)
        return k

    def sumNumbers(self, root: Optional[TreeNode]) -> int:

        self.result = 0
        def dfs(root, temp):
            if root is None:
                return

            if root.left is None and root.right is None:
                temp.append(str(root.val))
                f = ''.join(temp) # string
                self.result += int(f)
                return

            dfs(root.left, temp + [str(root.val)])
            dfs(root.right, temp + [str(root.val)])

        dfs(root, [])

        return self.result

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        #if temp + root.val < temp : break
        # root.left , root.right , root
        l = self.dfs(root.left, 0)
        r = self.dfs(root.right, 0)
        v = self.dfs(root, 0)

        return max(l, r, v)

    def dfs(self, root, temp):
        if root is None:
            return

        if temp + root.val > temp:
            temp += root.val

        l = self.dfs(root.left, temp)
        r = self.dfs(root.right, temp)

        return max(l , r)


    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        self.result = 0
        def dfs(root, temp):
            if root is None:
                return 0

            if root.val == temp:


                return max(l_max, r_max) + 1
            else:
                temp = root.val

            l_max = dfs(root.left, temp)
            r_max = dfs(root.right, temp)
            return max(l_max, r_max) + 1

        dfs(root, root.val)

        return self.result

    def longestUnivaluePath(self, root):
        self.ans = 0

        def arrow_length(node):
            if not node:
                return 0

            left_length = arrow_length(node.left)
            right_length = arrow_length(node.right)

            if node.left and node.left.val == node.val:
                left_arrow = left_length + 1
            else:
                left_arrow = 0

            if node.right and node.right.val == node.val:
                right_arrow = right_length + 1
            else:
                right_arrow = 0

            self.ans = max(self.ans, left_arrow + right_arrow)

            return max(left_arrow, right_arrow)

        arrow_length(root)

        return self.ans


    def maxPathSum(self, root):
        self.result = float('-inf')
        # ms
        # root must be used
        # at most one children can be return

        def ms(root):
            if root is None:
                return 0

            l_max =  max(ms(root.left), 0)
            r_max = max(ms(root.right), 0)

            self.result = max(self.result, l_max + r_max + root.val)

            return max(l_max, r_max) + root.val

        ms(root)

        return self.result

#669. Trim a Binary Search Tree
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        def trim(node):
            if node is None:
                return None
            if node.val > high:
                return trim(node.left)
            elif node.val < low:
                return trim(node.right)
            else:
                node.left = trim(node.left)
                node.right = trim(node.right)
                return node

        return trim(root)

    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        # dfs  find leaf
        # from buttom to up in case remain target

        if root is None:
                return None

        root.left = self.removeLeafNodes(root.left, target)
        root.right = self.removeLeafNodes(root.right, target)

        if root.left is None and root.right is None and root.val == target:
            return None
        else:
            return root

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result = []
        if root is None:
            return result

        #dfs
        def dfs(root, targetSum, temp):
            if root is None:
                return

            if root.left is None and root.right is None:
                if targetSum == root.val:
                    temp.append(root.val)
                    result.append(temp)

            dfs(root.left, targetSum - root.val, temp + [root.val])
            dfs(root.right,targetSum - root.val, temp + [root.val])

        dfs(root, targetSum, [])

        return result

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # dfs
        self.count = 0
        def dfs(root,targetSum):
            if root is None:
                return 0

            l = root.val + dfs(root.left, targetSum, root.left.val)
            r = root.val + dfs(root.right, targetSum, root.right.val)

            if l == targetSum:
                self.count += 1

            if r == targetSum:
                self.count += 1

            return
        dfs(root, targetSum)

        return self.count

    def rob(self, root: Optional[TreeNode]) -> int:
        #dfs 一定要間隔
        # max sum from root to leaf but not connected
        # 1 3 5 7
        # 2 4 6 8
        # slow and fast pointer?
        self.total = 0
        def dfs(root):
            if root is None:
                return

            self.total += root.val

            if root.left:
                if root.left.left:
                    dfs(root.left.left)
                if root.left.right:
                    dfs(root.left.right)

            if root.right:
                if root.right.left:
                    dfs(root.right.left)
                if root.right.right:
                    dfs(root.right.right)

        dfs(root)
        first = self.total
        self.total = 0

        dfs(root.left)
        left = self.total
        self.total = 0

        dfs(root.right)
        right = self.total
        self.total = 0


        return max(first, left + right)

    def rob(self, nums: List[int]) -> int:
        #dp
        if len(nums) < 2:
            return nums[0]

        if len(nums) < 3:
            return max(nums[0], nums[1])

        moneys = [0]* len(nums)
        moneys[0] = nums[0]
        moneys[1] = nums[1]

        # 3 => max(nums[0], nums[1])
        for i in range(2, len(nums)):
            temp_max = max(moneys[i-2], moneys[i-3])
            moneys[i] = temp_max + nums[i]

        return max(moneys[-1], moneys[-2])

    def rob(self, nums: List[int]) -> int:
        self.dic = {}

        res = self.r(nums, len(nums) - 1)

        return res


    #divide question into small question
    def r(self, nums, i):
        if i < 0:
            return 0

        if i in self.dic:
            return self.dic[i]

        res = max(self.r(nums, i-2) + nums[i], self.r(nums, i-1))
        self.dic[i] = res

        return res

    def rob(self, root: Optional[TreeNode]) -> int:
        self.ans = []
        moneys = []

        self.dfs(root, moneys)
    
        return self.ans

    def dfs(self, root, moneys):
        if root is None:
            return 0

        p1 = 0
        p2 = 0
        if len(moneys) > 1:
            p2 = moneys[-2]

        if len(moneys) > 0:
            p1 = moneys[-1]


        temp_max = max(p2 + root.val, p1)
        moneys.append(temp_max)

        if root.left is None and root.right is None:
            self.ans.append(temp_max)

        self.dfs(root.left, moneys)
        self.dfs(root.right, moneys)
        moneys.pop()



    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        # dfs start with last 2 node,
        #record the node install or not
        #if the node install => children can't

        #if the node not install =>children 2 choice
        if root.left is None and root.right is None:
            return 1

        l1 = self.dfs(root.left, False)
        r1 = self.dfs(root.right, False)

        l2 = self.dfs(root.left, True)
        r2 = self.dfs(root.right, True)

        return min(l1 + r1, l2 + r2 + 1)

    def dfs(self, root, is_parent_install):
        if root is None:
            return 0

        if root.left is None and root.right is None:
            return 0

        if is_parent_install:
            l = self.dfs(root.left, False)
            r = self.dfs(root.right, False)
            return l + r
        else:
            l_i = self.dfs(root.left, True)
            r_i = self.dfs(root.right, True)

            return l_i + r_i + 1


    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        # dfs
        # find leafs
        # while
        self.result = []
        self.temp = []
        while root.left is not None or root.right is not None:
            root = self.dfs(root)
            self.result.append(self.temp)
            self.temp = []

        self.result.append([root.val])
        return self.result


    def dfs(self,root):
        if root is None:
            return

        if root.left is None and root.right is None:
            self.temp.append(root.val)
            root = None
            return

        root.left = self.dfs(root.left)
        root.right = self.dfs(root.right)

        return root


    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # dfs inorder L V R
        def helper(root, min, max):
            if root is None:
                return False

            if root.val <= min or root.val >= max:
                return False

            left = helper(root.left, min, root.val)
            right = helper(root.right, root.val, max)

            return left or right

        return helper(root, -math.inf, math.inf)


    def swap(self, nums):
        n = len(nums)
        x = y = None # Initialize x and y as a value that cannot be the value of a node.
        for i in range(n - 1):
            if nums[i + 1] < nums[i]:
                y = nums[i + 1]
                # first swap occurrence
                if x is None:
                    x = nums[i]
                # second swap occurrence
                else:
                    break
        return x, y

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

        res = self.buildTree(nums)

        return res

    def buildTree(self, nums):
        n = len(nums)

        if n == 0:
            return None

        mid =  n // 2
        root = TreeNode(nums[mid])

        root.left = self.buildTree(nums[:mid:])
        root.right = self.buildTree(nums[mid + 1::])

        return root

    def coinChange(self, coins: List[int], amount: int) -> int:
        # 樹的遍歷，
        #top buttom
        self.dic = {0: 0}
        self.helper(coins, amount)
        if self.dic[amount] == float('inf'):
            return -1

        return self.dic[amount]

    def helper(self, coins, amount):
        if amount in self.dic:
            return self.dic[amount]

        m = float('inf')

        for coin in coins:
            if amount - coin < 0:
                continue

            count = self.helper(coins, amount - coin) + 1

            m = min(m, count)


        self.dic[amount] = m

        return m

    def findTargetSumWays(self, nums, S):
        def memoization(i, S):
            if i == 0:
                if S ==0:
                    return 1
                else:
                    return 0

            elif i in dic and S in dic[i]:
                return dic[i][S]

            res = memoization(i-1, S-nums[i-1]) + memoization(i-1, S+nums[i-1])

            if i in dic:
                dic[i][S] = res
            else:
                dic[i] = {S:res}
            return res

        dic = {}

        return memoization(len(nums), S)

    def lengthOfLongestSubstring(self, s: str) -> int:
        max_count = 0
        dic = {}
        i = 0
        for j in range(len(s)):
            if s[j] in dic:
                i = dic[s[j]] + 1

            max_count = max(max_count, j - i + 1)
            dic[s[j]] = j

        return max_count

    def add_to_dq(self, dq, nums, i):
        while len(dq) and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        return
    #1,3,-1,-3,2,3,6,7
    #
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if k == 0:
            return []
        dq = deque()
        for i in range(k):
            self.add_to_dq(dq, nums, i)
        result, start, end = [], 0, k-1
        while end < len(nums):
            while True:
                if dq[0] >= start:
                    result.append(nums[dq[0]])
                    break
                else:
                    dq.popleft()
            start, end = start+1,end+1
            if end < len(nums):
                self.add_to_dq(dq, nums, end)
        return result

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # sliding window
        # when sum of it >= target , count len(nums)
        # remove first element  check sum
        min_length = len(nums)

        i = 0
        array = []
        while i < len(nums):
            while sum(array) >= target:
                min_length = min(min_length, len(array))
                array.pop(0)

            array.append(nums[i])
            i += 1

        return min_length

    def longestRepeatingSubstring(self, s: str) -> int:
        # 如果有長度k的子字串重複
        # 代表 k-1 長度的一定也是重複
        # 由於不知道從哪個長度開始找 所以使用 binary search 比較快 並尋找對大值
        i = 0
        j = len(s) - 1
        ans = 0
        while i <= j:
            mid = (i + j) // 2

            repeat = self.findRepeating(mid, s)
            if repeat:
                ans = max(ans, mid)
                i = mid + 1
            else:
                j = mid - 1

        return ans


    def findRepeating(self, length, s):
        _set = set()

        for i in range(len(s)):
            if i + length > len(s):
                break
            temp = ""
            for j in range(length):
                temp += s[i + j]

            if temp in _set:
                return True
            else:
                _set.add(temp)

        return False

    def longestDupSubstring(self, s: str) -> str:
        # for if there is any dupulicate string length k
        # then length k-1 must dupulicate too,
        # but we do not konw the langth of it, by using binary search we can search it more quickly
        # then len(s) len(s) - 1 .....
        left = 0
        right = len(s) -1
        ans = ""
        while left <= right:
            mid = (left + right) // 2
            b, substring = self.find_dupulicate(mid, s)
            if b:
                if len(ans) < len(substring):
                    ans = substring
                left = mid + 1
            else:
                right = mid - 1

        return ans

    def find_dupulicate(self, length, s):
        _set = set()
        for i in range(len(s)):
            temp = ""
            if i + length > len(s):
                break
            for j in range(length):
                temp += s[i + j]

            if temp in _set:
                return True, temp
            else:
                _set.add(temp)

        return False, ""

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # binary search
        count = 0
        n1 = len(nums1)
        for i in range(n1):
            temp = []
            for j in range(i, n1):
                temp.append(nums1[j])
                if temp & nums2:
                    count += 1
                else:
                    break
        return count


    def threeSum(self, nums: List[int]) -> List[List[int]]:
        two_set = []
        for i in range(len(nums)):
            temp = set()
            temp.add(nums[i])
            for j in range(i + 1, len(nums)):
                temp.add(nums[j])
                if temp not in two_set:
                    two_set.append((list(temp),  0 - sum(temp)))
                temp.remove(nums[j])
        ans = []
        for e, d in two_set:
            if d in nums:
                if d not in e:
                    e.append(d)
                    if e not in ans:
                        ans.append(e)

        return ans

    def minWindow(self, s: str, t: str) -> str:
        compare = {}
        ans = float('inf'), None, None
        base_dic = Counter(t)
        formed = 0
        required = len(base_dic)

        left = 0
        right = 0
        while right < len(s):
            r = s[right]
            compare[r] = compare.get(r, 0) + 1

            if r in base_dic and compare[r] == base_dic[r]:
                formed += 1

            if formed == required:
                t = right - left + 1
                if ans[0] > t:
                    ans = t, left, right

            while formed == required:
                l = s[left]
                compare[l] -= 1
                left += 1
                if l in base_dic and compare[l] < base_dic[l]:
                    formed -= 1
                else:
                    t = right - left + 1
                    if ans[0] > t:
                        ans = t, left, right

            right += 1

        return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]


    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        #dp[i][j] = nums1[0....i] and nums2[0...j] same subarray count
        # if nums[i] == nums[j]: dp[i][j] =  d[i-1][j-1] + 1
        dp = [[ 0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]
        ans = 0
        for i in range(1,len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = 0

        return max(max(row) for row in dp)


    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        #sliding window
        #condition:check same character length, record, then contiune character length, using max to record but remember the last
        left = 0
        right = 0
        ans = 0
        n = len(s)
        hashmap = defaultdict()
        while right < n:
            r = s[right]
            hashmap[r] = right
            right += 1
            if len(hashmap) == 3:
                del_index = min(hashmap.values())
                del hashmap[s[del_index]]
                left = del_index + 1
            ans = max(ans, right - left)

        return ans

    def findAnagrams(self, s: str, p: str) -> List[int]:
        ans = []
        if len(p) > len(s):
            return ans

        left = 0
        right = len(p) - 1
        p1 = [0] * 26   #要比較兩個字串是否相同 可以使用 array
        s1 = [0] *26    # 利用 ord('') 可以找到acsii index ord(p[i]) - ord("a")

        for i in range(len(p)):
            t = ord(p[i]) - ord("a")
            p1[t] += 1

        for i in range(len(p)):
            t = ord(s[i]) - ord("a")
            s1[t] += 1


        while right <= len(s):
            if p1 == s1:
                ans.append(left)

            if left < len(s):
                t = ord(s[left]) - ord("a")
                s1[t] -= 1

            left += 1
            right += 1

            if right < len(s):
                t = ord(s[right]) - ord("a")
                s1[t] += 1

        return ans

    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1_count = [0] * 26
        s2_count = [0] * 26

        for i in range(len(s1)):
            t = ord(s1[i]) - ord('a')
            s1_count[t] += 1

        left = 0
        right = 0

        while right < len(s2):
            r = s2[right]
            t = ord(r) - ord('a')
            s2_count[t] += 1
            right += 1
            if right >= len(s1):
                if s1_count == s2_count:
                    return True
                else:
                    l = s2[left]
                    t = ord(l) - ord('a')
                    s2_count[t] -= 1
                    left += 1

        return False

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        #using set 
        # sliding window
        # using left, right
        # 將問題改成最多擁有Ｋ的不同元素的子序列
        # f(k) = f(k) - f(k-1)
        a = self.helper(nums, k)
        b = self.helper(nums, k - 1)
        return a - b

    # 將問題改成最多擁有Ｋ的不同元素的子序列
    def helper(self, nums, k) -> int:
        ans = 0
        left = 0
        right = 0
        dic = defaultdict(int)
        while right < len(nums):
            r = nums[right]
            dic[r] += 1

            while len(dic) > k:
                l = nums[left]
                dic[l] -= 1
                if dic[l] == 0:
                    del dic[l]
                left += 1

            #others len(dic) <= k
            ans += right - left + 1
            right += 1

        return ans


    def isLongPressedName(self, name: str, typed: str) -> bool:
        # same order , each character in typed is >= name
        i = 0
        j = 0
        # alex , aaleex
        while i < len(name) and j < len(typed):
            if name[i] == typed[j]:
                j += 1
                continue
            i += 1

        if i == len(name) - 1 and j == len(typed) - 1:
            return False

        return True


    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
        queue=deque([[0,1]])
        level=0
        visited=set()
        visited.add((0,1))

        furthest = max(x, max(forbidden)) + a + b

        while queue:
            for _ in range(len(queue)):
                index,dir=queue.popleft()

                if index==x:
                    return level

                if dir==1:
                    if index + a <=furthest and (index+a,1) not in visited and index + a not in forbidden:
                        visited.add((index+a,1))
                        queue.append([index+a,1])

                    if index-b>0 and (index-b,-1) not in visited and index - b not in forbidden:
                        visited.add((index-b,-1))
                        queue.append([index-b,-1])

                elif dir==-1:
                    if index+a<=furthest and (index+a,1) not in visited and index + a not in forbidden:
                        visited.add((index+a,1))
                        queue.append([index+a,1])

            level+=1

        return -1

    def trap(self, height: List[int]) -> int:
        #  n < 2*10^4 need O(N)
        # i , j   min([0~i], [i+1~n]) - height[i]

        c = 0
        n = len(height)

        left = [0] * n
        right = [0] * n
        left[0] = height[0]

        right[n-1] = height[n-1]

        for i in range(1, n):
            left[i] = max(left[i-1], height[i])

        for i in range(n-2, -1, -1):
            right[i] = max(right[i+1], height[i])

        i = 0
        while i < len(height):
            if height[i] < left[i] and height[i] < right[i]:
                c += min(left[i], right[i]) - height[i]
            i += 1

        return c


    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i = 0
        j = 0 
        ans = []
        while i < len(A) and j < len(B):
            s1 = A[i][0]
            e1 = A[i][1]
            
            s2 = B[j][0]
            e2 = B[j][1]
            
            s = max(s1, s2) # start find max
            e = min(e1, e2) # end find min
            
            if s <= e: # like [5, 5]
                ans.append([s, e])

            if e1 < e2:
                i += 1
            else:
                j += 1

        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        n = len(nums)
        nums.sort()

        for i in range(n):
            self.twoSum(nums, i, ans)

        return ans
 

    def twoSum(self, nums, i, ans):
        right = len(nums) - 1
        left = i + 1
        pivot = nums[i]

        while left < right:
            s = pivot + nums[left] + nums[right]
            if s == 0:
                if [pivot, nums[left], nums[right]] not in ans:
                    ans.append([pivot, nums[left], nums[right]])
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            elif s > 0:
                right -= 1

        return ans


    def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) -1
        result = [0] * len(nums)

        while left < right:
            if abs(nums[left]) < abs(nums[right]):
                s = nums[right]
                result[right] = s * s
                right -= 1
            else:
                s = nums[left]
                result[right] = s * s
                left += 1

        return result


    def searchInsert(self, nums: List[int], target: int) -> int:
        #binary search
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid

        return left

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]

        left = 0
        right = len(nums) - 1
        start = -1
        # left
        while left <= right:
            mid = (left + right) // 2

            if nums[mid] >= target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1

        start =  left
        left = 0
        right = len(nums) - 1
        end = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] <= target:
                left = mid + 1

        end = right

        if nums[end] != target:
            return [-1, -1]


        return [start, end]

    def search(self, nums: List[int], target: int) -> int:
        pivot = self.helper(nums, 0, len(nums) - 1)


        if target <= nums[-1]:
            ans = self.b_search(nums[pivot:], target)
            if ans == -1:
                return -1
            return pivot + ans
        else:
            return self.b_search(nums[:pivot], target)


    def helper(self, nums, left, right):
        if nums[left] < nums[right]:
            return 0

        while left <= right:
            mid = (left + right) // 2
            if nums[mid + 1] < nums[mid]:
                return mid
            else:
                if nums[mid] < nums[left]:
                    right = mid -1
                else:
                    left = mid + 1

    def b_search(self, nums, target):
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid

        return -1

    def search(self, nums: List[int], target: int) -> bool:
        if len(nums) == 1:
            if  nums[0] == target:
                return True
            else:
                return False

        pivot = self.findPivot(nums, 0, len(nums)-1)

        if self.binerySearch(nums[pivot:], target):
            return True

        return self.binerySearch(nums[:pivot], target)

    def findPivot(self, nums, left, right):
        if nums[left] < nums[right]:
            return 0

        while left <= right:
            mid = (left + right) // 2
            if nums[mid + 1] < nums[mid]:
                return mid
            else:
                if nums[mid] < nums[left]:
                    right = mid -1
                else:
                    left = mid + 1

    def helper(self, nums):
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid + 1] < nums[mid]:
                return mid + 1
            else:
                if nums[mid] >= nums[left]:
                    left = mid + 1
                else:
                    right = mid -1
        return -1


    def binerySearch(self, nums, target):
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return True
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1

        return False

    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        #return peak!
        #binary search find max
        left = 0
        right = len(arr) - 1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid + 1] > arr[mid]:
                left = mid + 1
            elif arr[mid + 1] < arr[mid]:
                right = mid -1

        return left


    def findPeakElement(self, arr: List[int]) -> int:
        left = 0
        right = len(arr) - 1
        if len(arr) == 1:
            return 0

        if len(arr) == 2:
            if arr[0] > arr[1]:
                return 0
            else:
                return 1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid + 1] > arr[mid]:
                left = mid + 1
            elif arr[mid + 1] < arr[mid]:
                right = mid -1

        return left

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # k * h >= sum(piles)
        # assume speed = k, then check need hours
        left = 1
        right = max(piles)

        while left <= right:
            times = 0
            mid = (left + right) // 2

            for i in range(len(piles)):
                times += math.ceil(piles[i] / mid)

            if times > h: # too slow => mid up
                left = mid + 1
            elif times < h: # too quick => mid down
                right = mid - 1
            elif times == h:
                right = mid - 1

        return left


    def shipWithinDays(self, weights: List[int], days: int) -> int:
        # return weight
        left = max(weights)
        right = sum(weights)

        ans = 0
        while left <= right:
            mid = (left + right) // 2

            temp_days = self.countDays(weights, mid)

            if temp_days > days: # too slow
                left = mid + 1
            elif temp_days < days: # too quick
                right = mid - 1
            elif temp_days == days:
                right = mid - 1
                ans = mid

        return ans

    def countDays(self, weights, weight):
        days = 0
        carr = 0
        i = 0

        while i < len(weights):
            if carr + weights[i] <= weight:
                carr += weights[i]
                i += 1
            else:
                days += 1
                carr = 0

        if carr > 0:
            days += 1

        return days


    def kmp(self, s, p):
        n = len(s)
        m = len(p)

        failure_function = [0] * m

        for i in range(1, m):
            j = failure_function[i - 1]

            while p[i] != p[j]:
                if j == 0:
                    break
                j = failure_function[j - 1]

            if p[i] == p[j]:
                p[i] = j + 1
            else:
                p[i] = 0

        i = 0
        j = 0
        while i < n:
            if s[i] == p[j]:
                i += 1
                j += 1
            else:
                if j == 0:
                    i += 1
                    continue
                # Lookup the longest proper suffix and prefix before current character
                j = failure_function[j-1]

            if j == m:
                return True
        return False

    def findKthNumber(self, m: int, n: int, k: int) -> int:
        # m*n
        r = []
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                heapq.heappush(r, i*j)
        w = heapq.nsmallest(k, r)

        return w[k-1]


    def letterCombinations(self, digits: str) -> List[str]:
        self.dic = {
            '2':["a","b","c"],
            '3':["d","e","f"],
            '4':["g","h","i"],
            '5':["j","k","l"],
            '6':["m","n","o"],
            '7':["p","q","r","s"],
            '8':["t","u","v"],
            '9':["w","x","y","z"]
        }
        self.result = []
        self.helper(digits, 0, "")

        return self.result

    def helper(self, digits, index, temp):
        if len(temp) == len(digits):
            self.result.append(temp)
            return


        curr = digits[index]
        for latter in self.dic[curr]:
            self.helper(digits, index + 1, temp + latter)

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.result = []
        self.helper(candidates, target, [], 0)
        return self.result
    
    def helper(self, candidates, target, temp, index):
        s = sum(temp)
        if s == target:
            self.result.append(list(temp))
        
        if s > target:
            return
            
        for i in range(index, len(candidates)):
            self.helper(candidates, target, temp + candidates[i], i)

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # only once !
        self.result = []
        candidates.sort()
        self.helper(candidates, target, [], 0)
        return self.result

    def helper(self, candidates, target, temp, index):
        if sum(temp) == target:
            self.result.append(temp)
            return

        if sum(temp) > target:
            return

        for i in range(index, len(candidates)):
            if i > index and candidates[i] == candidates[i - 1]:
                    continue
            self.helper(candidates, target, temp + [candidates[i]], i + 1)

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()  #要判斷是否重複，先利用排序

        def helper(nums, temp, index):
            result.append(temp)

            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i - 1]:
                    #因為已經排序過了，所以先相同的數字會排再一起
                    #當i>index 且 nums[i] == nums[i -1] 代表上一次一定已經循環過了
                    continue
                helper(nums, temp + [nums[i]], i + 1)

        helper(nums, [], 0)
        return result


    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        r1 = self.reverse(l1)
        r2 = self.reverse(l2)

        # add
        added = self.add(r1, r2)

        # reverse
        ans = self.reverse(added)

        return ans

    def add(self, l1, l2):
        result = curr = ListNode(-1)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next

            curr.next = ListNode(carry%10)
            curr = curr.next
            carry = carry // 10

        return result.next


    def reverse(self, head):
        result = None

        while head:
            temp = head.next
            head.next = result
            result = head
            head = temp

        return result

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        result = []
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                dic[nums[i]] += 1
            else:
                dic[nums[i]] = 1

        def helper(nums, temp):
            if len(temp) == len(nums):
                result.append(temp)
                return

            for i, v in dic.items():
                if dic[i] > 0:
                    dic[i] -= 1
                    helper(nums, temp + [i])
                    dic[i] += 1

        helper(nums, [])
        return list(result)


    def letterCasePermutation(self, S: str):
        self.result = []
        self.visited = []

        self.helper(S, [], 0)

        return self.result


    def helper(self, S: str, temp, index):
        if len(S) == len(temp):
            t = ''.join(temp)
            if hash(t) not in self.visited:
                self.result.append(t)
                self.visited.append(t)
            return

        if not S[index].isalpha():
            self.helper(S, temp + [S[index]], index + 1)
        else:
            self.helper(S, temp + [S[index].lower()], index + 1)
            self.helper(S, temp + [S[index].upper()], index + 1)

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        result = []
        i = 0
        j = 0
        while i < m and j < n:
            if nums1[i] <= nums2[j]:
                result.append(nums1[i])
                i += 1
            else:
                result.append(nums2[j])
                j += 1

        if i < m:
            result.extend(nums1[i:])

        if j < n:
            result.extend(nums2[j:])

        for i in range(m + n):
            nums1[i] = result[i]

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        s = set()

        while headB:
            s.add(headB)
            headB = headB.next

        while headA:
            if headA in s:
                return headA
            headA = headA.next


    def canPartitionKSubsets(self, arr: List[int], k: int) -> bool:
        total_sum = sum(arr)
        if total_sum % k != 0:
            return False

        target_sum = total_sum // k

        taken = [0] * len(arr)

        memo = {}
        # combination
        def helper(index, curr_sum, count, taken):
            if count == k:
                return True

            if curr_sum > target_sum:
                return False

            taken_str = ''.join(taken)

            if taken_str in memo:
                return memo[taken_str]

            if curr_sum == target_sum:
                memo[taken_str] = helper(0, 0, count+1) # 當找到一個組合後 將curr_sum 歸0, index也從0開始找
                return memo[taken_str]
            # 當找到一個組合後，根據目前找到的組合繼續往下找，當找完整個序列後，紀錄找尋結果

            for i in range(index, len(arr)):
                if not taken[i]:
                    taken[i] = True
                    if helper(i+1, curr_sum + arr[i], count): # 此處的意思是 當有true回來代表已經找到一種方式 全部分配完成 所以直接return true
                        return True
                    taken[i] = False # 如果是false 代表現在的組合方式行不同 所以移除最後一個拿個數字 並且繼續往下走
                memo[taken_str] = False
                return memo[taken_str]

        return helper(0, 0, 0)



    def restoreIpAddresses(self, s: str) -> List[str]:
        # combination of s into 4 part
        # return all possible ip addresses form
        # no leading zero
        # not allow reorder
        # 0 ~ 256
        # record which index is taken
        n = len(s)
        result = []

        def helper(start, temp_subset):
            if start == n and len(temp_subset) == 4:
                result.append('.'.join(temp_subset))
                return

            for i in range(start, min(start + 3, n)):
                if s[start] == '0' and i > start:
                    continue
                if 0 <= int(s[start:i+1]) <= 255:
                    helper(i + 1, temp_subset + [s[start:i+1]])

        helper(0, [])
        return result

    def partition(self, s: str) -> List[List[str]]:
        result = []
        self.dfs(s, 0, [], result)
        return result

    def dfs(self, s, start, temp, result):
        if start == len(s):
            result.append(temp)
            return

        for i in range(start, len(s)):
            t = s[start: i + 1]
            if t == t[::-1]:
                self.dfs(s, i + 1, temp + [t], result)


    def splitIntoFibonacci(self, num: str) -> List[int]:
        result = []
        self.helper(num, 0, result)
        return result

    def helper(self, num, start, result):
        if start == len(num) and self.isValid(result):
            return True

        if not self.isValid(result):
            return False

        for i in range(start, len(num)):
            if num[start] == 0 and i > start:
                continue

            #pick how many number
            t = int(''.join(num[start: i + 1]))
            if len(result) < 2:
                result.append(t)

            if self.helper(num, i + 1, result):
                return True

            result.remove(t)

    def isValid(self, result):
        if len(result) <= 2:
            return True

        if result[-1] == result[-2] + result[-3]:
            return True

        return False

a = Solution()
x = TreeNode(val=1, left=TreeNode(val=2),right=TreeNode(3))
x1 = TreeNode(val=1, left=TreeNode(val=2),right=TreeNode(3))

x3 = TreeNode(val=3,left=TreeNode(5,left=TreeNode(6),right=TreeNode(2,left=TreeNode(7),right=TreeNode(4))),right=TreeNode(1,left=TreeNode(9),right=TreeNode(8)))

x4 = TreeNode(val=1,left=TreeNode(2,left=TreeNode(2)),right=TreeNode(3,left=TreeNode(2),right=TreeNode(4)))

x5 = TreeNode(val=1,left=TreeNode(2,right=TreeNode(5),left=TreeNode(4)),right=TreeNode(3))

l2 = ListNode(2, ListNode(4, ListNode(3)))

l1 = ListNode(7, l2)
trace = "jeff_test_" + str(uuid.uuid4())
t = bytes(trace.encode())
print(t)
w = a.splitIntoFibonacci("1101111")
print(w)

