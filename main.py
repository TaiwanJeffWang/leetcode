from os import close
from typing import Dict, List, Optional
import math
from collections import deque
import heapq

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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
    def pathSum_3(self, root: Optional[TreeNode], targetSum: int) -> int:
        # we can use a dfs  when we find from the start to the leaf , we can check it by prefix sum.
        mp = { 0:1 }
        prefix = 0
        count = []
        self.getPathSum_3(root, targetSum, prefix, mp, count)

        return sum(count)

    def getPathSum_3(self, root: Optional[TreeNode], targetSum: int, prefix: int, mp: dict , result: List):
        if root is None:
            return 0

        prefix = root.val + prefix
        diff =  prefix - targetSum

        if mp.get(diff):
            result.append(mp[diff])

        if mp.get(prefix):
             mp[prefix] = mp[prefix] +1
        else:
            mp[prefix] = 1

        self.getPathSum_3(root.left, targetSum, prefix, mp, result)
        self.getPathSum_3(root.right, targetSum, prefix ,mp , result)
        mp[prefix] = mp[prefix] -1
        prefix -= root.val

    #2. Add Two Numbers
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        queue_1, queue_2 = deque(), deque()
        curr_1, curr_2 = l1, l2

        while curr_1:
            queue_1.append(curr_1)
            curr_1 = curr_1.next
        while curr_2:
            queue_2.append(curr_2)
            curr_2 = curr_2.next

        result = ListNode(-1)
        curr = result
        while queue_1 or queue_2:
            total = 0
            if queue_1 and queue_2:
                item_1 = queue_1.popleft()
                item_2 = queue_2.popleft()
                total = item_1.val + item_2.val
            elif queue_1:
                item_1 = queue_1.popleft()
                total = item_1.val
            elif queue_2:
                item_2 = queue_2.popleft()
                total = item_2.val

            if total >= 10:
                r = total % 10
                n = total / 10
                temp = ListNode(r)
                if queue_1:
                    last = queue_1[0]
                    last.val += int(n)
                    queue_1[0] = last
                else:
                    queue_1.append(TreeNode(int(n)))
            else:
                temp = ListNode(total)

            curr.next = temp
            curr = curr.next
        return result.next

    #14. Longest Common Prefix
    def longestCommonPrefix(self, strs: List[str]) -> str:
        x = zip(*strs)
        ref = strs[0]
        result = ""

        for i in range(0,len(ref)):
            for j in range(1,len(strs)):
                if len(strs[j])-1 < i or strs[j][i] != ref[i]:
                    return result
            result += ref[i]

        return result

    #19. Remove Nth Node From End of List
    def removeNthFromEnd_1(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        queue = deque()
        temp = head
        count = 0
        while temp:
            queue.append(temp)
            temp = temp.next
            count += 1
        curr = 0
        k = ListNode(-1)
        running = k

        while queue:
            item = queue.popleft()
            if curr + n == count:
                pass
            else:
                running.next = item
                running.next.next = None
                running = running.next
            curr+=1
        return k.next

    #3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        count = 0
        max_count = 0
        temp = []
        i = 0
        dic = {}
        #bbtablud
        while i < len(s):
            if s[i]  in temp:
                temp.append(s[i])
                start_index = dic[s[i]]
                dic[s[i]] = i
                count = i - start_index
            else:
                temp.append(s[i])
                dic[s[i]]=i
                count+=1

            i+=1
            if count > max_count:
                max_count = count

        return max_count

    #4. Median of Two Sorted Arrays
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)
        result = []
        mid = m+n
        x = mid // 2
        i , j = 0 , 0
        while i < m and j < n:
            if nums1[i] >= nums2[j]:
                result.append(nums2[j])
                j+=1
            elif nums1[i] < nums2[j]:
                result.append(nums1[i])
                i+=1

            if len(result) >= x+1:
                if mid % 2 == 0:
                    return (result[-1] + result[-2])/2
                else:
                    return result[-1]

        if i == m:
            for item in range(j,n):
                result.append(nums2[item])
                if len(result) >= x+1:
                    if mid % 2 == 0:
                        return (result[-1] + result[-2])/2
                    else:
                        return result[-1]
        if j == n:
            for item in range(i,m):
                result.append(nums1[item])
                if len(result) >= x+1:
                    if mid % 2 == 0:
                        return (result[-1] + result[-2])/2
                    else:
                        return result[-1]


    count = 0
    def goodNodes(self, root: TreeNode) -> int:

        self.dfs(root,root.val)
        return self.count

    def dfs(self, root: TreeNode, max_number: int):
        if root is None:
            return

        if root.val >= max_number:
            max_number = root.val
            self.count +=1

        self.dfs(root.left,max_number)
        self.dfs(root.right,max_number)

    def arraySign(self, nums: List[int]) -> int:
        result = nums[0]
        for i in range(1,len(nums)):
            result = result * nums[i]
            if result == 0:
                break

        if result < 0:
            return -1
        if result > 0:
            return 1

    #9. Palindrome Number
    def isPalindrome(self, x: int) -> bool:
        s = str(x)
        leng = len(s)
        i = 0
        j = leng - 1

        while i <= j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return False

        return True


    #23. Merge k Sorted Lists
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # bad method , 更糟糕的暴力解法 因為重複比較元素 不如先把 每一個元素拆開 sort 在連接再一起
        # k = 0
        # result = ListNode(-1)
        # curr = result
        # while lists:
        #     for i in range(len(lists)):
        #         item = lists[i]
        #         if item and lists[k] and item.val < lists[k].val:
        #             k = i

        #     if lists[k]:
        #         curr.next = ListNode(lists[k].val)
        #         curr = curr.next
        #         lists[k] = lists[k].next
        #     if  not lists[k]:
        #         del lists[k]

        #     k = 0

        # return result.next
        result = ListNode(-1)
        curr = result
        heap = []
        for i in range(len(lists)):
            if lists[i]:
                heap.append([lists[i].val,i])

        heapq.heapify(heap)


        while heap:
            item = heapq.heappop(heap)
            val = item[0]
            index = item[1]
            curr.next = ListNode(val)
            curr = curr.next
            lists[index] = lists[index].next
            if lists[index]:
                heap.append([lists[index].val, index])
            heapq.heappush(heap)

        return result.next

    def maxProduct(self, nums: List[int]) -> int:
        heap = []
        for i in range(len(nums)):
            heap.append([nums[i], i])

        heapq._heapify_max(heap)

        f = heapq._heappop_max(heap)
        s = heapq._heappop_max(heap)

        result = (nums[f[1]]-1) * (nums[s[1]]-1)

        return result

    #253. Meeting Rooms II [[9,10],[4,9],[4,17]] => [4,9],[4,17],[9,10]
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        heap = []
        count = 0
        for i in intervals:
            if i:
                heap.append(i)

        heapq.heapify(heap)
        # i =heapq.heappop(heap)
        end_times = []

        while heap:
            meeting = heapq.heappop(heap)
            start_time = meeting[0]
            end_time = meeting[1]
            end_times.append(end_time)

            end_times.sort()

            if start_time < end_times[0]:
                count += 1
            else:
                del end_times[0]


        return count

    #1338. Reduce Array Size to The Half
    def minSetSize(self, arr: List[int]) -> int:
        dic = {}
        leng = len(arr)
        for ar in arr:
            if dic.get(ar):
                dic[ar] = dic[ar] + 1
            else:
                dic[ar] = 1

            if dic[ar] >= leng/2:
                return ar

        sort = sorted(dic.items(), key= lambda x: x[1],reverse= True)
        result = []
        count = 0
        values = 0
        for item in sort:
            temp = item[1]
            # if temp > leng/2:
            #     return temp
            # else:
            result.append(item[0])
            count += temp
            values +=1
            if count >= leng/2:
                return values

        return values

    #451. Sort Characters By Frequency
    def frequencySort(self, s: str) -> str:
        dic = {}
        result = ""

        for l in s:
            if dic.get(l):
                dic[l] = dic[l] + 1
            else:
                dic[l] = 1

        sort = sorted(dic.items(), key= lambda x:x[1], reverse=True)

        for item in sort:
            result += item[0] * item[1]

        return result

    #56. Merge Intervals
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        result = []
        intervals.sort()
        temp = []
        for item in intervals:
            if temp:
                if temp[1] >= item[0]:
                    if temp[1] < item[1]:
                        temp[1] = item[1]
                else:
                    result.append(list(temp))
                    temp = [item[0], item[1]]
            else:
                temp = [item[0], item[1]]

        result.append(temp)

        return result


tree1 = TreeNode(1, left=TreeNode(3, left=TreeNode(5)), right=TreeNode(2, left=TreeNode(7, left=TreeNode(8)), right=TreeNode(9)))
tree2 = TreeNode(2, left=TreeNode(1, right=TreeNode(4)), right=TreeNode(3, right=TreeNode(7)))
bst =TreeNode(8, left=TreeNode(val=3,left=TreeNode(1),right=TreeNode(6,left=TreeNode(4),right=TreeNode(7)))
,right=TreeNode(10, right=TreeNode(14,left=TreeNode(val=13))))
tree3 = TreeNode(1, left=TreeNode(2, right=TreeNode(6),left=TreeNode(5)), right=TreeNode(3))
tree4 = TreeNode(val=5)
sub1 = TreeNode(val=4,left=TreeNode(val=11,left=TreeNode(val=7),right=TreeNode(val=2)))
sub2 = TreeNode(val=8,left=TreeNode(val=13),right=TreeNode(val=4,left=TreeNode(val=5),right=TreeNode(val=1)))
tree4.left=sub1
tree4.right=sub2

tree5 = TreeNode(val=1,left=TreeNode(val=-2,right=TreeNode(3),left=TreeNode(val=1,left=TreeNode(-1))),right=TreeNode(val=-3,left=TreeNode(val=-2)))

listNode = ListNode(3)
a = ListNode(2)
b = ListNode(0)
c = ListNode(-4)

listNode.next = a
a.next = b
b.next = c
c.next =a

# l1 = ListNode(9)
# l1.next = ListNode(9)
# l1.next.next = ListNode(9)
# l1.next.next.next = ListNode(9)
# l1.next.next.next.next = ListNode(9)
# l1.next.next.next.next.next = ListNode(9)
# l1.next.next.next.next.next.next = ListNode(9)


l2 = ListNode(1)
l2.next = ListNode(2)
# l2.next.next = ListNode(3)
# l2.next.next.next = ListNode(4)
# l2.next.next.next.next = ListNode(5)

goodNodes = TreeNode(val=3, left= TreeNode(val=1, left= TreeNode(3)), right= TreeNode(4, left= TreeNode(9), right= TreeNode(5)))

grid=[[1,3,1],[1,5,1],[4,2,1]]
orangesRottings = [[2,1,1],[1,1,0],[0,1,1]]

l1 = ListNode(1)
l1.next = ListNode(2)
l1.next.next = ListNode(2)

l2 = ListNode(1)
l2.next = ListNode(3)
l2.next.next = ListNode(5)


l3 = ListNode(2)
l3.next = ListNode(3)
l3.next.next = ListNode(4)

a = Solution()
w = a.merge([[1,4],[4,5]])
print(w)

