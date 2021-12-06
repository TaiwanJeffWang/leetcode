
# combination => 列出所有順序, ex. 123 == 132 c m取Ｎ or first == "[" or first == "{"
# 所以要設定start index , 不可以從0開始
# def dfs(self  , nums , start_index , temp: list):
#     if len(temp) = len(nums):
#         return

#     for i in range(start_index,len(nums)): ### key word!!!!! the start index!!!!
#         temp.append(nums[i])
#         self.dfs(nums,nums,i+1,temp) ### !!!!!!key word here , i+1
#         temp.pop() ### !!!!!! key here remove the last element



# permutation => 列出所有組合方式, ex. 123 != 132  p m取Ｎ
# def dfs(self  , nums , start_index , temp: list):
#     if len(temp) = len(nums):
#         return

#     for i in range(len(nums)): ### key word!!!!! the start index!!!!
#         if nums(i) in temp:
#           continue            ### keyword!!!!! 因為要把每個元素跑一輪 所以要從頭跑 但是已經在裡面的元素則略過
#         temp.append(nums[i])
#         self.dfs(nums,nums,temp) 
#         temp.pop() ### !!!!!! key here remove the last element