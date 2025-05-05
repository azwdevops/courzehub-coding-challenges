import re
import heapq
from sys import maxsize
from collections import defaultdict
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 227. Basic Calculator II
# https://leetcode.com/problems/basic-calculator-ii/description/


class Solution:
    def calculate(self, s: str) -> int:
        # we add '+' to ensure if stack still remains after for loop the stack items ate emptied and the operations done
        s = s.replace(" ", "") + "+"
        if s == "":
            return 0
        stack = []
        current = 0
        sign = "+"

        for char in s:
            if char.isdigit():
                current = current * 10 + int(char)
            else:
                if sign == "+":
                    stack.append(current)
                elif sign == "-":
                    stack.append(-current)
                elif sign == "*":
                    stack.append(stack.pop() * current)
                elif sign == "/":
                    stack.append(int(stack.pop() / current))
                sign = char
                current = 0

        result = sum(stack)
        return result


# 229. Majority Element II
# https://leetcode.com/problems/majority-element-ii/description/
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        count1 = 0
        count2 = 0
        el1 = float("-inf")
        el2 = float("-inf")

        for i in range(len(nums)):
            if count1 == 0 and el2 != nums[i]:
                count1 = 1
                el1 = nums[i]
            elif count2 == 0 and el1 != nums[i]:
                count2 = 1
                el2 = nums[i]
            elif nums[i] == el1:
                count1 += 1
            elif nums[i] == el2:
                count2 += 1
            else:
                count1 -= 1
                count2 -= 1
        count1 = 0
        count2 = 0

        for i in range(len(nums)):
            if el1 == nums[i]:
                count1 += 1
            elif el2 == nums[i]:
                count2 += 1

        times = len(nums) // 3

        result = []
        if count1 > times:
            result.append(el1)
        if count2 > times:
            result.append(el2)

        return result


# 230. Kth Smallest Element in a BST
# https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        result = None
        count = 0

        def recursive(node):
            if not node:
                return
            recursive(node.left)
            nonlocal count
            count += 1
            if count == k:
                nonlocal result
                result = node.val
                return
            recursive(node.right)

        recursive(root)

        return result


# 235. Lowest Common Ancestor of a Binary Search Tree
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/


class Solution:
    def lowestCommonAncestor(self, root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


# 236. Lowest Common Ancestor of a Binary Tree
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/


class Solution:
    def lowestCommonAncestor(self, root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
        def recursive(node):
            if not node:
                return None

            if node == p or node == q:
                return node

            left = recursive(node.left)
            right = recursive(node.right)

            if left and right:
                return node

            return left if left else right

        return recursive(root)


# 237. Delete Node in a Linked List
# https://leetcode.com/problems/delete-node-in-a-linked-list/description/


class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        temp = node.next
        while temp:
            node.val = temp.val
            if not temp.next:
                node.next = None
                break
            node = temp
            temp = temp.next


# 238. Product of Array Except Self
# https://leetcode.com/problems/product-of-array-except-self/description/
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        result = []

        prefix = 1
        for num in nums:
            result.append(prefix)
            prefix *= num

        suffix = 1

        n = len(nums)
        for i in range(n - 1, -1, -1):
            result[i] *= suffix
            suffix *= nums[i]

        return result


# 240. Search a 2D Matrix II
# https://leetcode.com/problems/search-a-2d-matrix-ii/description/


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])

        col = n - 1
        row = 0

        while row <= m - 1 and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1

        return False


# 241. Different Ways to Add Parentheses
# https://leetcode.com/problems/different-ways-to-add-parentheses/description/


class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}

        def backtrack(left, right):
            result = []
            for i in range(left, right + 1):
                item = expression[i]
                if item in operations:
                    nums1 = backtrack(left, i - 1)
                    nums2 = backtrack(i + 1, right)

                    for n1 in nums1:
                        for n2 in nums2:
                            result.append(operations[item](n1, n2))

            if result == []:
                result.append(int(expression[left : right + 1]))

            return result

        return backtrack(0, len(expression) - 1)


# 244. Shortest Word Distance II
# https://leetcode.com/problems/shortest-word-distance-ii/description/
class WordDistance:
    def __init__(self, wordsDict: List[str]):
        # space O(n)
        self.hashmap = defaultdict(list)
        for index, word in enumerate(wordsDict):
            self.hashmap[word].append(index)

    def shortest(self, word1: str, word2: str) -> int:
        index1 = self.hashmap[word1]
        index2 = self.hashmap[word2]

        i = 0
        j = 0
        ans = maxsize

        while i < len(index1) and j < len(index2):
            ans = min(ans, abs(index1[i] - index2[j]))
            if index1[i] < index2[j]:
                i += 1
            else:
                j += 1

        return ans


# 245. Shortest Word Distance III
# https://leetcode.com/problems/shortest-word-distance-iii/description/


class Solution:
    def shortestWordDistance(self, wordDict: List[str], word1: str, word2: str) -> int:
        index1 = index2 = -1
        min_dist = float("inf")

        for i, word in enumerate(wordDict):
            if word == word1 == word2:
                if index1 != -1:
                    min_dist = min(min_dist, i - index1)
                index1 = i
            elif word == word1:
                if index2 != -1:
                    min_dist = min(min_dist, i - index2)
                index1 = i
            if word == word2:
                if index1 != -1:
                    min_dist = min(min_dist, i - index1)
                index1 = i

        return min_dist


# 247. Strobogrammatic Number II
# https://leetcode.com/problems/strobogrammatic-number-ii/description/


class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        strobo_numbers = {"0": "0", "1": "1", "8": "8", "6": "9", "9": "6"}

        if len(num) == 1:
            return num in ["0", "1", "8"]
        else:
            i = 0
            j = len(num) - 1
            while i <= j:
                if num[i] in strobo_numbers and strobo_numbers[num[i]] == num[j]:
                    i += 1
                    j -= 1
                else:
                    return False
            return True


# 249. Group Shifted Strings
# https://leetcode.com/problems/group-shifted-strings/description/
class Solution:
    def groupShiftedString(self, arr):
        lowercase = "abcdefghijklmnopqrstuvwxyz"
        # code here
        shift_dict = defaultdict(list)
        for word in arr:
            n = len(word)
            if n == 1:
                shift_dict["1"].append(word)
                continue
            key = ""
            for i in range(1, n):
                index_diff = lowercase.index(word[i]) - lowercase.index(word[i - 1])
                if index_diff < 0:
                    index_diff += 26
                key += str(index_diff)
            shift_dict[key].append(word)

        return list(shift_dict.values())


# 250. Count Univalue Subtrees
# https://leetcode.com/problems/count-univalue-subtrees/
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        result = 0

        def traverse(node):
            if not node:
                return float("-inf")
            left = traverse(node.left)
            right = traverse(node.right)
            if (left == float("-inf") or left == node.val) and (right == float("-inf") or right == node.val):
                nonlocal result
                result += 1
                return node.val
            return float("inf")

        traverse(root)

        return result


# 253. Meeting Rooms II
# https://leetcode.com/problems/meeting-rooms-ii/description/
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort()

        meeting_rooms = 1
        heap = [intervals[0][1]]

        for start, end in intervals[1:]:
            if heap[0] <= start:
                heapq.heappop(heap)
            heapq.heappush(heap, end)
            meeting_rooms = max(meeting_rooms, len(heap))

        return meeting_rooms


# 254. Factor Combinations
# https://leetcode.com/problems/factor-combinations/description/


def factorCombinations(n):
    # Write your code here.
    result = []

    def recursive(start, product, current):
        if product == n:
            result.append(current[:])
        for i in range(start, n):
            if product * i > n:
                break
            if n % (product * i) == 0:
                current.append(i)
                recursive(i, product * i, current)
                current.pop()

    recursive(2, 1, [])

    return result


# 255. Verify Preorder Sequence in Binary Search Tree
# https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/description/


class Solution:
    def verifyPreorder(self, preorder):
        stack = []
        lastPopped = float("-inf")

        for x in preorder:
            if lastPopped > x:
                return False
            while stack and x > stack[-1]:
                lastPopped = stack.pop()
            stack.append(x)

        return True


# 256. Paint House
# https://www.naukri.com/code360/problems/paint-house_1460385?leftPanelTabValue=SUBMISSION
def minCost(cost):
    # Write your code here.
    n = len(cost)

    dp = [[0] * 3 for _ in range(n)]
    dp[0] = cost[0]

    for i in range(1, n):
        for j in range(3):
            if j == 0:
                dp[i][0] = cost[i][0] + min(dp[i - 1][1], dp[i - 1][2])
            elif j == 1:
                dp[i][1] = cost[i][1] + min(dp[i - 1][0], dp[i - 1][2])
            elif j == 2:
                dp[i][2] = cost[i][2] + min(dp[i - 1][0], dp[i - 1][1])
    return min(dp[-1])


# 259. 3Sum Smaller
def threeSumSmaller(n: int, arr: List[int], target: int) -> int:
    # write your code here
    arr.sort()
    count = 0

    for i in range(n - 2):
        left = i + 1
        right = n - 1

        while left < right:
            if arr[i] + arr[left] + arr[right] < target:
                count += right - left
                left += 1
            else:
                right -= 1

    return count


# 260. Single Number III
# https://leetcode.com/problems/single-number-iii/description/


class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xor = 0

        for num in nums:
            xor ^= num

        diff_bit = xor & -xor

        a = 0
        b = 0

        for num in nums:
            if diff_bit & num:
                a ^= num
            else:
                b ^= num

        return [a, b]


# 261. Graph Valid Tree
# https://www.naukri.com/code360/problems/graph-valid-tree_1376618?leftPanelTabValue=SUBMISSION


def checkgraph(edges, n, m):

    # Write your code here
    # Return a boolean variable 'True' or 'False' denoting the answer
    adj_list = [[] for _ in range(n)]

    for start, end in edges:
        adj_list[start].append(end)
        adj_list[end].append(start)

    visited = set()

    def dfs(node, parent):
        visited.add(node)

        for neighbor in adj_list[node]:
            if neighbor not in visited:
                if not dfs(neighbor, node):
                    return False
            elif neighbor != parent:
                return False

        return True

    if not dfs(0, -1):
        return False

    return len(visited) == n


# numbers in this set 227, 229, 235, 236, 237, 238, 240, 241, 245,247, 250,253, 254, 255, 256, 259, 260, 261
