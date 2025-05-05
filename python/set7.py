from typing import Optional, List
from collections import deque
from functools import cmp_to_key


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False


# 173. Binary Search Tree Iterator
# https://leetcode.com/problems/binary-search-tree-iterator/description/


class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node: Optional[TreeNode]):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        # The next node is the top of the stack
        node = self.stack.pop()
        # If the node has a right child, we push its leftmost children to the stack
        if node.right:
            self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0


# 179. Largest Number
# https://leetcode.com/problems/largest-number/description/
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        nums = [str(item) for item in nums]

        def compare(str1, str2):
            if str1 + str2 > str2 + str1:
                return -1
            else:
                return 1

        nums = sorted(nums, key=cmp_to_key(compare))

        return str(int("".join(nums)))


# 187. Repeated DNA Sequences
# https://leetcode.com/problems/repeated-dna-sequences/description/


class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        n = len(s)
        if n <= 10:
            return []

        left = 0
        right = 10

        seen = set()
        result = []
        while right <= n:
            current = s[left:right]
            if current in seen:
                if current not in result:
                    result.append(current)
            else:
                seen.add(current)
            left += 1
            right += 1

        return result


# 189. Rotate Array
# https://leetcode.com/problems/rotate-array/description/


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n

        count = 0
        while count < k:
            last = nums.pop()
            nums.insert(0, last)
            count += 1


# 198. House Robber
# https://leetcode.com/problems/house-robber/description/


class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return max(nums)

        dp = [0] * n

        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return dp[-1]


# 199. Binary Tree Right Side View
# https://leetcode.com/problems/binary-tree-right-side-view/description/


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        result = []

        temp = deque([root])

        while temp:
            result.append(temp[-1].val)
            next_level = []
            for _ in range(len(temp)):
                node = temp.popleft()
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            temp = deque(next_level)

        return result


# 200. Number of Islands
# https://leetcode.com/problems/number-of-islands/description/


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        m = len(grid)
        n = len(grid[0])

        visited = [[0] * n for _ in range(m)]
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        def bfs(row, col):
            visited[row][col] = 1

            for delta_row, delta_col in directions:
                nrow = row + delta_row
                ncol = col + delta_col

                if 0 <= nrow < m and 0 <= ncol < n and grid[nrow][ncol] == "1" and visited[nrow][ncol] == 0:
                    bfs(nrow, ncol)

        count = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1" and visited[i][j] == 0:
                    count += 1
                    bfs(i, j)

        return count


# 201. Bitwise AND of Numbers Range
# https://leetcode.com/problems/bitwise-and-of-numbers-range/description/


class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shift = 0
        while left < right:
            left >>= 1
            right >>= 1
            shift += 1

        return left << shift


# 204. Count Primes
# https://leetcode.com/problems/count-primes/description/


class Solution:
    def countPrimes(self, n: int) -> int:
        if n <= 2:
            return 0

        primes = [1] * (n + 1)
        primes[0] = 0
        primes[1] = 0

        for i in range(2, n + 1):
            if primes[i] == 1:
                for j in range(2 * i, n + 1, i):
                    if primes[j] == 0:
                        continue
                    primes[j] = 0

        result = [num for num in primes[:-1] if num == 1]

        return len(result)


# 207. Course Schedule
# https://leetcode.com/problems/course-schedule/description/


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if len(prerequisites) == 0:
            return True
        adj_list = [[] for _ in range(numCourses)]

        in_degree = {i: 0 for i in range(numCourses)}
        for [course, prereq] in prerequisites:
            in_degree[course] += 1
            adj_list[prereq].append(course)

        stack = []
        for key, value in in_degree.items():
            if value == 0:
                stack.append(key)

        if len(stack) == 0:
            return False

        while stack:
            prereq = stack.pop()
            for course in adj_list[prereq]:
                in_degree[course] -= 1
                if in_degree[course] == 0:
                    stack.append(course)

        for key, value in in_degree.items():
            if value != 0:
                return False

        return True


# 208. Implement Trie (Prefix Tree)
# https://leetcode.com/problems/implement-trie-prefix-tree/description/


class Trie:

    def __init__(self):
        self.chars = {}

    def insert(self, word: str) -> None:
        current = self.chars
        for i in range(len(word)):
            char = word[i]
            if char not in current:
                current[char] = {"children": {}, "endWord": False}
            current = current[char]["children"]
        current["endWord"] = True

    def search(self, word: str) -> bool:
        current = self.chars

        for char in word:
            if char not in current:
                return False
            current = current[char]["children"]
        return current.get("endWord", False)

    def startsWith(self, prefix: str) -> bool:
        current = self.chars
        for char in prefix:
            if char not in current:
                return False
            current = current[char]["children"]
        return True


# 209. Minimum Size Subarray Sum
# https://leetcode.com/problems/minimum-size-subarray-sum/description/


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0

        min_length = float("inf")
        n = len(nums)
        current = 0
        for right in range(n):
            current += nums[right]
            while current >= target:
                min_length = min(min_length, right - left + 1)
                current -= nums[left]
                left += 1

        return 0 if min_length == float("inf") else min_length


# 210. Course Schedule II
#  https://leetcode.com/problems/course-schedule-ii/description/


class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj_list = [[] for _ in range(numCourses)]
        in_degree = [0 for _ in range(numCourses)]

        for course, prereq in prerequisites:
            adj_list[prereq].append(course)
            in_degree[course] += 1

        stack = []
        for i in range(numCourses):
            if in_degree[i] == 0:
                stack.append(i)
        if len(stack) == 0:
            return []

        result = []

        while stack:
            prereq = stack.pop()
            result.append(prereq)
            for course in adj_list[prereq]:
                in_degree[course] -= 1
                if in_degree[course] == 0:
                    stack.append(course)

        return result if len(result) == numCourses else []


# 211. Design Add and Search Words Data Structure
# https://leetcode.com/problems/design-add-and-search-words-data-structure/description/


class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]

        current.endWord = True

    def search(self, word: str) -> bool:
        def dfs(index, node):
            if index == len(word):
                return node.endWord
            if word[index] == ".":
                for child in node.children.values():
                    if dfs(index + 1, child):
                        return True
                return False
            else:
                if word[index] not in node.children:
                    return False
                return dfs(index + 1, node.children[word[index]])

        return dfs(0, self.root)


# 215. Kth Largest Element in an Array
# https://leetcode.com/problems/kth-largest-element-in-an-array/description/


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        k = len(nums) - k

        def quickSelect(left, right):
            less = left
            equal = left
            greater = right
            pivot = nums[right]
            while equal <= greater:
                while equal <= greater and nums[equal] < pivot:
                    nums[equal], nums[less] = nums[less], nums[equal]
                    less += 1
                    equal += 1
                while equal <= greater and nums[equal] == pivot:
                    equal += 1
                while equal <= greater and nums[equal] > pivot:
                    nums[equal], nums[greater] = nums[greater], nums[equal]
                    greater -= 1

            if k > greater:
                return quickSelect(greater + 1, right)
            elif k < less:
                return quickSelect(left, less - 1)
            else:
                return nums[greater]

        return quickSelect(0, len(nums) - 1)


# 224. Basic Calculator
# https://leetcode.com/problems/basic-calculator/description/
class Solution:
    def calculate(self, s: str) -> int:
        result = 0
        current_num = 0
        sign = 1
        stack = []

        for c in s:
            # if digit
            if c.isdigit():
                current_num = (current_num * 10) + int(c)
            elif c in "-+":
                result += current_num * sign
                current_num = 0
                if c == "-":
                    sign = -1
                else:
                    sign = 1
            elif c == "(":
                stack.append(result)
                stack.append(sign)
                result = 0
                sign = 1
            elif c == ")":
                result += current_num * sign
                result *= stack.pop()  # this will be the last previous sign seen
                result += stack.pop()  # this will be the previous result computed
                current_num = 0

        return result + (current_num * sign)


# 221. Maximal Square
# https://leetcode.com/problems/maximal-square/description/


class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0] * n for _ in range(m)]

        max_value = 0

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                    max_value = max(max_value, dp[i][j])

        return max_value**2


# 223. Rectangle Area
# https://leetcode.com/problems/rectangle-area/description/


class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        intersection_x = min(ax2, bx2) - max(ax1, bx1)
        intersection_y = min(ay2, by2) - max(ay1, by1)

        intersection = intersection_x * intersection_y
        if intersection_x <= 0 or intersection_y <= 0:
            intersection = 0

        def area(x1, x2, y1, y2):
            return (x2 - x1) * (y2 - y1)

        return area(ax1, ax2, ay1, ay2) + area(bx1, bx2, by1, by2) - intersection


# numbers in this set 173, 179, 187, 189, 198, 199, 200, 201, 204,207,208, 210, 211, 215, 224, 221,223
