from typing import List, Optional
from collections import deque


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 74. Search a 2D Matrix
# https://leetcode.com/problems/search-a-2d-matrix/description/


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix)
        m = len(matrix[0])

        low = 0
        high = (n * m) - 1

        while low <= high:
            mid = (low + high) // 2

            row = mid // m
            col = mid % m

            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                high = mid - 1
            else:
                low = mid + 1
        return False


# 75. Sort Colors
# https://leetcode.com/problems/sort-colors/description/


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)

        low = 0
        mid = 0
        high = n - 1

        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1


# 77. Combinations
# https://leetcode.com/problems/combinations/description/


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []

        def recursive(start, current, count):
            if count == 0:
                result.append(current[:])
                return
            if start == n + 1:
                return

            for i in range(start, n + 1):
                recursive(i + 1, current + [i], count - 1)

        recursive(1, [], k)

        return result


# 78. Subsets
# https://leetcode.com/problems/subsets/description/


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = [[]]

        for i in nums:
            result += [new_list + [i] for new_list in result]

        return result


# 79. Word Search
# https://leetcode.com/problems/word-search/description/


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        word_len = len(word)
        n = len(board)
        m = len(board[0])

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def dfs(index, row, col):
            if index == word_len:
                return True
            if (
                row == -1
                or row == n
                or col == -1
                or col == m
                or word[index] != board[row][col]
                or board[row][col] == ""
            ):
                return False

            removed_char = board[row][col]
            board[row][col] = ""

            for delta_row, delta_col in directions:
                nrow = row + delta_row
                ncol = col + delta_col
                if dfs(index + 1, nrow, ncol):
                    return True

            board[row][col] = removed_char
            return False

        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    if dfs(0, i, j):
                        return True

        return False


# 80. Remove Duplicates from Sorted Array II
# https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left = 0
        right = 0

        while right < len(nums) and left <= right:
            while right < len(nums) and nums[left] == nums[right]:
                if right - left > 1:
                    nums.pop(right)
                else:
                    right += 1
            left = right

        return len(nums)


# 81. Search in Rotated Sorted Array II
# https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/


class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        n = len(nums)
        low = 0
        high = n - 1

        while low <= high:
            mid = (low + high) // 2

            if nums[mid] == target:
                return True
            elif nums[low] == nums[mid] == nums[high]:
                low += 1
                high -= 1
                continue
            elif nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return False


# 82. Remove Duplicates from Sorted List II
# https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/


class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        dummy = ListNode(0, head)
        dummy.next = head
        prev = dummy
        current = head

        while current:
            if current.next and current.val == current.next.val:
                while current.next and current.val == current.next.val:
                    current = current.next
                prev.next = current.next
            else:
                prev = prev.next

            current = current.next
        return dummy.next


# 86. Partition List
# https://leetcode.com/problems/partition-list/description/


class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        greater = deque()
        lesser = deque()

        current = head

        while current:
            if current.val < x:
                lesser.append(current)
            else:
                greater.append(current)
            current = current.next

        dummy = ListNode(0, None)

        current = dummy

        while lesser:
            current.next = lesser.popleft()
            current = current.next

        while greater:
            current.next = greater.popleft()
            current = current.next

        current.next = None

        return dummy.next


# 89. Gray Code
# https://leetcode.com/problems/gray-code/


class Solution:
    def grayCode(self, n: int) -> List[int]:
        result = []

        for i in range(1 << n):
            result.append(i ^ (i >> 1))

        return result


# 90. Subsets II
# https://leetcode.com/problems/subsets-ii/description/


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)

        result = []

        def recursive(current, start):
            result.append(current[:])

            for i in range(start, n):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                current.append(nums[i])
                recursive(current, i + 1)
                current.pop()

        recursive([], 0)

        return result


# 91. Decode Ways
# https://leetcode.com/problems/decode-ways/description/


class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == "0":
            return 0
        n = len(s)

        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        dp[1] = 1

        for i in range(2, n + 1):
            ways = 0
            if s[i - 1] != "0":
                ways += dp[i - 1]
            if 10 <= int(s[i - 2 : i]) <= 26:
                ways += dp[i - 2]
            dp[i] = ways

        return dp[-1]


# 92. Reverse Linked List II
# https://leetcode.com/problems/reverse-linked-list-ii/description/


class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        stack = []

        position = 0
        dummy = ListNode(0, head)
        current = dummy

        while position < left - 1:
            current = current.next
            position += 1

        before = current

        current = current.next

        for _ in range(left, right + 1):
            stack.append(current)
            current = current.next

        after = current

        while stack:
            before.next = stack.pop()
            before = before.next

        before.next = after

        return dummy.next


# 93. Restore IP Addresses
# https://leetcode.com/problems/restore-ip-addresses/description/


class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []
        n = len(s)

        def recursive(start, count, current):
            if start == n and count == 4:
                result.append(current[:-1])  # remove trailing .
                return

            if start >= n or count >= 4:
                return

            for i in range(1, 4):
                if start + i > n:
                    break

                segment = s[start : start + i]
                if segment.startswith("0") and len(segment) > 1 or int(segment) > 255:
                    continue
                recursive(start + i, count + 1, current + segment + ".")

        recursive(0, 0, "")

        return result


# 95. Unique Binary Search Trees II
# https://leetcode.com/problems/unique-binary-search-trees-ii/description/


class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:

        def generate(left, right):
            if left == right:
                return [TreeNode(left)]
            if left > right:
                return [None]

            result = []

            for val in range(left, right + 1):
                for leftTree in generate(left, val - 1):
                    for rightTree in generate(val + 1, right):
                        root = TreeNode(val, leftTree, rightTree)
                        result.append(root)

            return result

        return generate(1, n)


# 96. Unique Binary Search Trees
# https://leetcode.com/problems/unique-binary-search-trees/description/


class Solution:
    def numTrees(self, n: int) -> int:
        dp = [1] * (n + 1)

        for nodes in range(2, n + 1):
            total = 0
            for root in range(1, nodes + 1):
                left = root - 1
                right = nodes - root

                total += dp[left] * dp[right]

            dp[nodes] = total

        return dp[n]


# 97. Interleaving String
# https://leetcode.com/problems/interleaving-string/description/


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False

        memo = {}

        def recursion(i, j):
            if (i, j) in memo:
                return memo[(i, j)]

            if i == len(s1) and j == len(s2):
                return True

            if i < len(s1) and s1[i] == s3[i + j]:
                if recursion(i + 1, j):
                    memo[(i + 1, j)] = True
                    return True

            if j < len(s2) and s2[j] == s3[i + j]:
                if recursion(i, j + 1):
                    memo[(i, j + 1)] = True
                    return True
            memo[(i, j)] = False
            return False

        return recursion(0, 0)


# numbers in this set 74, 75, 77, 78, 79, 80, 81, 82, 86, 89, 90, 91, 92, 93, 95, 97
