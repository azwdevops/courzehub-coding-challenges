from typing import List, Optional
from collections import defaultdict


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 47. Permutations II
# https://leetcode.com/problems/permutations-ii/description/


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result = []
        freq = {item: 0 for item in nums}

        for item in nums:
            freq[item] += 1

        def recursive(current):
            if len(current) == n:
                result.append(current[:])
                return

            for item in freq:
                if freq[item] > 0:
                    freq[item] -= 1
                    current.append(item)
                    recursive(current)
                    current.pop()
                    freq[item] += 1

        recursive([])

        return result


# 48. Rotate Image
# https://leetcode.com/problems/rotate-image/description/


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        left = 0
        right = n - 1

        while left < right:
            for i in range(right - left):
                top, bottom = left, right

                # save the top left
                topLeft = matrix[top][left + i]

                # move bottom left to top left
                matrix[top][left + i] = matrix[bottom - i][left]

                # move bottom right into bottom left
                matrix[bottom - i][left] = matrix[bottom][right - i]

                # move the top right into bottom right
                matrix[bottom][right - i] = matrix[top + i][right]

                # move top left into top right
                matrix[top + i][right] = topLeft

            right -= 1
            left += 1


# 49. Group Anagrams
# https://leetcode.com/problems/group-anagrams/description/


# Time Complexity O(n log n)
# Space Complexity O(n)
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        group = defaultdict(list)

        for word in strs:
            sorted_word = "".join(sorted(word))
            group[sorted_word].append(word)

        return list(group.values())


# 50. Pow(x, n)
# https://leetcode.com/problems/powx-n/description/


class Solution:
    def myPow(self, x: float, n: int) -> float:
        result = 1

        unsigned_n = abs(n)

        while unsigned_n > 0:
            if unsigned_n % 2 == 1:
                result *= x
                unsigned_n -= 1

            else:
                x = x * x
                unsigned_n /= 2

        if n < 0:
            return 1 / result

        return result


# 53. Maximum Subarray
# https://leetcode.com/problems/maximum-subarray/description/

# Time Complexity O(n)
# Space Complexity O(1)


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = float("-inf")
        current_sum = 0

        n = len(nums)

        for i in range(n):
            current_sum += nums[i]

            max_sum = max(max_sum, current_sum)
            if current_sum < 0:
                current_sum = 0

        return max_sum


# 54. Spiral Matrix
# https://leetcode.com/problems/spiral-matrix/description/


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        n = len(matrix)
        m = len(matrix[0])

        left = 0
        right = m - 1
        top = 0
        bottom = n - 1

        result = []

        while top <= bottom and left <= right:
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1

            for i in range(top, bottom + 1):
                result.append(matrix[i][right])

            right -= 1

            if top <= bottom:
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])

                bottom -= 1

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])

                left += 1

        return result


# 55. Jump Game
# https://leetcode.com/problems/jump-game/description/

# Time Complexity O(n)
# Space Complexity O(1)


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        farthest = 0
        lastIndex = n - 1

        for i in range(n):
            if i > farthest:
                return False
            if i + nums[i] > farthest:
                farthest = i + nums[i]
            if farthest >= lastIndex:
                return True


# 56. Merge Intervals
# https://leetcode.com/problems/merge-intervals/description/
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()

        n = len(intervals)
        result = []

        for i in range(n):
            if len(result) == 0 or result[-1][1] < intervals[i][0]:
                result.append(intervals[i])
            else:
                result[-1][1] = max(result[-1][1], intervals[i][1])

        return result


# 57. Insert Interval
# https://leetcode.com/problems/insert-interval/description/


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        n = len(intervals)

        result = []
        i = 0

        # left before new Interval
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        # where intervals overlap
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1

        result.append(newInterval)

        # intervals after newInterval
        while i < n:
            result.append(intervals[i])
            i += 1

        return result


# 59. Spiral Matrix II
# https://leetcode.com/problems/spiral-matrix-ii/description/


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        top = 0
        bottom = n - 1
        left = 0
        right = n - 1

        result = [[0 for _ in range(n)] for _ in range(n)]

        val = 1

        while top <= bottom:
            for j in range(left, right + 1):
                result[top][j] = val
                val += 1
            top += 1

            for i in range(top, bottom + 1):
                result[i][right] = val
                val += 1
            right -= 1

            for j in range(right, left - 1, -1):
                result[bottom][j] = val
                val += 1
            bottom -= 1

            for i in range(bottom, top - 1, -1):
                result[i][left] = val
                val += 1
            left += 1

        return result


# 61. Rotate List
# https://leetcode.com/problems/rotate-list/description/


class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or k == 0:
            return head
        current = head
        n = 0
        while current:
            current = current.next
            n += 1
        k = k % n

        left = head
        right = head
        count = 0

        while right.next:
            if count >= k:
                left = left.next
            right = right.next
            count += 1

        right.next = head
        new_head = left.next
        left.next = None

        return new_head


# 62. Unique Paths
# https://leetcode.com/problems/unique-paths/description/


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        result = [[0] * m for _ in range(n)]

        result[0][0] = 1

        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    result[0][j] = result[0][j - 1]
                elif j == 0:
                    result[i][0] = result[i - 1][0]
                else:
                    result[i][j] = result[i - 1][j] + result[i][j - 1]

        return result[-1][-1]


# 64. Minimum Path Sum
# https://leetcode.com/problems/minimum-path-sum/description/


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])

        result = [[0] * m for _ in range(n)]

        result[0][0] = grid[0][0]

        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    continue
                if i == 0:
                    result[0][j] = result[0][j - 1] + grid[0][j]
                elif j == 0:
                    result[i][0] = result[i - 1][0] + grid[i][0]
                else:
                    result[i][j] = min(result[i - 1][j], result[i][j - 1]) + grid[i][j]
        return result[-1][-1]


# 71. Simplify Path
# https://leetcode.com/problems/simplify-path/description/

# Time Complexity O(n)
# Space Complexity O(n)


class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []

        for dir_name in path.split("/"):
            if dir_name == "..":
                if stack:
                    stack.pop()
            elif dir_name == "." or dir_name == "":
                continue
            else:
                stack.append(dir_name)

        return "/" + "/".join(stack)


# 72. Edit Distance
# https://leetcode.com/problems/edit-distance/description/


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}

        def recursive(i, j):
            if (i, j) in memo:
                return memo[(i, j)]

            if i == -1:
                return j + 1
            if j == -1:
                return i + 1

            if word1[i] == word2[j]:
                memo[(i, j)] = 0 + recursive(i - 1, j - 1)
                return memo[(i, j)]
            else:
                delete_char = 1 + recursive(i - 1, j)
                insert_char = 1 + recursive(i, j - 1)
                replace_char = 1 + recursive(i - 1, j - 1)

                memo[(i, j)] = min(delete_char, insert_char, replace_char)
                return memo[(i, j)]

        return recursive(len(word1) - 1, len(word2) - 1)


# 73. Set Matrix Zeroes
# https://leetcode.com/problems/set-matrix-zeroes/description/


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows = set()
        cols = set()

        n = len(matrix)
        m = len(matrix[0])

        for i in range(n):
            for j in range(m):
                if i in rows and j in cols:
                    continue
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        for i in range(n):
            for j in range(m):
                if i in rows or j in cols:
                    matrix[i][j] = 0


# numbers in this set 47, 48, 49, 50, 53, 54, 55, 56, 57, 59, 61, 62, 64, 71, 72, 73
