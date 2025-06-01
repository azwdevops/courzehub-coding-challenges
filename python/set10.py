from typing import List


# 300. Longest Increasing Subsequence
# https://leetcode.com/problems/longest-increasing-subsequence/description/


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)

        dp = [1] * n

        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])

        return max(dp)


# 304. Range Sum Query 2D - Immutable
# https://leetcode.com/problems/range-sum-query-2d-immutable/description/


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        n = len(matrix)
        m = len(matrix[0])
        self.updated_matrix = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                self.updated_matrix[i][j] = (
                    self.updated_matrix[i - 1][j]
                    + self.updated_matrix[i][j - 1]
                    - self.updated_matrix[i - 1][j - 1]
                    + matrix[i - 1][j - 1]
                )

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        # since we did n + 1 and m + 1, it means we also need to adjust the row1, col1, row2, col2
        row1 += 1
        row2 += 1
        col1 += 1
        col2 += 1
        return (
            self.updated_matrix[row2][col2]
            - self.updated_matrix[row1 - 1][col2]
            - self.updated_matrix[row2][col1 - 1]
            + self.updated_matrix[row1 - 1][col1 - 1]
        )


# 306. Additive Number
# https://leetcode.com/problems/additive-number/description/
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:

        def dfs(num1, num2, num3):
            if (
                (len(num1) > 1 and num1[0] == "0")
                or (len(num2) > 1 and num2[0] == "0")
                or (len(num3) > 1 and num3[0] == "0")
            ):
                return False
            target_sum = sum_two(num1, num2)
            if target_sum == num3:
                return True  #
            length = len(target_sum)
            if len(num3) <= length:
                return False

            # now num3 is longer than target_sum
            if target_sum == num3[:length]:
                return dfs(num2, target_sum, num3[length:])
            else:
                return False

        def sum_two(num1, num2):
            ans = []
            p1 = len(num1) - 1
            p2 = len(num2) - 1
            plusOne = 0

            while p1 >= 0 and p2 >= 0:
                plusOne, current_value = divmod(int(num1[p1]) + int(num2[p2]) + plusOne, 10)
                ans.append(str(current_value))
                p1 -= 1
                p2 -= 1

            while p1 >= 0:
                plusOne, current_value = divmod(int(num1[p1]) + plusOne, 10)
                ans.append(str(current_value))
                p1 -= 1

            while p2 >= 0:
                plusOne, current_value = divmod(int(num1[p2]) + plusOne, 10)
                ans.append(current_value)
                p2 -= 1

            if plusOne:
                ans.append("1")
            return "".join(ans[::])

        n = len(num)
        # find all num1 and num2 combinations
        for i in range(n):
            for j in range(i):
                num1 = num[: j + 1]
                num2 = num[j + 1 : i + 1]
                num3 = num[i + 1 :]
                if dfs(num1, num2, num3):
                    return True

        return False


# numbers in this set 300, 304, 306
