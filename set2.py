from typing import List, Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 201. Bitwise AND of Numbers Range
# https://leetcode.com/problems/bitwise-and-of-numbers-range/description/


class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        count = 0
        while left != right:
            left >>= 1
            right >>= 1
            count += 1

        return left << count


# 476. Number Complement
# https://leetcode.com/problems/number-complement/description/


class Solution:
    def findComplement(self, num: int) -> int:
        result = 0
        i = 0

        while num:
            if num & 1 == 0:
                result += 1 << i

            i += 1
            num >> 1

        return result


# 22. Generate Parentheses
# https://leetcode.com/problems/generate-parentheses/description/


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []

        def generate(open, close, current):
            if open == close == 0:
                result.append(current)
                return
            if open > 0:
                generate(open - 1, close, current + "(")
            if open < close:
                generate(open, close - 1, current + ")")

        generate(n, n, "")

        return result


# 24. Swap Nodes in Pairs
# https://leetcode.com/problems/swap-nodes-in-pairs/description/


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        prev = dummy
        curr = head

        while curr and curr.next:
            # save temp pointers
            nextPair = curr.next.next
            temp = curr

            # reverse current pair
            curr = curr.next
            curr.next = temp

            temp.next = nextPair
            prev.next = curr

            curr = nextPair
            prev = temp

        return dummy.next


# https://leetcode.com/problems/divide-two-integers/
# 29. Divide Two Integers


class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if dividend == divisor:
            return 1

        sign = True
        if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
            sign = False
        num = abs(dividend)
        denom = abs(divisor)

        quotient = 0

        while num >= denom:
            count = 0
            while num >= (denom << count + 1):
                count += 1

            num -= denom << count

            quotient += 1 << count

        if quotient > (1 << 31) - 1 and sign:
            return (1 << 31) - 1
        elif quotient < -(1 << 31) and not sign:
            return -(1 << 31)
        else:
            return quotient if sign else -quotient


# 31. Next Permutation
# https://leetcode.com/problems/next-permutation/description/


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        index = -1
        n = len(nums)

        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                index = i
                break

        if index == -1:
            nums.reverse()

        else:
            for i in range(n - 1, index, -1):
                if nums[i] > nums[index]:
                    nums[i], nums[index] = nums[index], nums[i]
                    break

            left = index + 1
            right = n - 1

            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1


# 33. Search in Rotated Sorted Array
# https://leetcode.com/problems/search-in-rotated-sorted-array/description/


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        low = 0
        high = n - 1

        while low <= high:
            mid = (low + high) // 2

            if nums[mid] == target:
                return mid

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

        return -1


# 34. Find First and Last Position of Element in Sorted Array
# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)

        def upperBound(arr, n, x):
            low = 0
            high = n - 1
            result = n

            while low <= high:
                mid = (low + high) // 2
                if arr[mid] > x:
                    result = mid
                    # look for more smaller index on left
                    high = mid - 1
                else:
                    low = mid + 1  # look on the right
            return result

        def lowerBound(arr, n, x):
            low = 0
            high = n - 1
            result = n

            while low <= high:
                mid = (low + high) // 2
                if arr[mid] >= x:
                    result = mid
                    # look for more smaller index on left
                    high = mid - 1
                else:
                    # look on the right
                    low = mid + 1
            return result

        lower_bound = lowerBound(nums, n, target)
        if lower_bound == n or nums[lower_bound] != target:
            return [-1, -1]
        return [lower_bound, upperBound(nums, n, target) - 1]


# 36. Valid Sudoku
# https://leetcode.com/problems/valid-sudoku/description/

# Time Complexity O(1)
# Space Complexity O(1)


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        grid = [set() for _ in range(9)]

        for row in range(9):
            for col in range(9):
                num = board[row][col]
                if num == ".":
                    continue

                grid_index = (col // 3) * 3 + row // 3  # could also be (row // 3) * 3 + col // 3

                if num in rows[row] or num in cols[col] or num in grid[grid_index]:
                    return False

                rows[row].add(num)
                cols[col].add(num)
                grid[grid_index].add(num)

        return True


# 38. Count and Say
# https://leetcode.com/problems/count-and-say/description/


class Solution:
    def countAndSay(self, n: int) -> str:
        def helper(s):
            result = ""
            count = 1

            for i in range(len(s)):
                if i == len(s) - 1 or s[i] != s[i + 1]:
                    result += str(count) + s[i]
                    count = 1
                else:
                    count += 1

            return result

        ans = "1"

        for _ in range(2, n + 1):
            temp = helper(ans)
            ans = temp

        return ans


# 39. Combination Sum
# https://leetcode.com/problems/combination-sum/description/


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        result = []
        n = len(candidates)

        def recursive(current, startIndex, target):
            if target == 0:
                result.append(current)
                return
            for i in range(startIndex, n):
                if candidates[i] > target:
                    break
                recursive(current + [candidates[i]], i, target - candidates[i])

            return

        recursive([], 0, target)

        return result


# 40. Combination Sum II
# https://leetcode.com/problems/combination-sum-ii/description/


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        n = len(candidates)
        result = []

        def recursive(current, startIndex, target):
            if target == 0:
                result.append(current)
                return

            for i in range(startIndex, n):
                if i > startIndex and candidates[i] == candidates[i - 1]:
                    continue
                if candidates[i] > target:
                    break
                recursive(current + [candidates[i]], i + 1, target - candidates[i])

        recursive([], 0, target)

        return result


# 43. Multiply Strings
# https://leetcode.com/problems/multiply-strings/description/


class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if "0" in [num1, num2]:
            return "0"

        num1, num2 = num1[::-1], num2[::-1]
        n, m = len(num1), len(num2)

        result = [0] * (n + m)

        for i1 in range(n):
            for i2 in range(m):
                digit = int(num1[i1]) * int(num2[i2])
                result[i1 + i2] += digit  # first we add the digit
                result[i1 + i2 + 1] += result[i1 + i2] // 10

                result[i1 + i2] = result[i1 + i2] % 10  # since it may be two digit we mode it to only take the last val

        result = result[::-1]
        start = 0

        while start < (n + m) and result[start] == 0:
            start += 1

        result_str = map(str, result[start:])

        return "".join(result_str)


# 45. Jump Game II
# https://leetcode.com/problems/jump-game-ii/description/


class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        memo = {}

        def recursive(index):
            if index in memo:
                return memo[index]
            if index >= n - 1:
                return 0
            if nums[index] == 0:
                return float("inf")
            min_jumps = float("inf")
            for step in range(1, nums[index] + 1):
                min_jumps = min(min_jumps, 1 + recursive(index + step))
            memo[index] = min_jumps
            return min_jumps

        return recursive(0)


# 46. Permutations
# https://leetcode.com/problems/permutations/


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        n = len(nums)

        def recursive(current):
            if len(current) == n:
                result.append(current[:])
                return

            for i in range(n):
                if nums[i] not in current:
                    current.append(nums[i])
                    recursive(current)
                    current.pop()

        recursive([])

        return result


# numbers in this set 201, 476, 22, 24, 31, 33, 34, 36, 38, 39, 40, 43, 45, 46
