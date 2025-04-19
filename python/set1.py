from typing import List, Optional
from collections import deque


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 220. Contains Duplicate III
# https://leetcode.com/problems/contains-duplicate-iii


# this solution gives TLE at the moment
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        n = len(nums)
        if valueDiff == 0 and n == len(set(nums)):
            return False
        for i in range(n):
            for j in range(i + 1, i + 1 + indexDiff):
                if j >= n:
                    break
                if abs(nums[i] - nums[j]) <= valueDiff:
                    return True
        return False


# 2425. Bitwise XOR of All Pairings
# https://leetcode.com/problems/bitwise-xor-of-all-pairings/


class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        ans = 0
        m = len(nums1)
        if m & 1:
            for el in nums2:
                ans ^= el
        n = len(nums2)
        if n & 1:
            for el in nums1:
                ans ^= el

        return ans


# 273. Integer to English Words
# https://leetcode.com/problems/integer-to-english-words/description/


class Solution:
    def numberToWords(self, num: int) -> str:
        unitPlace = {
            "0": "",
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
        }

        tenPlace = {
            "0": "",
            "1": "Ten",
            "2": "Twenty",
            "3": "Thirty",
            "4": "Forty",
            "5": "Fifty",
            "6": "Sixty",
            "7": "Seventy",
            "8": "Eighty",
            "9": "Ninety",
        }
        oneCase = {
            "0": "Ten",
            "1": "Eleven",
            "2": "Twelve",
            "3": "Thirteen",
            "4": "Fourteen",
            "5": "Fifteen",
            "6": "Sixteen",
            "7": "Seventeen",
            "8": "Eighteen",
            "9": "Nineteen",
        }

        n = num
        digits = []

        while n:
            digits.append(n % 10)
            n /= 10

        while len(digits) < 10:
            digits.append(0)

        return self.digits_2_string(digits, unitPlace, tenPlace, oneCase)

    def digits_2_string(self, digits, unitPlace, tenPlace, oneCase):
        result = ""
        # for billion
        if digits[9]:
            result += unitPlace[digits[9]] + " Billion "

        #  for million
        if digits[8]:
            result += unitPlace[digits[8]] + " hundred "
        if digits[7] == 1:
            result += oneCase[digits[6]] + " "
        else:
            if digits[7]:
                result += tenPlace[digits[7]] + " "
            if digits[6]:
                result += unitPlace[digits[6]] + " "

        if digits[8] or digits[7] or digits[6]:
            result += " Million "

        # for thousand
        if digits[5]:
            result += unitPlace[digits[5]] + " hundred "
        if digits[4] == 1:
            result += oneCase[digits[3]] + " "
        else:
            if digits[4]:
                result += tenPlace[digits[4]] + " "
            if digits[3]:
                result += unitPlace[digits[3]]
        if digits[5] or digits[4] or digits[3]:
            result += " thousand "

        # for hundred
        if digits[2]:
            result += unitPlace[digits[2]] + " Hundred "
        if digits[1] == 1:
            result += oneCase[digits[0]] + " "
        else:
            if digits[1]:
                result += tenPlace[digits[1]] + " "
            if digits[0]:
                result += unitPlace[digits[0]]

        # remove spaces from the end
        while len(result) and result[-1] == " ":
            result.pop()

        return result


# 3. Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/description/


# Time Complexity O(n)
# Space Complexity O(n)
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = 0
        end = 0
        max_length = 0
        index_dict = {}

        while end < len(s):
            if s[end] in index_dict and index_dict[s[end]] >= start:
                start = index_dict[s[end]] + 1
            max_length = max(max_length, end - start + 1)
            index_dict[s[end]] = end
            end += 1

        return max_length


# 5. Longest Palindromic Substring
# https://leetcode.com/problems/longest-palindromic-substring/description/


# Time Complexity O(n^2)
# Space Complexity O(1)
class Solution:
    def isPalidrome(self, string, left, right):
        while left > 0 and right < (len(string) - 1) and string[left - 1] == string[right + 1]:
            left -= 1
            right += 1

        return left, right, right - left + 1

    def longestPalindrome(self, s: str) -> str:
        max_length = 0
        left = 0
        right = 0
        for index in range(len(s) - 1):
            # middle is a char
            l1, r1, max_len1 = self.isPalidrome(s, index, index)

            if max_len1 > max_length:
                max_length = max_len1
                left, right = l1, r1

            # middle between two characters
            if s[index] == s[index + 1]:
                l2, r2, max_len2 = self.isPalidrome(s, index, index + 1)

                if max_len2 > max_length:
                    max_length = max_len2
                    left, right = l2, r2

        return s[left : right + 1]


# 6. Zigzag Conversion
# https://leetcode.com/problems/zigzag-conversion/description/


# Time Complexity O(n * numRows)
# Space Complexity O(numRows + n)
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s

        current_row = 0
        direction = 1  # means going down

        matrix = [[] for _ in range(numRows)]

        for char in s:
            matrix[current_row].append(char)

            if current_row == 0:
                direction = 1  # means going down
            elif current_row == numRows - 1:
                direction = -1  # means going up
            current_row += direction

        result = ""

        for i in range(numRows):
            result += "".join(matrix[i])

        return result


# 7. Reverse Integer
# https://leetcode.com/problems/reverse-integer/description/


class Solution:
    def reverse(self, x: int) -> int:
        reversed_x = 0
        sign = -1 if x < 0 else 1
        x = abs(x)

        while x != 0:
            remainder = x % 10
            x = x // 10

            reversed_x = (reversed_x * 10) + remainder

        reversed_x *= sign
        if reversed_x > pow(2, 31) - 1 or reversed_x < -(pow(2, 31)):
            return 0

        return reversed_x


# 8. String to Integer (atoi)
# https://leetcode.com/problems/string-to-integer-atoi/description/


# Time Complexity O(n)
# Space Complexity O(1)
class Solution:
    def myAtoi(self, s: str) -> int:
        nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        s = s.strip()
        if len(s) == 0:
            return 0
        sign = 1
        if s[0] == "-":
            queue = deque(list(s[1:]))
            sign = -1
        elif s[0] == "+":
            queue = deque(list(s[1:]))
        else:
            queue = deque(list(s))

        result = 0
        while queue and queue[0] in nums:
            result = result * 10 + int(queue.popleft())

        signed_result = sign * result

        if signed_result > pow(2, 31) - 1:
            return pow(2, 31) - 1
        elif signed_result < -(pow(2, 31)):
            return -(pow(2, 31))

        return signed_result


# https://leetcode.com/problems/container-with-most-water/
# 11. Container With Most Water


# Time Complexity O(n)
# Space Complexity O(1)
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1

        max_vol = 0

        while left < right:
            current_vol = (right - left) * min(height[left], height[right])

            max_vol = max(max_vol, current_vol)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_vol


# https://leetcode.com/problems/integer-to-roman/description/
# 12. Integer to Roman


# Time Complexity O(1)
# Space Complexity O(1)
class Solution:
    def intToRoman(self, num: int) -> str:
        vals = [
            [1000, "M"],
            [900, "CM"],
            [500, "D"],
            [400, "CD"],
            [100, "C"],
            [90, "XC"],
            [50, "L"],
            [40, "XL"],
            [10, "X"],
            [9, "IX"],
            [5, "V"],
            [4, "IV"],
            [1, "I"],
        ]

        result = ""

        for n, symbol in vals:
            count = num // n
            if count > 0:
                result += symbol * count
            num = num % n

        return result


# 15. 3Sum
# https://leetcode.com/problems/3sum/description/


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()

        result = []

        for i in range(n):
            # to avoid repeating triplets
            if i != 0 and nums[i] == nums[i - 1]:
                continue

            left = i + 1
            right = n - 1

            while left < right:
                current_total = nums[i] + nums[left] + nums[right]
                if current_total > 0:
                    right -= 1
                elif current_total < 0:
                    left += 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1

                    # to avoid duplicate triplets
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
        return result


# 16. 3Sum Closest
# https://leetcode.com/problems/3sum-closest/description/

# Time Complexity O(n^2)
# Space Complexity O(1)


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        n = len(nums)
        nums.sort()

        closest_sum = float("inf")

        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = n - 1

            while left < right:
                current_total = nums[i] + nums[left] + nums[right]

                if current_total == target:
                    return current_total

                if abs(current_total - target) < abs(closest_sum - target):
                    closest_sum = current_total

                if current_total < target:
                    left += 1
                else:
                    right -= 1

        return closest_sum


# 17. Letter Combinations of a Phone Number
# https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/

# Time Complexity O(n * 4^n)
# Space Complexity O(1)


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        result = []
        digits_to_char = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(i, currentString):
            if len(currentString) == len(digits):
                result.append(currentString)
                return

            for c in digits_to_char[digits[i]]:
                backtrack(i + 1, currentString + c)

        if digits:
            backtrack(0, "")

        return result


# 18. 4Sum
# https://leetcode.com/problems/4sum/description/

# Time Complexity O(n^3)
# Space Complexity O(1)


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        n = len(nums)
        result = []

        nums.sort()

        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                k = j + 1
                l = n - 1
                while k < l:
                    current_sum = nums[i] + nums[j] + nums[k] + nums[l]
                    if current_sum == target:
                        temp = [nums[i], nums[j], nums[k], nums[l]]
                        result.append(temp)
                        k += 1
                        l -= 1

                        while k < l and nums[k] == nums[k - 1]:
                            k += 1
                        while k < l and nums[l] == nums[l + 1]:
                            l -= 1
                    elif current_sum < target:
                        k += 1
                    else:
                        l -= 1
        return result


# 19. Remove Nth Node From End of List
# https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/

# Time Complexity O(n)
# Space Complexity O(1)


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        p1 = head
        p2 = dummy
        count = 0

        while p1 is not None:
            p1 = p1.next
            count += 1
            if count > n:
                p2 = p2.next

        p2.next = p2.next.next

        return dummy.next


# numbers in this set 220, 2425, 273, 3, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18,19
