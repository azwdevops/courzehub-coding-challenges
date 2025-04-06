from typing import List

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
