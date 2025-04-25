from typing import List, Optional
from collections import deque, defaultdict

# 131. Palindrome Partitioning
# https://leetcode.com/problems/palindrome-partitioning/description/


class NodeWithRandom:
    def __init__(self, x: int, next: "NodeWithRandom" = None, random: "NodeWithRandom" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class ListNode:
    def __init__(self, x, next=None):
        self.val = x
        self.next = next


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def isPalidrome(s_part):
            return s_part == s_part[::-1]

        n = len(s)

        result = []

        def recursive(start, current):
            if start == n:
                result.append(current[:])
                return
            for end in range(start + 1, n + 1):
                if isPalidrome(s[start:end]):
                    recursive(end, current + [s[start:end]])

        recursive(0, [])
        return result


# 134. Gas Station
# https://leetcode.com/problems/gas-station/description/


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        start_index = 0
        total_gas = 0
        current_gas = 0

        for i in range(n):
            current_gas += gas[i] - cost[i]
            total_gas += gas[i] - cost[i]

            if current_gas < 0:
                start_index = i + 1
                current_gas = 0

        return start_index if total_gas >= 0 else -1


# 137. Single Number II
# https://leetcode.com/problems/single-number-ii/description/


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        n = len(nums)

        for bitIndex in range(32):
            count = 0
            for i in range(n):
                if nums[i] & (1 << bitIndex):  # check if the bit at bitIndex is set
                    count += 1
            # check if set bits count divided by 3 is 1 is so, the bit of the single number is set
            if count % 3 == 1:  # means bit of single number at bitIndex is set
                result = result | (1 << bitIndex)

        if result >= 2**31:
            result -= 2**32

        return result


# 138. Copy List with Random Pointer
# https://leetcode.com/problems/copy-list-with-random-pointer/description/


class Solution:
    def copyRandomList(self, head: Optional[NodeWithRandom]) -> Optional[NodeWithRandom]:
        if not head:
            return None
        stack = [head]
        nodesMap = {}
        nodesMap[head] = NodeWithRandom(head.val)

        while stack:
            current = stack.pop()

            if current.random:
                if current.random not in nodesMap:
                    nodesMap[current.random] = NodeWithRandom(current.random.val)
                    stack.append(current.random)
                nodesMap[current].random = nodesMap[current.random]
            if current.next:
                if current.next not in nodesMap:
                    nodesMap[current.next] = NodeWithRandom(current.next.val)
                    stack.append(current.next)
                nodesMap[current].next = nodesMap[current.next]

        return nodesMap[head]


# 139. Word Break
# https://leetcode.com/problems/word-break/
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False for _ in range(n + 1)]
        dp[0] = True

        for end in range(1, n + 1):
            for start in range(end):
                if dp[start] == True and s[start:end] in wordDict:
                    dp[end] = True
                    break

        return dp[n]


# 142. Linked List Cycle II
# https://leetcode.com/problems/linked-list-cycle-ii/description/
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        visited = set()
        current = head

        while current:
            if current in visited:
                return current
            visited.add(current)
            current = current.next

        return None


# 143. Reorder List
# https://leetcode.com/problems/reorder-list/description/


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """

        queue = deque([])
        current = head

        while current:
            queue.append(current)
            current = current.next

        while queue:
            first = queue.popleft()
            last = None
            if queue:
                last = queue.pop()

            temp = first.next

            first.next = last

            if last:
                last.next = temp
        if not last:
            first.next = None
        else:
            last.next = None


# 146. LRU Cache
# https://leetcode.com/problems/lru-cache/description/


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.recent = deque([])
        self.items = {}

    def get(self, key: int) -> int:
        if key in self.items:
            self.recent.remove(key)
            self.recent.append(key)
            return self.items[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.recent:
            self.recent.remove(key)

        elif len(self.items.keys()) == self.capacity:
            del self.items[self.recent[0]]
            self.recent.popleft()

        self.items[key] = value
        self.recent.append(key)


# 147. Insertion Sort List
# https://leetcode.com/problems/insertion-sort-list/description/


class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        prev = head
        current = head.next

        while current:
            if current.val >= prev.val:
                prev = current
                current = current.next
                continue

            temp = dummy
            while current.val > temp.next.val:
                temp = temp.next

            prev.next = current.next
            current.next = temp.next
            temp.next = current
            current = prev.next

        return dummy.next


# 150. Evaluate Reverse Polish Notation
# https://leetcode.com/problems/evaluate-reverse-polish-notation/description/


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        n = len(tokens)
        if n == 1:
            return int(tokens[0])
        operators = set(["+", "-", "*", "/"])
        stack = []

        for i in range(n):
            if tokens[i] in operators:
                operator = tokens[i]
                operand1 = int(stack.pop())
                operand2 = int(stack.pop())

                if operator == "+":
                    result = operand2 + operand1
                elif operator == "-":
                    result = operand2 - operand1
                elif operator == "*":
                    result = operand2 * operand1
                elif operator == "/":
                    result = int(operand2 / operand1)
                stack.append(result)
            else:
                stack.append(tokens[i])

        return stack[0]


# 151. Reverse Words in a String
# https://leetcode.com/problems/reverse-words-in-a-string/description/


class Solution:
    def reverseWords(self, s: str) -> str:
        stack = s.split(" ")

        result = ""

        while stack:
            current = stack.pop()
            if current == "":
                continue
            result += " " + current

        return result.strip()


# 152. Maximum Product Subarray
# https://leetcode.com/problems/maximum-product-subarray/description/


class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        max_product = nums[0]
        max_ending = nums[0]
        min_ending = nums[0]

        for num in nums[1:]:
            if num < 0:
                min_ending, max_ending = max_ending, min_ending

            max_ending = max(num, max_ending * num)
            min_ending = min(num, min_ending * num)

            max_product = max(max_product, max_ending)

        return max_product


# 153. Find Minimum in Rotated Sorted Array
# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/


class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        low = 0
        high = n - 1

        min_item = float("inf")

        while low <= high:
            mid = (low + high) // 2

            if nums[low] <= nums[mid]:
                min_item = min(min_item, nums[low])
                low = mid + 1
            else:
                min_item = min(min_item, nums[mid])
                high = mid - 1

        return min_item


# 155. Min Stack
# https://leetcode.com/problems/min-stack/description/


class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(self.min_stack[-1], val))

    def pop(self) -> None:
        self.min_stack.pop()
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# 164. Maximum Gap
# https://leetcode.com/problems/maximum-gap/description/


class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        low, high = min(nums), max(nums)

        bucket = defaultdict(list)
        for num in nums:
            if num == high:
                index = n - 1
            else:
                index = abs(low - num) * (n - 1) // (high - low)
            bucket[index].append(num)

        temp = []
        for i in range(n):
            if bucket[i]:
                temp.append((min(bucket[i]), max(bucket[i])))

        output = 0
        for i in range(1, len(temp)):
            output = max(output, abs(temp[i - 1][-1] - temp[i][0]))

        return output


# 165. Compare Version Numbers
# https://leetcode.com/problems/compare-version-numbers/description/


class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        queue1 = deque(version1.split("."))
        queue2 = deque(version2.split("."))
        n = len(queue1)
        m = len(queue2)

        if n < m:
            queue1.extend(["0"] * (m - n))
        elif n > m:
            queue2.extend(["0"] * (n - m))

        while queue1 and queue2:
            v1 = int(queue1.popleft())
            v2 = int(queue2.popleft())
            if v1 > v2:
                return 1
            elif v2 > v1:
                return -1

        return 0


# 166. Fraction to Recurring Decimal
# https://leetcode.com/problems/fraction-to-recurring-decimal/description/


class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator == 0:
            return "0"
        result = []
        if (numerator < 0) != (denominator < 0):
            result.append("-")

        numerator = abs(numerator)
        denominator = abs(denominator)

        # integer part
        result.append(str(numerator // denominator))
        remainder = numerator % denominator

        if remainder == 0:
            return "".join(result)

        result.append(".")

        remainder_map = {}

        while remainder != 0:
            if remainder in remainder_map:
                insert_index = remainder_map[remainder]
                result.insert(insert_index, "(")
                result.append(")")
                break

            remainder_map[remainder] = len(result)
            remainder *= 10
            result.append(str(remainder // denominator))

            remainder %= denominator

        return "".join(result)


# 167. Two Sum II - Input Array Is Sorted
# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left + 1, right + 1]
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                left += 1


# 172. Factorial Trailing Zeroes
# https://leetcode.com/problems/factorial-trailing-zeroes/description/


class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0

        multiples = 5

        while multiples <= n:
            count += n // multiples
            multiples *= 5

        return count


# numbers in this set 131, 134, 137, 138, 139, 142, 146, 147, 150, 151, 152, 153, 164, 165, 166, 167, 172
