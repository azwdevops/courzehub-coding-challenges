import heapq
from copy import deepcopy
from collections import defaultdict, Counter, deque
from typing import List

# 264. Ugly Number II
# https://leetcode.com/problems/ugly-number-ii/


class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap = [1]
        seen = set([1])

        for _ in range(n):
            ugly = heapq.heappop(heap)

            for factor in [2, 3, 5]:
                new_ugly = ugly * factor
                if new_ugly not in seen:
                    seen.add(new_ugly)
                    heapq.heappush(heap, new_ugly)

        return ugly


# Palindrome Permutation
# https://www.naukri.com/code360/problems/palindrome-permutation_1171180?leftPanelTabValue=PROBLEM


def palindromeString(s):
    # Write your code here.
    freq = defaultdict(int)
    for char in s:
        freq[char] += 1

    n = len(s)
    odds = 0

    for key, value in freq.items():
        if value % 2 == 1:
            odds += 1
        if n % 2 == 0 and odds > 0:
            return False
        elif n % 2 == 1 and odds > 1:
            return False

    return True


# 267. Palindrome Permutation II
# https://www.geeksforgeeks.org/problems/pallindrome-patterns0809/1
class Solution:
    def all_palindromes(self, s):
        # Code here
        freq = Counter(s)
        odd_char = ""
        half = []

        for key, value in freq.items():
            if value % 2 == 1:
                if odd_char:
                    return []
                odd_char = key
            half.extend([key] * (value // 2))

        half.sort()
        result = []
        used = [False] * len(half)

        def backtrack(path):
            if len(path) == len(half):
                half_str = "".join(path)
                full_palidrome = half_str + odd_char + half_str[::-1]
                result.append(full_palidrome[:])
                return

            for i in range(len(half)):
                if used[i]:
                    continue
                if i > 0 and half[i] == half[i - 1] and not used[i - 1]:
                    continue
                used[i] = True
                path.append(half[i])
                backtrack(path)
                path.pop()
                used[i] = False

        backtrack([])

        return result


# 271. Encode and Decode Strings
# https://www.geeksforgeeks.org/problems/encode-and-decode-strings/1


class Solution:
    def encode(self, s):
        # code here
        n = len(s)

        encoded = ""
        for word in s:
            encoded += str(len(word)) + "#" + word

        return encoded

    def decode(self, s):
        # code here
        result = []

        start = 0

        n = len(s)

        while start < n:
            end = start
            while s[end] != "#":
                end += 1

            length = int(s[start:end])

            word = s[end + 1 : end + 1 + length]

            result.append(word)
            start = end + 1 + length

        return result


# 274. H-Index
# https://leetcode.com/problems/h-index/description/
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()

        h_index = 0
        n = len(citations)
        for i in range(n):
            h = n - i
            if citations[i] >= h:
                h_index = max(h_index, h)

        return h_index


# 275. H-Index II
# https://leetcode.com/problems/h-index-ii/description/


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)

        low = 0
        high = n - 1

        h_index = 0

        while low <= high:
            mid = (high + low) // 2
            if citations[mid] >= n - mid:
                h_index = n - mid
                high = mid - 1
            else:
                low = mid + 1

        return h_index


# 276. Paint Fence
# https://www.geeksforgeeks.org/problems/painting-the-fence3727/1


class Solution:
    def countWays(self, n, k):
        # code here.
        if n == 0:
            return 0

        if n == 1:
            return k

        if n == 2:
            return k * k

        same = k * 1
        different = k * (k - 1)

        for i in range(3, n + 1):
            prev_same = same
            prev_different = different

            same = prev_different
            different = (prev_same + prev_different) * (k - 1)

        return same + different


# 277. Find the Celebrity
# https://www.geeksforgeeks.org/problems/the-celebrity-problem/1


class Solution:
    def celebrity(self, mat):
        # code here
        n = len(mat)
        knows = [0] * n
        known = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if mat[i][j] == 1:
                    knows[i] += 1
                    known[j] += 1

        for i in range(n):
            if knows[i] == 0 and known[i] == n - 1:
                return i

        return -1


# 279. Perfect Squares
# https://leetcode.com/problems/perfect-squares/description/
class Solution:
    def numSquares(self, n: int) -> int:
        nums = []
        i = 1
        while i * i <= n:
            nums.append(i * i)
            i += 1

        dp = [float("inf") for _ in range(n + 1)]
        dp[0] = 0

        for square in nums:
            for target in range(square, n + 1):
                dp[target] = min(dp[target], 1 + dp[target - square])

        return dp[n]


# 280. Wiggle Sort
# https://www.naukri.com/code360/problems/wiggle-sort_3155169?leftPanelTabValue=SUBMISSION


def wiggleSort(n: int, arr: List[int]) -> List[int]:
    # write your code here
    n = len(arr)
    if n == 1:
        return arr
    bigger = True
    for i in range(1, n):
        if bigger:
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
            bigger = False
        else:
            if arr[i] > arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
            bigger = True

    return arr


# 281. Zigzag Iterator


class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        self.v = []
        for a, b in zip(v1, v2):
            self.v.extend([a, b])
        k = abs(len(v1) - len(v2))
        if len(v1) > len(v2):
            self.v.extend(v1[-k:])
        elif len(v2) > len(v1):
            self.v.extend(v2[-k:])

    def next(self) -> int:
        return self.v.pop(0)

    def hasNext(self) -> bool:
        return self.v


# 284. Peeking Iterator
# https://leetcode.com/problems/peeking-iterator/description/


class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.peekItem = iterator.next()

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.peekItem

    def next(self):
        """
        :rtype: int
        """
        # current = self.v[self.index]
        # self.index += 1
        result = self.peekItem
        self.peekItem = self.iterator.next()

        return result

    def hasNext(self):
        """
        :rtype: bool
        """
        return True if 1 <= self.peekItem <= 1000 else False


# 285. Inorder Successor in BST
# https://www.geeksforgeeks.org/problems/inorder-successor-in-bst/1


class Solution:
    # returns the inorder successor of the Node x in BST (rooted at 'root')
    def inorderSuccessor(self, root, x):
        successor = None

        node = root

        while node:
            if x.data < node.data:
                successor = node
                node = node.left
            else:
                node = node.right

        return successor.data if successor else -1


# 286. Walls and Gates
# https://www.naukri.com/code360/problems/walls-and-gates_1092887?leftPanelTabValue=PROBLEM
def wallsAndGates(a, n, m):
    # Write your Code here.
    queue = deque([])
    for i in range(n):
        for j in range(m):
            if a[i][j] == 0:
                queue.append((i, j))

    directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    while queue:
        row, col = queue.popleft()

        for delta_row, delta_col in directions:
            nrow = row + delta_row
            ncol = col + delta_col
            if 0 <= nrow < n and 0 <= ncol < m and a[nrow][ncol] == 2147483647:
                a[nrow][ncol] = a[row][col] + 1
                queue.append((nrow, ncol))

    return a


# 287. Find the Duplicate Number
#  https://leetcode.com/problems/find-the-duplicate-number/description/


class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = nums[0]
        fast = nums[0]

        # emulating do while loop here
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        fast = nums[0]

        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]

        return slow


# 288. Unique Word Abbreviation
class validWordAbbr:
    def getAbbre(self, word):
        if len(word) <= 2:
            return word
        return word[0] + str(len(word) - 2) + word[-1]

    def __init__(self, dictionary: List[str]):
        self.lookup = defaultdict(list)

        for word in dictionary:
            self.lookup[self.getAbbre(word)].append(word)

    def isUnique(self, word: str) -> bool:
        abbr = self.getAbbre(word)
        if abbr not in self.lookup:
            return True
        if len(self.lookup[abbr]) >= 2:
            return False

        return list(self.lookup[abbr])[0] == word


# 289. Game of Life
# https://leetcode.com/problems/game-of-life/description/


class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

        n = len(board)
        m = len(board[0])

        board_copy = deepcopy(board)

        for i in range(n):
            for j in range(m):
                n_alive = 0
                for delta_row, delta_col in directions:
                    nrow = i + delta_row
                    ncol = j + delta_col

                    if 0 <= nrow < n and 0 <= ncol < m:
                        if board_copy[nrow][ncol] == 1:
                            n_alive += 1
                if board_copy[i][j] == 1:
                    if n_alive < 2 or n_alive > 3:
                        board[i][j] = 0
                elif n_alive == 3:
                    board[i][j] = 1


# 291. Word Pattern II


class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        p = len(pattern)
        n = len(s)

        def canMatch(p_index, s_index, lookup, rlookup):
            if s_index == n and p_index == p:
                return True
            if s_index == n and p_index != p:
                return False

            if p_index == p and s_index != n:
                return False

            if pattern[p_index] in lookup:
                m = lookup[pattern[p_index]]

                if s[s_index : s_index + len(m)] == m:
                    return canMatch(p_index + 1, s_index + len(m), lookup, rlookup)
                return False

            for end in range(s_index, n):
                current = s[s_index : end + 1]

                if current in rlookup:
                    continue
                lookup[pattern[p_index]] = current
                rlookup[current] = pattern[p_index]

                good = canMatch(p_index + 1, end + 1, lookup, rlookup)

                del lookup[pattern[p_index]]
                del rlookup[current]

                if good:
                    return True
            return False

        return canMatch(0, 0, {}, {})


# 294. Flip Game II
class Solution:
    def canWin(self, currentState: str) -> bool:
        def dfs(state):
            for i in range(len(currentState) - 1):
                if (state >> i) & 1 or (state >> (i + 1)) & 1:
                    continue
                if not dfs(state | 1 << i | state | 1 << (i + 1)):
                    return True
            return False

        result = 0
        for i in range(len(currentState)):
            if currentState[i] == "-":
                result += 1 << i

        return dfs(result)


# 298. Binary Tree Longest Consecutive Sequence
# https://www.geeksforgeeks.org/problems/longest-consecutive-sequence-in-binary-tree/1


class Solution:
    # your task is to complete this function
    # function should print the top view of the binary tree
    # Note: You aren't required to print a new line after every test case
    def longestConsecutive(self, root):
        # Code here
        if not root:
            return 0
        max_length = 1
        queue = deque([(root, 1)])

        while queue:
            node, length = queue.popleft()
            max_length = max(max_length, length)

            if node.left:
                if node.left.data - 1 == node.data:
                    queue.append((node.left, length + 1))
                else:
                    queue.append((node.left, 1))
            if node.right:
                if node.right.data - 1 == node.data:
                    queue.append((node.right, length + 1))
                else:
                    queue.append((node.right, 1))

        return max_length if max_length > 1 else -1


# 299. Bulls and Cows
# https://leetcode.com/problems/bulls-and-cows/description/


class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        secret_freq = [0] * 10
        guess_freq = [0] * 10
        bulls = 0
        n = len(secret)

        for i in range(n):
            if secret[i] == guess[i]:
                bulls += 1
            else:
                secret_freq[int(secret[i])] += 1
                guess_freq[int(guess[i])] += 1
        cows = sum(min(secret_freq[i], guess_freq[i]) for i in range(10))

        return f"{bulls}A{cows}B"


# numbers in this set 264, 267, 271, 274, 275, 276, 277, 279, 280, 284, 285, 286, 288, 289, 291,294, 299
