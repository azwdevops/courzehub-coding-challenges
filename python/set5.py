from copy import deepcopy
from typing import Optional, List
from collections import deque, defaultdict


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: "Node" = None, right: "Node" = None, next: "Node" = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


# 98. Validate Binary Search Tree
# https://leetcode.com/problems/validate-binary-search-tree/description/


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def recursion(node, lower, upper):
            if not node:
                return True

            if not (lower < node.val < upper):
                return False

            return recursion(node.left, lower, node.val) and recursion(node.right, node.val, upper)

        return recursion(root, float("-inf"), float("inf"))


# 99. Recover Binary Search Tree
# https://leetcode.com/problems/recover-binary-search-tree/description/


class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """

        first = second = prev = None

        def inorder(node):
            nonlocal first, second, prev

            if not node:
                return
            inorder(node.left)

            if prev and node.val < prev.val:
                if not first:
                    first = prev
                second = node

            prev = node
            inorder(node.right)

        inorder(root)

        if first and second:
            first.val, second.val = second.val, first.val


# 102. Binary Tree Level Order Traversal
# https://leetcode.com/problems/binary-tree-level-order-traversal/description/


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []

        current = deque([root])

        while current:
            next_level = deque([])
            temp = []
            n = len(current)
            for _ in range(n):
                node = current.popleft()
                temp.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            current = next_level
            result.append(temp[:])

        return result


# 103. Binary Tree Zigzag Level Order Traversal
# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/


class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        right = False

        current = deque([root])
        result = []

        while current:
            n = len(current)
            next_level = deque([])
            temp = []
            for _ in range(n):
                node = current.pop()
                temp.append(node.val)
                if right:
                    if node.right:
                        next_level.append(node.right)
                    if node.left:
                        next_level.append(node.left)
                else:
                    if node.left:
                        next_level.append(node.left)
                    if node.right:
                        next_level.append(node.right)
            right = not right
            current = next_level
            result.append(temp)

        return result


# 105. Construct Binary Tree from Preorder and Inorder Traversal
# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        inMap = defaultdict(int)

        for i in range(len(inorder)):
            inMap[inorder[i]] = i

        def recursive(preStart, preEnd, inStart, inEnd):
            if preStart > preEnd or inStart > inEnd:
                return None
            node = TreeNode(preorder[preStart])

            inRoot = inMap[node.val]
            numsLeft = inRoot - inStart

            node.left = recursive(preStart + 1, preStart + numsLeft, inStart, inRoot - 1)
            node.right = recursive(preStart + numsLeft + 1, preEnd, inRoot + 1, inEnd)

            return node

        root = recursive(0, len(preorder) - 1, 0, len(inorder) - 1)

        return root


# 106. Construct Binary Tree from Inorder and Postorder Traversal
# https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/


class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if len(inorder) != len(postorder):
            return None

        inMap = defaultdict(int)
        for i in range(len(inorder)):
            inMap[inorder[i]] = i

        def recursive(inStart, inEnd, postStart, postEnd):
            if postStart > postEnd or inStart > inEnd:
                return None

            node = TreeNode(postorder[postEnd])

            inRoot = inMap[postorder[postEnd]]
            numsLeft = inRoot - inStart

            node.left = recursive(inStart, inRoot - 1, postStart, postStart + numsLeft - 1)
            node.right = recursive(inRoot + 1, inEnd, postStart + numsLeft, postEnd - 1)

            return node

        return recursive(0, len(inorder) - 1, 0, len(postorder) - 1)


# 107. Binary Tree Level Order Traversal II
# https://leetcode.com/problems/binary-tree-level-order-traversal-ii/description/


class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        result = deque([])
        current = deque([root])

        while current:
            next_level = []
            temp = []
            n = len(current)
            for _ in range(n):
                node = current.popleft()
                temp.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            result.appendleft(temp)
            current = deque(next_level)

        return list(result)


# 109. Convert Sorted List to Binary Search Tree
# https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/


class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        n = 0
        current = head
        while current:
            current = current.next
            n += 1
        self.head = head

        def recursive(start, end):
            if start > end:
                return None
            mid = (start + end) // 2

            left = recursive(start, mid - 1)

            node = TreeNode(self.head.val)

            self.head = self.head.next
            node.left = left
            node.right = recursive(mid + 1, end)

            return node

        return recursive(0, n - 1)


# 113. Path Sum II
# https://leetcode.com/problems/path-sum-ii/description/


class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if not root:
            return []
        result = []

        def recursive(current, target, node):
            if not node:
                return

            if target + node.val == targetSum and not node.left and not node.right:
                result.append(current + [node.val])
                return

            recursive(current + [node.val], target + node.val, node.left)
            recursive(current + [node.val], target + node.val, node.right)

        recursive([], 0, root)

        return result


# 114. Flatten Binary Tree to Linked List
# https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        prev = None

        def recursive(node):
            nonlocal prev
            if not node:
                return
            recursive(node.right)
            recursive(node.left)

            node.right = prev
            node.left = None
            prev = node

        recursive(root)


# 116. Populating Next Right Pointers in Each Node
# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/


class Solution:
    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return None
        queue = deque([root])

        while queue:
            next_level = []
            for i in range(len(queue)):
                node = queue.popleft()
                if len(queue) > 0:
                    node.next = queue[0]

                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            queue = deque(next_level)

        return root


# 117. Populating Next Right Pointers in Each Node II
# https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/


class Solution:
    def connect(self, root: "Node") -> "Node":
        if not root:
            return None
        queue = deque([root])

        while queue:
            next_level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                if len(queue) > 0:
                    node.next = queue[0]
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            queue = deque(next_level)

        return root


# 120. Triangle
# https://leetcode.com/problems/triangle/description/


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = deepcopy(triangle)

        for i in range(1, n):
            m = len(triangle[i])
            for j in range(m):
                if j == 0:
                    dp[i][j] += dp[i - 1][j]
                elif j == m - 1:
                    dp[i][j] += dp[i - 1][-1]
                else:
                    dp[i][j] += min(dp[i - 1][j], dp[i - 1][j - 1])

        return min(dp[-1])


# 122. Best Time to Buy and Sell Stock II
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy = prices[0]
        max_profit = 0

        for price in prices:
            if price - buy > 0:
                max_profit += price - buy

            buy = price

        return max_profit


# 128. Longest Consecutive Sequence
# https://leetcode.com/problems/longest-consecutive-sequence/description/


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        nums_set = set(nums)
        longest = 1

        for item in nums_set:
            if item - 1 in nums_set:
                continue
            new_item = item
            count = 1
            while new_item + 1 in nums_set:
                count += 1
                new_item += 1
            longest = max(longest, count)

        return longest


# 129. Sum Root to Leaf Numbers
# https://leetcode.com/problems/sum-root-to-leaf-numbers/description/


class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        total = 0

        def recursive(node, current):
            if not node.left and not node.right:
                current += str(node.val)
                nonlocal total
                total += int(current)

            if node.left:
                recursive(node.left, current + str(node.val))
            if node.right:
                recursive(node.right, current + str(node.val))

        recursive(root, "")

        return total


# 130. Surrounded Regions
# https://leetcode.com/problems/surrounded-regions/description/


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])
        visited = [[0] * n for _ in range(m)]
        region_cells = []
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        for i in range(m):
            for j in range(n):
                if visited[i][j] != 0 or i == m - 1 or i == 0 or j == 0 or j == n - 1:
                    continue
                if board[i][j] == "O":
                    region_okay = True  # means region is not in the boundary
                    temp = [(i, j)]
                    stack = [(i, j)]
                    while stack:
                        row, col = stack.pop()
                        visited[row][col] = 1
                        for delta_row, delta_col in directions:
                            nrow = row + delta_row
                            ncol = col + delta_col
                            if (
                                0 <= nrow < m
                                and 0 <= ncol < n
                                and visited[nrow][ncol] == 0
                                and board[nrow][ncol] == "O"
                            ):
                                if nrow == 0 or ncol == 0 or nrow == m - 1 or ncol == n - 1:
                                    region_okay = False
                                stack.append((nrow, ncol))
                                temp.append((nrow, ncol))

                    if region_okay:
                        region_cells.extend(temp)
        for row, col in region_cells:
            board[row][col] = "X"


# numbers in this set 98, 99, 102, 103, 105, 106, 107, 109, 113, 114, 116, 117, 122, 128, 129, 130
