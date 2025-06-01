import re
from typing import List, Optional
from collections import Counter


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 390. Elimination Game
# https://leetcode.com/problems/elimination-game/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def lastRemaining(self, n: int) -> int:
        return 1 if n == 1 else 2 * (n // 2 + 1 - self.lastRemaining(n // 2))


# 486. Predict the Winner
# https://leetcode.com/problems/predict-the-winner/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def predictTheWinner(self, nums: List[int]) -> bool:
        memo = {}

        def recursive(start, end):
            if (start, end) in memo:
                return memo[(start, end)]
            if start == end:
                return nums[start]

            pick_start = nums[start] - recursive(start + 1, end)
            pick_end = nums[end] - recursive(start, end - 1)

            memo[(start, end)] = max(pick_start, pick_end)
            return memo[(start, end)]

        return recursive(0, len(nums) - 1) >= 0


# 779. K-th Symbol in Grammar
# https://leetcode.com/problems/k-th-symbol-in-grammar/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def kthGrammar(self, n: int, k: int) -> int:
        current = 0

        left = 1
        right = 2 ** (n - 1)

        for _ in range(n - 1):
            mid = (left + right) // 2
            if k <= mid:
                right = mid
            else:
                left = mid + 1
                current = 1 if current == 0 else 0
        return current


# 894. All Possible Full Binary Trees
# https://leetcode.com/problems/all-possible-full-binary-trees/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        if n % 2 == 0:
            return []
        if n == 1:
            return [TreeNode()]

        result = []
        for left_nodes in range(1, n, 2):
            right_nodes = n - 1 - left_nodes
            for left in self.allPossibleFBT(left_nodes):
                for right in self.allPossibleFBT(right_nodes):
                    root = TreeNode()
                    root.left = left
                    root.right = right
                    result.append(root)

        return result


# 1545. Find Kth Bit in Nth Binary String
# https://leetcode.com/problems/find-kth-bit-in-nth-binary-string/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        def reverse_str(s):
            return s[::-1]

        def invert_str(s):
            result = ""
            for char in s:
                result += str(1 - int(char))
            return result

        current = "0"
        for i in range(1, n):
            new = current + "1" + reverse_str(invert_str(current))
            current = new

        return current[k - 1]


# 1922. Count Good Numbers
# https://leetcode.com/problems/count-good-numbers/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        MOD = 10**9 + 7

        def pow(x, n):
            if n == 0:
                return 1
            result = 1
            while n > 0:
                if n % 2:
                    result = (result * x) % MOD
                n = n // 2
                x = (x * x) % MOD
            return result

        even = ceil(n / 2)
        odd = n // 2

        return (pow(5, even) * pow(4, odd)) % MOD


# 1969. Minimum Non-Zero Product of the Array Elements
# https://leetcode.com/problems/minimum-non-zero-product-of-the-array-elements/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def minNonZeroProduct(self, p: int) -> int:
        MOD = 10**9 + 7
        top = pow(2, p, MOD) - 1
        mid = top - 1
        midcount = pow(2, p - 1) - 1

        return (pow(mid, midcount, MOD) * top) % MOD


# 2550. Count Collisions of Monkeys on a Polygon
# https://leetcode.com/problems/count-collisions-of-monkeys-on-a-polygon/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def monkeyMove(self, n: int) -> int:
        MOD = 10**9 + 7
        return (pow(2, n, MOD) - 2) % MOD


# 233. Number of Digit One
# https://leetcode.com/problems/number-of-digit-one/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def countDigitOne(self, n: int) -> int:
        s = str(n)

        @cache
        def recursive(i, count, is_limit, is_num):
            if i == len(s):
                return count
            result = 0
            if not is_num:
                result = recursive(i + 1, count, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9

            for d in range(low, high + 1):
                result += recursive(i + 1, count + (d == 1), is_limit and d == high, True)

            return result

        return recursive(0, 0, True, False)


# 273. Integer to English Words
# https://leetcode.com/problems/integer-to-english-words/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        ones_map = {
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine",
            10: "Ten",
            11: "Eleven",
            12: "Twelve",
            13: "Thirteen",
            14: "Fourteen",
            15: "Fifteen",
            16: "Sixteen",
            17: "Seventeen",
            18: "Eighteen",
            19: "Nineteen",
        }
        tens_map = {
            20: "Twenty",
            30: "Thirty",
            40: "Forty",
            50: "Fifty",
            60: "Sixty",
            70: "Seventy",
            80: "Eighty",
            90: "Ninety",
        }

        def get_string(n):
            result = []
            hundreds = n // 100
            if hundreds:
                result.append(ones_map[hundreds] + " Hundred")
            last_2 = n % 100
            if last_2 >= 20:
                tens, ones = last_2 // 10, last_2 % 10
                result.append(tens_map[tens * 10])
                if ones:
                    result.append(ones_map[ones])
            elif last_2:
                result.append(ones_map[last_2])
            return " ".join(result)

        postfix = ["", " Thousand", " Million", " Billion"]
        i = 0
        result = []

        while num:
            digits = num % 1000
            s = get_string(digits)
            if s:
                result.append(s + postfix[i])
            num = num // 1000

            i += 1

        result.reverse()

        return " ".join(result)


# 736. Parse Lisp Expression
# https://leetcode.com/problems/parse-lisp-expression/description/?envType=problem-list-v2&envId=recursion


class Solution:
    def evaluate(self, expression: str) -> int:
        stack = []

        parenthesisEnd = {}
        for index, char in enumerate(expression):
            if char == "(":
                stack.append(index)
            elif char == ")":
                parenthesisEnd[stack.pop()] = index

        def parse(low, high):
            arr = []
            word = []
            i = low

            while i < high:
                if expression[i] == "(":
                    arr.append(parse(i + 1, parenthesisEnd[i]))
                    i = parenthesisEnd[i]
                elif expression[i] == " " or expression[i] == ")" and word != []:
                    if "".join(word) != "":
                        arr.append("".join(word))
                    word = []
                    i += 1
                elif expression[i] != ")":
                    word.append(expression[i])
                    i += 1
                else:
                    i += 1
            if word != []:
                arr.append("".join(word))
            return arr

        expressionlist = parse(1, len(expression) - 1)
        return self.genEval(expressionlist, {})

    def genEval(self, expression, scope):
        if type(expression) != list:
            try:
                return int(expression)
            except:
                return scope[expression]
        else:
            if expression[0] == "let":
                expression = expression[1:]
                while len(expression) > 2:
                    scope = self.letEval(expression, scope.copy())
                    expression = expression[2:]
                return self.genEval(expression[0], scope.copy())
            elif expression[0] == "add":
                return self.addEval(expression, scope.copy())
            elif expression[0] == "mult":
                return self.multEval(expression, scope.copy())

    def letEval(self, expression, scope):
        scope[expression[0]] = self.genEval(expression[1], scope)
        return scope

    def addEval(self, expression, scope):
        return self.genEval(expression[1], scope) + self.genEval(expression[2], scope)

    def multEval(self, expression, scope):
        return self.genEval(expression[1], scope) * self.genEval(expression[2], scope)


# 761. Special Binary String
# https://leetcode.com/problems/special-binary-string/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def makeLargestSpecial(self, s: str) -> str:
        n = len(s)

        if n <= 2:
            return s

        subs = []
        i = 0
        count = 0

        for j, char in enumerate(s):
            if char == "1":
                count += 1
            else:
                count -= 1

            if count == 0:
                inner = self.makeLargestSpecial(s[i + 1 : j])
                subs.append("1" + inner + "0")
                i = j + 1

        subs.sort(reverse=True)

        return "".join(subs)


# 770. Basic Calculator IV
# https://leetcode.com/problems/basic-calculator-iv/description/?envType=problem-list-v2&envId=recursion


class Solution:
    def basicCalculatorIV(self, expression: str, evalvars: List[str], evalints: List[int]) -> List[str]:
        var_map = dict(zip(evalvars, evalints))
        tokens = re.findall(r"[a-z]+|\d+|[-+*()]", expression)

        def parse_expression(tokens):
            def parse_term():
                token = tokens.pop(0)
                if token == "(":
                    res = parse_expression(tokens)
                    tokens.pop(0)  # pop )
                elif token.isdigit():
                    return Counter({(): int(token)})
                elif token.isalpha():
                    if token in var_map:
                        return {(): var_map[token]}
                    else:
                        return Counter({(token,): 1})

            def parse_product():
                res = parse_term()
                while tokens and tokens[0] == "*":
                    tokens.pop(0)
                    right = parse_term()
                    new_res = Counter()
                    for vars1, coeff1 in res.items():
                        for vars2, coeff2 in right.items():
                            new_vars = tuple(sorted(vars1 + vars2))
                            new_res[new_vars] += coeff1 * coeff2
                    res = new_res

                return res

            res = parse_product()
            while tokens and tokens[0] in "+-":
                op = tokens.pop(0)
                right = parse_product()
                if op == "+":
                    res += right
                else:
                    res -= right
            return res

        expr = parse_expression(tokens)

        expr = +expr

        def sort_key(item):
            vars, coeff = item
            return (-len(vars), vars)

        return [f"{coeff}{'*'.join(vars) if vars else ''}" for vars, coeff in sorted(expr.items(), key=sort_key)]

        print(tokens)
        return None


# 1106. Parsing A Boolean Expression
# https://leetcode.com/problems/parsing-a-boolean-expression/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def parseBoolExpr(self, expression: str) -> bool:
        def to_bool(char):
            return char == "t"

        stack = []

        for char in expression:
            if char == ",":
                continue
            elif char != ")":
                stack.append(char)
            else:
                vals = []
                while stack[-1] != "(":
                    vals.append(stack.pop())
                stack.pop()

                op = stack.pop()

                if op == "!":
                    result = not to_bool(vals[0])
                elif op == "|":
                    result = any(to_bool(value) for value in vals)
                elif op == "&":
                    result = all(to_bool(value) for value in vals)
                stack.append("t" if result else "f")
        return to_bool(stack.pop())


# 1808. Maximize Number of Nice Divisors
# https://leetcode.com/problems/maximize-number-of-nice-divisors/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def maxNiceDivisors(self, primeFactors: int) -> int:
        MOD = 10**9 + 7

        def power(x):
            if x == 0:
                return 1
            ans = power(x // 2)

            if x % 2 == 1:
                return (ans * ans * 3) % MOD
            return (ans * ans) % MOD

        if primeFactors == 1:
            return 1
        ans = 1
        if primeFactors % 3 == 0:
            ans = power(primeFactors // 3)
        elif primeFactors % 3 == 1:
            ans = (4 * power((primeFactors // 3) - 1)) % MOD
        elif primeFactors % 3 == 2:
            ans = (2 * power((primeFactors // 3))) % MOD
        return ans


# 3307. Find the K-th Character in String Game II
# https://leetcode.com/problems/find-the-k-th-character-in-string-game-ii/description/?envType=problem-list-v2&envId=recursion
class Solution:
    def kthCharacter(self, k: int, operations: List[int]) -> str:
        k -= 1

        def go(k, index):
            if index == 0:
                return 0
            first_half = pow(2, index - 1)
            if k >= first_half:
                if operations[index - 1] == 0:
                    return go(k - first_half, index - 1)
                else:
                    return go(k - first_half, index - 1) + 1
            else:
                return go(k, index - 1)

        return chr((go(k, len(operations)) % 26) + ord("a"))


# numbers in this set 390, 486, 779, 894, 1545, 1922, 1969, 2550, 233,273, 736, 761, 770, 1106, 1808,3307
