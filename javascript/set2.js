//  * Definition for singly-linked list.
function ListNode(val, next) {
  this.val = val === undefined ? 0 : val;
  this.next = next === undefined ? null : next;
}

// # 22. Generate Parentheses
// # https://leetcode.com/problems/generate-parentheses/description/

/**
 * @param {number} n
 * @return {string[]}
 */
const generateParenthesis = (n) => {
  const result = [];

  const generate = (open, close, current) => {
    if (open === 0 && close === 0) {
      result.push(current);
      return;
    }
    if (open > 0) {
      generate(open - 1, close, current + "(");
    }
    if (open < close) {
      generate(open, close - 1, current + ")");
    }
  };

  generate(n, n, "");

  return result;
};

/**
 * @param {ListNode} head
 * @return {ListNode}
 */

// 24. Swap Nodes in Pairs
// https://leetcode.com/problems/swap-nodes-in-pairs/description/

const swapPairs = function (head) {
  const dummy = new ListNode(0, head);
  const prev = dummy;
  const curr = head;

  while (curr && curr.next) {
    // save temp pointers
    const temp = curr;
    const nextPointer = curr.next.next;

    // swap nodes
    curr = curr.next;
    curr.next = temp;

    // connect the nodes back to the list
    temp.next = nextPointer;
    prev.next = curr;

    // move the pointers
    prev = temp;
    curr = nextPointer;
  }

  return dummy.head;
};

// # numbers in this set 22, 24
