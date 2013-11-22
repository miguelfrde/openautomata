



    # def get_nfa_without_void_transitions(self):
    #     "Return an NFA without void transitions"
    #     nfa = NFA(self.alphabet - {EPSILON}, accept_void=False)
    #     nfa.set_initial(self.initial_state)
    #     for state in self.states:
    #         if self.is_final(state):
    #             nfa.add_final(state)
    #         map(lambda (s, a): nfa.add_transition(state, s, a),
    #             ((state2, symbol)
    #             for symbol in nfa.alphabet
    #             for state2 in self.get_transition(state, symbol)))
    #     return nfa

OR      = ','
CLOSURE = '*'
POS_CLOSURE = '+'
CONCAT = '#'
SYMBOLS = (')', '(', OR, CLOSURE, POS_CLOSURE)
IGNORE_CONCAT = ('(', OR, CLOSURE, POS_CLOSURE)
OPERATORS = (OR, CLOSURE, POS_CLOSURE, CONCAT)

precedence = {
    CLOSURE: 3,
    POS_CLOSURE: 3,
    CONCAT: 2,
    OR: 1,
    '(': 0
}

def introduce_concatenation(regex):
    is_ok_to_concat = lambda c: c not in SYMBOLS or c == '('
    g = lambda x, y: x not in (OR, '(') and is_ok_to_concat(y)
    f = lambda x, y: x + (CONCAT if g(x[-1], y) else '') + y
    return reduce(f, regex)

def convert(infix):
    infix = introduce_concatenation(infix)
    postfix = ""
    stack = list()
    for c in infix:
        if c in OPERATORS:
            while len(stack) != 0 and stack[-1] in OPERATORS and \
                    precedence[c] - precedence[stack[-1]] <= 0:
                postfix += stack.pop()
            stack.append(c)
        elif c == ')':
            while stack[-1] != '(':
                if len(stack) == 0: raise Exception("Mismatched parentheses")
                postfix += stack.pop()
            stack.pop()
        elif c == '(':
            stack.append(c)
        else:
            postfix += c
    while len(stack) != 0:
        if stack[-1] == ')' or stack[-1] == '(':
            raise Exception("Mismatched parentheses")
        postfix += stack.pop()
    return ''.join(postfix)

def postToIn(postfix):
    stack = list()
    infix = ""
    for c in postfix:
        if c == '|':
            pass
print introduce_concatenation("a*|b")

t = ["ab", "a*b", "a,b", "a,bc,d", "ab*", "a,b*", "(a,b)(c,d)", "(ab)*", "(a,b)*", "(a,b)(c,d)", "((((ab),(cd))*)(e,f))"]
for x in t:
    print x, convert(x)