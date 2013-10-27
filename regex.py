
from automata import *
import time

OR      = ','
CLOSURE = '*'
POS_CLOSURE = '+'
SYMBOLS = (')', '(', OR, CLOSURE, POS_CLOSURE)

def balanced_parenthesis(txt):
    count = 0
    for c in txt:
        if c == '(': count += 1
        if c == ')': count -= 1
        if count < 0: return False
    return count == 0


class RegularExpression:

    def __init__(self, regex_str):
        if not balanced_parenthesis(regex_str):
            raise Exception("Parenthesis not balanced.")
        self.regex = '(' + regex_str + ')'
        self.nfa = None
        self.dfa = DFA.from_nfa(self.__get_nfa())
        self.dfa.minimize()

    def get_dfa(self):
        return self.dfa

    def __get_nfa(self):
        alphabet = set(c for c in self.regex if c not in SYMBOLS)
        nfa = NFA(alphabet)
        nfa.set_initial(0)
        nfa.add_final(len(self.regex))
        stack = list()
        N = len(self.regex)

        for i, c in enumerate(self.regex):
            ind = i
            if c in alphabet:
                nfa.add_transition(i, i + 1, c)
            elif c == '(':
                nfa.add_transition(i, i + 1, EPSILON)
                stack.append(i)
            elif c == ')':
                nfa.add_transition(i, i + 1, EPSILON)
                ind = stack.pop()
                tmplist = list()
                # Adds a transition between every or and the closing parenthesis
                while self.regex[ind] == OR:
                    tmplist.append(ind)
                    nfa.add_transition(ind, i, EPSILON)
                    ind = stack.pop()
                # Adds a transition between the opening parenthesis and every or
                for n in tmplist:
                    nfa.add_transition(ind, n + 1, EPSILON)                    
            elif c == OR:
                stack.append(i)
            elif c in (CLOSURE, POS_CLOSURE):
                nfa.add_transition(i, i + 1, EPSILON)
            if i < N - 1 and self.regex[i + 1] in (CLOSURE, POS_CLOSURE):
                if self.regex[i + 1] == CLOSURE: 
                    nfa.add_transition(ind, i + 1, EPSILON)
                nfa.add_transition(i + 1, ind, EPSILON)      
        return nfa

    def __str__(self):
        return self.regex

    def matches(self, text):
        state = self.dfa.initial_state
        for i, letter in enumerate(text):
            try:
                state = self.dfa.get_transition(state, letter)
            except SymbolNotInAlphabetError:
                return (False, i)
        result = any(map(lambda s: s in state, (f for f in self.dfa.final_states)))
        return (result, len(text))

    def search(self, text):
        current_states = list()
        result = list()
        for i, c in enumerate(text):
            current_states.append((i, {self.dfa.initial_state}))
            new_states = list()
            for initial, s in current_states:
                try:
                    t = self.dfa.get_transition(s, c)
                    if not t: continue
                    if self.dfa.contains_final(t):
                        result.append((initial, i))
                    new_states.append((initial, t))
                except SymbolNotInAlphabetError:
                    pass
            current_states = new_states
        return result


if __name__ == '__main__':
    regex = "a*"
    r = RegularExpression(regex)
    t0 = time.time()
    r.search("a" * 1000)
    print time.time() - t0
    with open("res.html", "w") as f:
        f.write(r.dfa.get_transition_html())
