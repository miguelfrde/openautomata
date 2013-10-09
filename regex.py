
from automata import NFA


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
        self.nfa = self.get_nfa()

    def get_nfa(self):
        if self.nfa: return self.nfa
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
        state = self.nfa.initial_state
        for i, letter in enumerate(text):
            try:
                state = self.nfa.get_transition(state, letter)
            except SymbolNotInAlphabetError:
                return (False, i)
        result = any(map(lambda s: s in state, (f for f in self.nfa.final_states)))
        return (result, len(text))

    def search(self, text):
        i = 0
        result = list()
        for i in xrange(len(text)):
            state = self.nfa.epsilon_closure(self.nfa.initial_state)
            offset = 0
            while True:
                try:
                    state = self.nfa.get_transition(state, text[i + offset])
                    if self.nfa.contains_final(state):
                        result.append((i, i+offset, text[i: i + offset + 1]))
                    offset += 1
                except (SymbolNotInAlphabetError, IndexError) as e:
                    break
        return result


if __name__ == '__main__':
	pass
