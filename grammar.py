
from collections import defaultdict


class NotValidSymbolError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


def mix(a, b, s=True):
    f = lambda x, y: x + y if s else (x, y)
    return (f(x,y) for x in a for y in b)

class ContextFreeGrammar:

    def __init__(self, alphabet, initial):
        "alphabet = grammar alphabet, initial = initial non terminal"
        self.alphabet = alphabet
        self.initial = initial
        # Maps from a non terminal state to the rest of states
        self.productions = defaultdict(set)

    def add_production(self, left, right):
        "Add the production Left -> Right to the grammar "
        self.productions[left].add(right)

    def contains(self, non_terminal, symbol):
        "Check if the production Non_terminal -> symbol exists"
        return symbol in self.productions[non_terminal]

    def accepts(self, w):
        """CYK algorithm.
           For the moment, assume the grammar is in Chomsky Normal Form"""
        n = len(w)
        N = defaultdict(set)
        for i in xrange(1, n + 1):
            N[i, 1] = {A for A in self.productions if self.contains(A, w[i-1])}
        for j in xrange(1, n + 1):
            for i in xrange(1, n - j + 2):
                for k in xrange(1, j):
                    B = N[i, k]
                    C = N[i + k, j - k]
                    for s in mix(B, C):
                        N[i, j] |= {A for A in self.productions if self.contains(A, s)}
        return self.initial in N[1, n]


    def __str__(self):
        to_or = lambda s: ' | '.join(self.productions[s])
        format = lambda s: '%s -> %s' % (s, to_or(s))
        s = '%s\n' % format(self.initial)
        return s + '\n'.join(format(p) for p in self.productions if p != self.initial)

if __name__ == "__main__":
    cfg = ContextFreeGrammar({'a', 'b', 'c'}, 'S')
    cfg.add_production('S', 'AB')
    cfg.add_production('S', 'BC')
    cfg.add_production('A', 'BA')
    cfg.add_production('A', 'a')
    cfg.add_production('B', 'CC')
    cfg.add_production('B', 'b')
    cfg.add_production('C', 'AB')
    cfg.add_production('C', 'a')
    print cfg
    print cfg.accepts('abaa')
