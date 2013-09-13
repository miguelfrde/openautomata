# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import wraps

EPSILON = '€'


class SymbolNotInAlphabetError(Exception):

    def __init__(self, symbol, alphabet):
        message = "The symbol %s doesn't belong to the alphabet %s" % \
            (symbol, alphabet)
        Exception.__init__(self, message)


def check_alphabet(f):
    @wraps(f)
    def _f(self, *args, **kwargs):
        symbol = args[-1]
        if symbol not in self.alphabet:
            raise SymbolNotInAlphabetError(symbol, self.alphabet)
        return f(self, *args, **kwargs)
    return _f


class Automata:

    def __init__(self, alphabet):
        self.transition = defaultdict(set)
        self.states = set()
        self.final_states = set()
        self.initial_state = None
        self.alphabet = set(alphabet)

    def set_initial(self, state):
        "Sets the initial state of the automata"
        assert isinstance(state, int)
        self.initial_state = state
        self.states.add(state)

    def add_final(self, state):
        "Add a state and make it final or set a state to be final"
        assert isinstance(state, int)
        self.final_states.add(state)
        self.states.add(state)

    def add_finals(self, states):
        "Makes every state in states final"
        for state in states:
            self.add_final(state)

    @check_alphabet
    def add_transition(self, s1, s2, symbol):
        "Adds a transition from s1 to s2 under symbol"
        self.states = self.states.union({s1, s2})
        self.transition[s1, symbol].add(s2)

    def add_transitions(self, transitions):
        """ Adds all transitions. transitions = {s1: (s2, a) ... }"""
        for s1, a in transitions:
            self.add_transition(s1, transitions[s1, a], a)

    def get_transition(self, state, symbol):
        """ Returns the state reached after applying symbol on state.
            Method used by NFA, DFA and other child classes. """
        raise NotImplemented

    def is_final(self, state):
        "Checks wether an state is final"
        return state in self.final_states

    def is_initial(self, state):
        "Checks wether an state is initial"
        return state == self.initial_state


class DFA(Automata):

    @check_alphabet
    def get_transition(self, state, symbol):
        "Returns the transition D(set_state_or_state, symbol)"
        # state = [s0, s1, s2, ...] or s0
        if isinstance(state, int):
            return self.transition[state, symbol]
        return reduce(set.union, [self.transition[s, symbol] for s in state])


class NFA(Automata):

    def __init__(self, alphabet):
        Automata.__init__(self, alphabet)
        self.alphabet.add(EPSILON)

    @check_alphabet
    def single_transition(self, state, symbol):
        "Returns the extended transition D(state, symbol)"
        assert isinstance(state, int) and state in self.states
        eclosure = self.epsilon_closure(state)
        r = [map(self.epsilon_closure, self.transition[s, symbol])
             for s in eclosure]
        r = sum(r, list())
        if len(r) == 0:
            return set()
        return reduce(set.union, r)

    def get_transition(self, state, symbol):
        "Returns the extended transition D(set_state_or_state, symbol)"
        # state = [s0, s1, s2, ...] or s0
        if isinstance(state, int):
            return self.single_transition(state, symbol)
        if len(state) == 1:
            return self.single_transition(list(state)[0], symbol)
        r = [self.single_transition(s, symbol)for s in state]
        if len(r) == 0:
            return set()
        return reduce(set.union, r)

    def epsilon_closure(self, state, result=set()):
        "Returns the Epsilon-closure or Lambda-closure of state."
        assert isinstance(state, int) and state in self.states
        result = result.union(self.transition[state, EPSILON] | {state})
        for s in result:
            for s2 in self.transition[s, EPSILON]:
                if s2 not in result:
                    result = self.epsilon_closure(s2, result)
        return result


# Show examples from class
if __name__ == '__main__':
    # (abb)*(ab)?
    nfa = NFA(['a', 'b'])
    nfa.set_initial(0)
    nfa.add_final(2)
    nfa.add_transition(0, 1, EPSILON)
    nfa.add_transition(0, 3, 'a')
    nfa.add_transition(1, 2, EPSILON)
    nfa.add_transition(3, 4, 'b')
    nfa.add_transition(4, 0, 'b')
    nfa.add_transition(4, 1, EPSILON)
    closures = [nfa.epsilon_closure(s) for s in xrange(5)]
    for c in closures:
        print c, nfa.get_transition(c, 'a'), nfa.get_transition(c, 'b')
    # (\+|-)?[0-9]*(\.[0-9]|[0-9]\.)[0-9]*
    nfa = NFA(['+', '-', '.'] + map(str, range(10)))
    nfa.set_initial(0)
    nfa.add_final(5)
    nfa.add_transition(0, 1, EPSILON)
    nfa.add_transition(0, 1, '+')
    nfa.add_transition(0, 1, '-')
    for n in xrange(10):
        nfa.add_transition(1, 1, str(n))
        nfa.add_transition(1, 4, str(n))
        nfa.add_transition(2, 3, str(n))
        nfa.add_transition(3, 3, str(n))
    nfa.add_transition(1, 2, '.')
    nfa.add_transition(4, 3, '.')
    nfa.add_transition(3, 5, EPSILON)
    closures = [nfa.epsilon_closure(s) for s in xrange(6)]
    for c in closures:
        print c, nfa.get_transition(c, '+'), nfa.get_transition(c, '0'), nfa.get_transition(c, '.')
        assert all([nfa.get_transition(c, '0') == nfa.get_transition(c, str(n))
                   for n in xrange(10)])
        assert nfa.get_transition(c, '+') == nfa.get_transition(c, '-')
    c = {1, 4}
