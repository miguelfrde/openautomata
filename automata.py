# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import wraps
from jinja2 import Environment, FileSystemLoader

EPSILON = '&'
OR      = ','
CLOSURE = '*'
POS_CLOSURE = '+'
SYMBOLS = (')', '(', OR, CLOSURE, POS_CLOSURE)


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

    def get_transition_html(self):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('transition_table.html')
        return template.render( alphabet = sorted(self.alphabet),
                                states   = sorted(self.states),
                                transition = self.transition)

    def contains_final(self, state):
        return any(map(lambda s: s in state, self.final_states))


class DFA(Automata):

    @check_alphabet
    def get_transition(self, state, symbol):
        "Returns the transition D(set_state_or_state, symbol)"
        # state = [s0, s1, s2, ...] or s0
        if isinstance(state, int):
            return self.transition[state, symbol]
        return reduce(set.union, [self.transition[s, symbol] for s in state])

    @classmethod
    def from_nfa(cls, nfa):
        dfa = DFA(nfa.alphabet - {EPSILON})
        initial = frozenset(nfa.epsilon_closure(0))
        states = {initial: 0}
        to_visit = [initial]
        if nfa.contains_final(initial):
            dfa.add_final(states[initial])
        next_index = 0
        dfa.set_initial(0)
        while to_visit:
            state = to_visit.pop(0)
            for symbol in dfa.alphabet:
                next = frozenset(nfa.get_transition(state, symbol))
                if next:
                    if next not in states:
                        next_index += 1
                        states[next] = next_index
                        to_visit.append(next)
                    if nfa.contains_final(next):
                        dfa.add_final(next_index)
                    dfa.add_transition(states[state], states[next], symbol)
        return dfa


    @classmethod
    def minimize(self):
        pass


class NFA(Automata):

    def __init__(self, alphabet, accept_void=True):
        Automata.__init__(self, alphabet)
        if accept_void: self.alphabet.add(EPSILON)

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
        r = [self.single_transition(s, symbol) for s in state]
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

    def get_nfa_without_void_transitions(self):
        nfa = NFA(self.alphabet - {EPSILON}, accept_void=False)
        nfa.set_initial(self.initial_state)
        for state in self.states:
            if self.is_final(state):
                nfa.add_final(state)
            map(lambda (s, a): nfa.add_transition(state, s, a),
                ((state2, symbol)
                for symbol in nfa.alphabet
                for state2 in self.get_transition(state, symbol)))
        return nfa


if __name__ == '__main__':
    nfa = NFA({'a', 'b', 'c'})
    nfa.set_initial(0)
    nfa.add_transition(0, 1, EPSILON)
    nfa.add_transition(0, 2, EPSILON)
    nfa.add_transition(0, 1, 'b')
    nfa.add_transition(0, 2, 'c')
    nfa.add_transition(1, 0, 'a')
    nfa.add_transition(1, 2, 'b')
    nfa.add_transition(1, 0, 'c')
    nfa.add_transition(1, 1, 'c')
    with open('res1.html', 'w') as f:
        f.write(nfa.get_transition_html())
    dfa = DFA.from_nfa(nfa)
    with open('res2.html', 'w') as f:
        f.write(dfa.get_transition_html())
    nfa = nfa.get_nfa_without_void_transitions()
    with open('res3.html', 'w') as f:
        f.write(nfa.get_transition_html())

