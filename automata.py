# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import wraps
from jinja2 import Environment, FileSystemLoader
import copy

EPSILON = '&'

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
        if isinstance(state, set) and len(state) == 1:
            return list(state)[0] in self.final_states
        return state in self.final_states

    def is_initial(self, state):
        "Checks wether an state is initial"
        return state == self.initial_state

    def get_transition_html(self):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('transition_table.html')
        return template.render( alphabet = sorted(self.alphabet),
                                states   = sorted(self.states),
                                transition = self.transition,
                                final_states = self.final_states,
                                initial = self.initial_state)

    def contains_final(self, state):
        return any(map(lambda s: s in state, self.final_states))

    def contains_initial(self, state):
        return any(map(lambda s: s == self.initial_state, state))

    def has_transition_with(self, state, symbol):
        return (state, symbol) in self.transition


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
                        dfa.add_final(states[next])
                    dfa.add_transition(states[state], states[next], symbol)
        return dfa

    def minimize(self):
        # Remove non reachable states
        non_reachable = self.non_reachable()
        self.states = self.states - non_reachable
        self.final_states = self.final_states - non_reachable
        self.transition = defaultdict(set, [((s, a), self.transition[s, a])
                            for s, a in self.transition if s not in non_reachable])
        # Add error_state
        self.add_error_state()
        # Fill table
        table = {(min(s1, s2), max(s1, s2)): (s1 in self.final_states) != (s2 in self.final_states) 
                            for i, s1 in enumerate(self.states)
                            for j, s2 in enumerate(self.states)
                            if i < j}
        distinguishable = lambda x, y: table[min(x,y), max(x,y)]
        changed = True
        while changed:
            changed = False
            for p, q in table:
                if table[p, q]: continue
                for a in self.alphabet:
                    r, s = self.transition[p, a], self.transition[q, a]
                    r, s = list(r)[0], list(s)[0]
                    if r == s: continue
                    if distinguishable(r, s):
                        table[p, q] = changed = True
                        break
        equivalences = {s: {s} for s in self.states}
        # Find equivalent subsets
        for states in filter(lambda x: table[x] == False, table): 
            for s1 in states:
                for s2 in states:
                    equivalences[s1].add(s2)
        states_map = {frozenset(s): min(s) for s in equivalences.values()}
        # Update self.transition, self.states, self.initial_states, etc.
        dfa = DFA(self.alphabet)
        for old_states_set, new_state in states_map.items():
            if self.contains_initial(old_states_set):
                dfa.set_initial(new_state)
            if self.contains_final(old_states_set):
                dfa.add_final(new_state)
            for old_state in old_states_set:
                for a in (a for s, a in self.transition if s == old_state):
                    to = equivalences[self.get_transition(old_state, a).pop()]
                    dfa.add_transition(new_state, states_map[frozenset(to)], a)
        self.states = dfa.states
        self.transition = dfa.transition
        self.initial_state = dfa.initial_state
        self.final_states = dfa.final_states
        self.remove_error_state()

    def non_reachable(self):
        non_visited = self.states.copy()
        transition = copy.deepcopy(self.transition)
        def f(s):
            non_visited.remove(s)
            transitions = filter(lambda (k, a): k == s, transition)
            for k, a in transitions:
                t = transition[k, a].pop()
                if t in non_visited:
                    f(t)
        f(self.initial_state)
        return non_visited

    def add_error_state(self, error_state=-1):
        self.states.add(error_state)
        for state in self.states:
            for symbol in self.alphabet:
                if not self.has_transition_with(state, symbol):
                    self.add_transition(state, error_state, symbol)

    def remove_error_state(self, error_state=-1):
        self.states.remove(error_state)
        transition = defaultdict(set)
        for t in self.transition:
            if -1 in self.transition[t]:
                self.transition[t].remove(-1)
            if t[0] != -1 and self.transition[t]:
                transition[t] = self.transition[t]
        self.transition = transition


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
    dfa = DFA({'0', '1'})
    dfa.set_initial(0)
    dfa.add_final(2)
    dfa.add_transition(0, 1, '0')
    dfa.add_transition(0, 5, '1')
    dfa.add_transition(1, 2, '1')
    dfa.add_transition(1, 6, '0')
    dfa.add_transition(2, 0, '0')
    dfa.add_transition(2, 2, '1')
    dfa.add_transition(3, 2, '0')
    dfa.add_transition(3, 7, '1')
    dfa.add_transition(4, 5, '1')
    dfa.add_transition(4, 7, '0')
    dfa.add_transition(5, 2, '0')
    dfa.add_transition(5, 6, '1')
    dfa.add_transition(6, 4, '1')
    dfa.add_transition(6, 6, '0')
    dfa.add_transition(7, 2, '1')
    dfa.add_transition(7, 6, '0')
    print dfa.non_reachable()
    dfa.minimize()
    with open("res.html", 'w') as f:
        f.write(dfa.get_transition_html())