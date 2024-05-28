#!/usr/bin/env python

"""PyFoma Finite-State Tool."""

import heapq, json, itertools, re as pyre
from collections import deque, defaultdict
from typing import Callable, Dict, Any, Iterable, TextIO
from os import PathLike
from pathlib import Path
from pyfoma.flag import FlagStringFilter, FlagOp
import subprocess
import pickle


def re(*args, **kwargs):
    return FST.re(*args, **kwargs)


def _multichar_matcher(multichar_symbols: Iterable[str]) -> pyre.Pattern:
    """Create matcher for unquoted multichar symbols in lexicons and
    regular expressions."""
    return pyre.compile(
        r"('(?:\\'|[^'])*')|("
        + "|".join(pyre.escape(sym)
                   for sym in multichar_symbols
                   if len(sym) > 1)
        + r")")


def _multichar_replacer(matchobj: pyre.Match):
    """Replace character or quoted string with quoted thing."""
    quoted, sym = matchobj.groups()
    if quoted is not None:
        return quoted
    return "'" + sym.replace("'", r"\'") + "'"


# TODO: Move all algorithm functions to the algorithms module
class FST:

    # region Initialization Methods
    def __init__(self, label:tuple=None, weight=0.0, alphabet=set()):
        """Creates an FST-structure with a single state.

        :param label: create a two-state FST that accepts label
        :param weight: add a weight to the final state
        :param alphabet: declare an alphabet explicitly

        If 'label' is given, a two-state automaton is created with label as the
        only transition from the initial state to the final state.

        If 'weight' is also given, the final state will have that weight.
        Labels are always tuples internally, so a two-state automaton
        that only accepts 'a' should have label = ('a',).

        If label is the empty string, i.e. ('',), the second state will not be
        created, but the initial state will be made final with weight 'weight'.
        """

        self.alphabet = alphabet
        """The alphabet used by the FST"""
        self.initial_state = State()
        """The initial (start) state of the FST"""
        self.states = {self.initial_state}
        """A set of all states in the FST"""
        self.final_states = set()
        """A set of all final (accepting) states of the FST"""

        if label == ('',):  # EPSILON
            self.final_states.add(self.initial_state)
            self.initial_state.final_weight = weight
        elif label is not None:
            self.alphabet = {s for s in label}
            target_state = State()
            self.states.add(target_state)
            self.final_states = {target_state}
            target_state.final_weight = weight
            self.initial_state.add_transition(target_state, label, 0.0)

    @classmethod
    def character_ranges(cls, ranges, complement = False) -> 'FST':
        """Returns a two-state FSM from a list of unicode code point range pairs.
           Keyword arguments:
           complement -- if True, the character class is negated, i.e. [^ ... ], and
           a two-state FST is returned with the single label . and all the symbols in
           the character class are put in the alphabet.
           """
        new_fst = cls()
        second_state = State()
        new_fst.states.add(second_state)
        new_fst.final_states = {second_state}
        second_state.final_weight = 0.0
        alphabet = set()
        for start, end in ranges:
            for symbol in range(start, end + 1):
                if symbol not in alphabet:
                    alphabet |= {chr(symbol)}
                    if not complement:
                        new_fst.initial_state.add_transition(second_state, (chr(symbol),), 0.0)
        if complement:
            new_fst.initial_state.add_transition(second_state, ('.',), 0.0)
            alphabet.add('.')
        new_fst.alphabet = alphabet
        return new_fst

    @classmethod
    def regex(cls, regular_expression, defined = {}, functions = set(), multichar_symbols=None):
        """Compile a regular expression and return the resulting FST.
           Keyword arguments:
           defined -- a dictionary of defined FSTs that the compiler can access whenever
                      a defined network is referenced in the regex, e.g. $vowel
           functions -- a set of Python functions that the compiler can access when a function
                       is referenced in the regex, e.g. $^myfunc(...)
        """
        import pyfoma.private.regexparse as regexparse
        if multichar_symbols is not None:
            escaper = _multichar_matcher(multichar_symbols)
            regular_expression = escaper.sub(_multichar_replacer, regular_expression)
        my_regex = regexparse.RegexParse(regular_expression, defined, functions)
        return my_regex.compiled

    re = regex

    @classmethod
    def from_strings(cls, strings, multichar_symbols=None):
        """Create an automaton that accepts words in the iterable 'strings'."""
        grammar = {"Start": ((w, "#") for w in strings)}
        lex = FST.rlg(grammar, "Start", multichar_symbols=multichar_symbols)
        return lex.determinize_as_dfa().minimize().label_states_topology()

    @classmethod
    def rlg(cls, grammar, start_symbol, multichar_symbols=None):
        """Compile a (weighted) right-linear grammar into an FST, similarly to lexc."""
        escaper = None
        if multichar_symbols is not None:
            escaper = _multichar_matcher(multichar_symbols)
        def _rlg_tokenize(w):
            if w == '':
                return ['']
            if escaper is not None:
                w = escaper.sub(_multichar_replacer, w)
            tokens = []
            tok_re = r"'(?P<multi>'|(?:\\'|[^'])*)'|\\(?P<esc>(.))|(?P<single>(.))"
            for mo in pyre.finditer(tok_re, w):
                token = mo.group(mo.last_group)
                if token == " " and mo.last_group == 'single':
                    token = ""  # normal spaces for alignment, escaped for actual
                elif mo.last_group == "multi":
                    token = token.replace(r"\'", "'")
                tokens.append(token)
            return tokens

        new_fst = FST(alphabet = set())
        state_dict = {name:State(name = name) for name in grammar.keys() | {"#"}}
        new_fst.initial_state = state_dict[start_symbol]
        new_fst.final_states = {state_dict["#"]}
        state_dict["#"].final_weight = 0.0
        new_fst.states = set(state_dict.values())

        for lex_state in state_dict.keys() - {"#"}:
            for rule in grammar[lex_state]:
                current_state = state_dict[lex_state]
                lhs = (rule[0],) if isinstance(rule[0], str) else rule[0]
                target = rule[1]
                i = _rlg_tokenize(lhs[0])
                o = i if len(lhs) == 1 else _rlg_tokenize(lhs[1])
                new_fst.alphabet |= {sym for sym in i + o if sym != ''}
                for ii, oo, idx in itertools.zip_longest(i, o, range(max(len(i), len(o))),
                    fillvalue = ''):
                    w = 0.0
                    if idx == max(len(i), len(o)) - 1:  # dump weight on last transition
                        target_state = state_dict[target] # before reaching another lex_state
                        w = 0.0 if len(rule) < 3 else float(rule[2])
                    else:
                        target_state = State()
                        new_fst.states.add(target_state)
                    new_tuple = (ii,) if ii == oo else (ii, oo)
                    current_state.add_transition(target_state, new_tuple, w)
                    current_state = target_state
        return new_fst

    # endregion

    # region Saving and loading
    def save(self, path: str):
        """Saves the current FST to a file.
        Args:
            path (str): The path to save to (without a file extension)
        """
        if not path.endswith('.fst'):
            path = path + '.fst'
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FST':
        """Loads an FST from a .fst file.
        Args:
            path (str): The path to load from. Must be a `.fst` file
        """
        if not path.endswith('.fst'):
            path = path + '.fst'
        with open(path, 'rb') as f:
            fst = pickle.load(f)
        return fst

    def save_att(self, base: PathLike[str], state_symbols=False, epsilon="@0@"):
        """Save to AT&T format files for use with other FST libraries
        (Foma, OpenFST, RustFST, HFST, etc).

        This will, in addition to saving the transitions in `base`,
        also create separate files with the extensions `.isyms` and
        `.osyms` containing the input and output symbol tables (so for
        example if base is `test.fst`, it will create `test.isyms` and
        `test.osyms`)

        Note also that the AT&T format has no mechanism for
        quoting or escaping characters (notably whitespace) in symbols
        and state names, but only tabs are used as field separators by
        default, so any other characters should be acceptable (though
        not always recommended).  The symbol `@0@` is used by default
        for epsilon (but can be changed with the `epsilon` parameter)
        as this is Foma's default, and will always have symbol ID 0 as
        this is required by OpenFST.

        If `state_symbols` is true, the names of states will be
        retained in the output file and a state symbol table created
        with the extension `.ssyms`.  This option is disabled by
        default since it is not compatible with Foma.
        """
        path = Path(base)
        ssym_path = path.with_suffix(".ssyms")
        isym_path = path.with_suffix(".isyms")
        osym_path = path.with_suffix(".osyms")
        # Number all states and create state symbol table
        if state_symbols:
            ssyms = [self.initial_state.name]
        else:
            ssyms = ["0"]
        ssymtab = {id(self.initial_state): ssyms[0]}
        for s in self.states:
            if s == self.initial_state:
                continue
            if s.name is None or not state_symbols:
                name = str(len(ssyms))
            else:
                name = s.name
            ssymtab[id(s)] = name
            ssyms.append(name)
        if state_symbols:
            with open(ssym_path, "wt") as output_file:
                for idx, name in enumerate(ssyms):
                    print(f"{name}\t{idx}", file=output_file)
        # Do a second pass to output the FST itself (we will always have
        # to do this because of the need to number states)
        isyms = {epsilon: 0}
        osyms = {epsilon: 0}

        def output_state(s: State, output_file: TextIO):
            name = ssymtab[id(s)]
            for label, arcs in s.transitions.items():
                if len(label) == 1:
                    isym = osym = (label[0] or epsilon)
                else:
                    isym, osym = ((x or epsilon) for x in label)
                if isym not in isyms:
                    isyms[isym] = len(isyms)
                if osym not in osyms:
                    osyms[osym] = len(osyms)
                for transition in arcs:
                    dest = ssymtab[id(transition.target_state)]
                    fields = [
                        name,
                        dest,
                        isym,
                        osym,
                    ]
                    if transition.weight != 0.0:
                        fields.append(transition.weight)
                    print("\t".join(fields), file=output_file)
            # NOTE: These are not required to be at the end of the file
            if s in self.final_states:
                name = ssymtab[id(s)]
                if s.final_weight != 0.0:
                    print(f"{name}\t{s.final_weight}", file=output_file)
                else:
                    print(name, file=output_file)

        with open(path, "wt") as output_file:
            output_state(self.initial_state, output_file)
            for s in self.states:
                if s != self.initial_state:
                    output_state(s, output_file)
        with open(isym_path, "wt") as output_file:
            for name, idx in isyms.items():
                print(f"{name}\t{idx}", file=output_file)
        with open(osym_path, "wt") as output_file:
            for name, idx in osyms.items():
                print(f"{name}\t{idx}", file=output_file)
    # endregion


    # region Utility Methods

    def __copy__(self):
        """Copy an FST. Actually calls copy_filtered()."""
        return self.copy_filtered()[0]

    def __len__(self):
        """Return the number of states."""
        return len(self.states)

    def __str__(self):
        """Generate an AT&T string representing the FST."""
        # Number states arbitrarily based on id()
        ids = [id(s) for s in self.states if s != self.initial_state]
        state_nums = {ids[i]:i+1 for i in range(len(ids))}
        state_nums[id(self.initial_state)] = 0 # The initial state is always 0
        st = ""
        for s in self.states:
            if len(s.transitions) > 0:
                for label in s.transitions.keys():
                    if len(label) == 1:
                        att_label = (label[0], label[0])
                    else:
                        att_label = label
                    # You get Foma's default here since it cannot be configured
                    att_label = ["@0@" if sym == "" else sym for sym in att_label]
                    for transition in s.transitions[label]:
                        st += '{}\t{}\t{}\t{}\n'.format(state_nums[id(s)],\
                        state_nums[id(transition.target_state)], '\t'.join(att_label),\
                        transition.weight)
        for s in self.states:
            if s in self.final_states:
                st += '{}\t{}\n'.format(state_nums[id(s)], s.final_weight)
        return st

    def __and__(self, other):
        """Intersection."""
        return self.intersection(other)

    def __or__(self, other):
        """Union."""
        return self.union(other)

    def __sub__(self, other):
        """Set subtraction."""
        return self.difference(other)

    def __pow__(self, other):
        """Cross-product."""
        return self.cross_product(other)

    def __mul__(self, other):
        """Concatenation."""
        return self.concatenate(other)

    def __matmul__(self, other):
        """Composition."""
        return self.compose(other)

    def become(self, other):
        """Hacky or what? We use this to mutate self for those algorithms that don't directly do it."""
        self.alphabet, self.initial_state, self.states, self.final_states = \
        other.alphabet, other.initial_state, other.states, other.final_states
        return self

    def number_unnamed_states(self, force = False) -> dict:
        """Sequentially number those states that don't have the 'name' attribute.
           If 'force == True', number all states."""
        cntr = itertools.count()
        ordered = [self.initial_state] + list(self.states - {self.initial_state})
        return {id(s):(next(cntr) if s.name == None or force == True else s.name) for s in ordered}

    def cleanup_sigma(self):
        """Remove symbols if they are no longer needed, including . ."""
        seen = {sym for _, lbl, _ in self.all_transitions(self.states) for sym in lbl}
        if '.' not in seen:
            self.alphabet &= seen
        return self

    def check_graphviz_installed(self):
        try:
            subprocess.run(["dot", "-V"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def view(self, raw=False, show_weights=False, show_alphabet=True) -> 'graphviz.Digraph':
        """Creates a 'graphviz.Digraph' object to view the FST. Will automatically display the FST in Jupyter.

            :param raw: if True, show label tuples and weights unformatted
            :param show_weights: force display of weights even if 0.0
            :param show_alphabet: displays the alphabet below the FST
            :return: A Digraph object which will automatically display in Jupyter.

           If you would like to display the FST from a non-Jupyter environment, please use :code:`FST.render`
        """
        import graphviz
        if not self.check_graphviz_installed():
            raise EnvironmentError("Graphviz executable not found. Please install [Graphviz](https://www.graphviz.org/download/). On macOS, use `brew install graphviz`.")

        def _float_format(num):
            if not show_weights:
                return ""
            s = '{0:.2f}'.format(num).rstrip('0').rstrip('.')
            s = '0' if s == '-0' else s
            return "/" + s

        def _str_fmt(s):  # Use greek lunate epsilon symbol U+03F5
            return (sublabel if sublabel != '' else '&#x03f5;' for sublabel in s)

        #        g = graphviz.Digraph('FST', filename='fsm.gv')

        sigma = "&Sigma;: {" + ','.join(sorted(a for a in self.alphabet)) + "}" \
            if show_alphabet else ""
        g = graphviz.Digraph('FST', graph_attr={"label": sigma, "rankdir": "LR"})
        state_nums = self.number_unnamed_states()
        if show_weights == False:
            if any(t.weight != 0.0 for _, _, t in self.all_transitions(self.states)) or \
                    any(s.final_weight != 0.0 for s in self.final_states):
                show_weights = True

        g.attr(rankdir='LR', size='8,5')
        g.attr('node', shape='doublecircle', style='filled')
        for s in self.final_states:
            g.node(str(state_nums[id(s)]) + _float_format(s.final_weight))
            if s == self.initial_state:
                g.node(str(state_nums[id(s)]) + _float_format(s.final_weight), style='filled, bold')

        g.attr('node', shape='circle', style='filled')
        for s in self.states:
            if s not in self.final_states:
                g.node(str(state_nums[id(s)]), shape='circle', style='filled')
                if s == self.initial_state:
                    g.node(str(state_nums[id(s)]), shape='circle', style='filled, bold')
            grouped_targets = defaultdict(set)  # {states}
            for label, t in s.all_transitions():
                grouped_targets[t.target_state] |= {(t.target_state, label, t.weight)}
            for target, target_label_set in grouped_targets.items():
                if raw == True:
                    label_list = sorted((str(l) + '/' + str(w) for t, l, w in target_label_set))
                else:
                    label_list = sorted((':'.join(_str_fmt(label)) + _float_format(w) for _, label, w in target_label_set))
                print_label = ', '.join(label_list)
                if s in self.final_states:
                    source_label = str(state_nums[id(s)]) + _float_format(s.final_weight)
                else:
                    source_label = str(state_nums[id(s)])
                if target in self.final_states:
                    target_label = str(state_nums[id(target)]) + _float_format(target.final_weight)
                else:
                    target_label = str(state_nums[id(target)])
                g.edge(source_label, target_label, label=graphviz.nohtml(print_label))
        return g

    def render(self, view=True, filename: str='FST', format='pdf'):
        """
        Renders the FST to a file and optionally opens the file.
        :param view: If True, the rendered file will be opened.
        :param format: The file format for the Digraph. Typically 'pdf', 'png', or 'svg'. View all formats: https://graphviz.org/docs/outputs/
        """
        digraph = self.view()
        digraph.format = format
        digraph.render(view=view, filename=filename, cleanup=True)

    def all_transitions(self, states):
        """Enumerate all transitions (state, label, Transition) for an iterable of states."""
        for state in states:
            for label, transitions in state.transitions.items():
                for t in transitions:
                    yield state, label, t

    def all_transitions_by_label(self, states):
        """Enumerate all transitions by label. Each yield produces a label, and those
           the target states. 'states' is an iterable of source states."""
        all_labels = {l for s in states for l in s.transitions.keys()}
        for l in all_labels:
            targets = set()
            for state in states:
                if l in state.transitions:
                    for transition in state.transitions[l]:
                        targets.add(transition.target_state)
            yield l, targets

    def copy_mod(self, mod_label=lambda l, w: l, mod_weight=lambda l, w: w):
        """Copies an FSM and possibly modifies labels and weights through functions.
           Keyword arguments:
           mod_label -- a function that modifies the label, takes label, weight as args.
           modweights -- a function that modifies the weight, takes label, weight as args.
        """
        new_fst = FST(alphabet=self.alphabet.copy())
        q1q2 = {k: State(name=k.name) for k in self.states}
        new_fst.states = set(q1q2.values())
        new_fst.final_states = {q1q2[s] for s in self.final_states}
        new_fst.initial_state = q1q2[self.initial_state]

        for s, lbl, t in self.all_transitions(q1q2.keys()):
            q1q2[s].add_transition(q1q2[t.target_state], mod_label(lbl, t.weight), mod_weight(lbl, t.weight))

        for s in self.final_states:
            q1q2[s].final_weight = s.final_weight

        return new_fst

    def copy_filtered(self, label_filter = lambda x: True):
        """Create a copy of self, possibly filtering out labels where them
           optional function 'label_filter' returns False."""
        new_fst = FST(alphabet = self.alphabet.copy())
        q1q2 = {k:State() for k in self.states}
        for s in self.states:
            q1q2[s].name = s.name
        new_fst.states = set(q1q2.values())
        new_fst.final_states = {q1q2[s] for s in self.final_states}
        new_fst.initial_state = q1q2[self.initial_state]

        for s, lbl, t in self.all_transitions(q1q2.keys()):
            if label_filter(lbl):
                q1q2[s].add_transition(q1q2[t.target_state], lbl, t.weight)

        for s in self.final_states:
            q1q2[s].final_weight = s.final_weight

        return new_fst, q1q2


    def generate(self: 'FST', word, weights=False, tokenize_outputs=False, obey_flags=True, print_flags=False):
        """Pass word through FST and return generator that yields all outputs."""
        yield from self.apply(word, inverse=False, weights=weights, tokenize_outputs=tokenize_outputs, obey_flags=obey_flags, print_flags=print_flags)

    def analyze(self: 'FST', word, weights=False, tokenize_outputs=False, obey_flags=True, print_flags=False):
        """Pass word through FST and return generator that yields all inputs."""
        yield from self.apply(word, inverse=True, weights=weights, tokenize_outputs=tokenize_outputs, obey_flags=obey_flags, print_flags=print_flags)

    def apply(self: 'FST', word, inverse=False, weights=False, tokenize_outputs=False, obey_flags=True, print_flags=False):
        """Pass word through FST and return generator that yields outputs.
           if inverse == True, map from range to domain.
           weights is by default False. To see the cost, set weights to True.
           obey_flags toggles whether invalid flag diacritic
           combinations are filtered out. By default, flags are
           treated as epsilons in the input. print_flags toggels whether flag
           diacritics are printed in the output. """
        IN, OUT = [-1, 0] if inverse else [0, -1]  # Tuple positions for input, output
        cntr = itertools.count()
        w = self.tokenize_against_alphabet(word)
        Q, output = [], []
        heapq.heappush(Q, (0.0, 0, next(cntr), [], self.initial_state))  # (cost, -pos, output, state)
        flag_filter = FlagStringFilter(self.alphabet) if obey_flags else None
        
        while Q:
            cost, negpos, _, output, state = heapq.heappop(Q)

            if state == None and -negpos == len(w) and (not obey_flags or flag_filter(output)):
                if not print_flags:
                    output = FlagOp.filter_flags(output)
                yield_output = ''.join(output) if not tokenize_outputs else output
                if weights == False:
                    yield yield_output
                else:
                    yield (yield_output, cost)
            elif state != None:
                if state in self.final_states:
                    heapq.heappush(Q, (cost + state.final_weight, negpos, next(cntr), output, None))
                for lbl, t in state.all_transitions():
                    if lbl[IN] == '' or FlagOp.is_flag(lbl[IN]):
                        heapq.heappush(Q, (cost + t.weight, negpos, next(cntr), output + [lbl[OUT]], t.target_state))
                    elif -negpos < len(w):
                        next_symbol = w[-negpos] if w[-negpos] in self.alphabet else '.'
                        appended_sym = w[-negpos] if (next_symbol == '.' and lbl[OUT] == '.') else lbl[OUT]
                        if next_symbol == lbl[IN]:
                            heapq.heappush(Q, (
                            cost + t.weight, negpos - 1, next(cntr), output + [appended_sym], t.target_state))

    def words(self: 'FST'):
        """A generator to yield all words. Yay BFS!"""
        Q = deque([(self.initial_state, 0.0, [])])
        while Q:
            s, cost, seq = Q.popleft()
            if s in self.final_states:
                yield cost + s.final_weight, seq
            for label, t in s.all_transitions():
                Q.append((t.target_state, cost + t.weight, seq + [label]))

    def tokenize_against_alphabet(self: 'FST', word) -> list:
        """Tokenize a string using the alphabet of the automaton."""
        tokens = []
        start = 0
        while start < len(word):
            t = word[start]  # Default is length 1 token unless we find a longer one
            for length in range(1, len(word) - start + 1):  # TODO: limit to max length
                if word[start:start + length] in self.alphabet:  # of syms in alphabet
                    t = word[start:start + length]
            tokens.append(t)
            start += len(t)
        return tokens

    def todict(self, utf16_maxlen=False) -> Dict[str, Any]:
        """Create a dictionary form of the FST for export to
        JSON/Javascript.  If it will ultimately be used by Javascript,
        pass `utf16_maxlen=True`."""
        # (re-)number all the states making sure the initial state is 0
        # (the Javascript code depends on this, all other state numbers
        # are arbitrary strings)
        state_nums = {id(self.initial_state): 0}
        for s in self.states:
            if s == self.initial_state:
                continue
            state_nums[id(s)] = len(state_nums)
        # No need to hold out the initial state since it has number 0
        transitions = {}
        finals = {}
        alphabet = {}
        maxlen = 0
        for s in self.states:
            src = state_nums[id(s)]
            for label, arcs in s.transitions.items():
                if len(label) == 1:
                    isym = osym = label[0]
                else:
                    isym, osym = label
                for sym in isym, osym:
                    # Omit epsilon from symbol table
                    if sym == "":
                        continue
                    if sym not in alphabet:
                        # Reserve 0, 1, 2 for epsilon, identity, unknown
                        alphabet[sym] = 3 + len(alphabet)
                        # For Javascript we will recompute this based on
                        # evil UTF-16
                        maxlen = max(maxlen, len(sym))
                # Nothing to do to the symbols beyond that as pyfoma
                # already uses the same convention of epsilon='', and JSON
                # encoding will take care of escaping everything for us
                transitions.setdefault(f"{src}|{isym}", []).extend(
                    # Note, weights are ignored...
                    {state_nums[id(arc.target_state)]: osym} for arc in arcs
                )
            if s in self.final_states:
                finals[src] = 1
        if utf16_maxlen:
            # Note utf-16le because we do not want a valuable BOM
            maxlen = max(len(k.encode('utf-16le')) for k in alphabet) // 2
        return {
            "t": transitions,
            "s": alphabet,
            "f": finals,
            "maxlen": maxlen,
        }

    def to_json(self, utf16_maxlen=False) -> str:
        """Create JSON (which is also Javascript) for an FST for use with
        `foma_apply_down.js`"""
        return json.dumps(self.todict(utf16_maxlen=utf16_maxlen), ensure_ascii=False)

    def to_js(self, js_netname: str = "myNet") -> str:
        """Create Javascript compatible with `foma2js.perl`"""
        return " ".join(("var", js_netname, "=", self.to_json(utf16_maxlen=True), ";"))
    # endregion


class Transition:
    __slots__ = ['target_state', 'label', 'weight']
    def __init__(self, target_state, label, weight):
        self.target_state = target_state
        self.label = label
        self.weight = weight


class State:
    def __init__(self, final_weight = None, name = None):
        __slots__ = ['transitions', '_transitions_in', '_transitions_out', 'final_weight', 'name']
        # Index both the first and last elements lazily (e.g. compose needs it)
        self.transitions = dict()     # (l_1,...,l_n):{transition1, transition2, ...}
        self._transitions_in = None    # l_1:(label, transition1), (label, transition2), ... }
        self._transitions_out = None   # l_n:(label, transition1), (label, transition2, ...)}
        if final_weight is None:
            final_weight = float("inf")
        self.final_weight = final_weight
        self.name = name

    @property
    def transitions_in(self) -> dict:
        """Returns a dictionary of the transitions from a state, indexed by the input
           label, i.e. the first member of the label tuple."""
        if self._transitions_in is None:
            self._transitions_in = defaultdict(set)
            for label, new_transition in self.transitions.items():
                for t in new_transition:
                    self._transitions_in[label[0]] |= {(label, t)}
        return self._transitions_in

    @property
    def transitions_out(self):
        """Returns a dictionary of the transitions from a state, indexed by the output
           label, i.e. the last member of the label tuple."""
        if self._transitions_out is None:
            self._transitions_out = defaultdict(set)
            for label, new_transition in self.transitions.items():
                for t in new_transition:
                    self._transitions_out[label[-1]] |= {(label, t)}
        return self._transitions_out

    def rename_label(self, original, new):
        """Changes labels in a state's transitions from original to new."""
        for t in self.transitions[original]:
            t.label = new
        self.transitions[new] = self.transitions.get(new, set()) | self.transitions[original]
        self.transitions.pop(original)

    def remove_transitions_to_targets(self, targets):
        """Remove all transitions from self to any state in the set targets."""
        newt = {}
        for label, transitions in self.transitions.items():
            newt[label] = {t for t in transitions if t.target_state not in targets}
            if len(newt[label]) == 0:
                newt.pop(label)
        self.transitions = newt

    def add_transition(self, other, label, weight):
        """Add transition from self to other with label and weight."""
        new_transition = Transition(other, label, weight)
        self.transitions[label] = self.transitions.get(label, set()) | {new_transition}

    def all_transitions(self):
        """Generator for all transitions out from a given state."""
        for label, transitions in self.transitions.items():
            for t in transitions:
                yield label, t

    def all_targets(self) -> set:
        """Returns the set of states a state has transitions to."""
        return {t.target_state for tr in self.transitions.values() for t in tr}

    def all_epsilon_targets_cheapest(self) -> dict:
        """Returns a dict of states a state transitions to (cheapest) with epsilon."""
        targets = defaultdict(lambda: float("inf"))
        for lbl, tr in self.transitions.items():
            if all(len(sublabel) == 0 for sublabel in lbl): # funky epsilon-check
                for s in tr:
                    targets[s.target_state] = min(targets[s.target_state], s.weight)
        return targets

    def all_targets_cheapest(self) -> dict:
        """Returns a dict of states a state transitions to (cheapest)."""
        targets = defaultdict(lambda: float("inf"))
        for tr in self.transitions.values():
            for s in tr:
                targets[s.target_state] = min(targets[s.target_state], s.weight)
        return targets
