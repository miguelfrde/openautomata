
import os, sys, argparse, collections
from multiprocessing import Process, Queue
from texttable import Texttable
from termcolor import colored
from openautomata.regex import RegularExpression, SYMBOLS

try:
    import colorama
    colorama.init()
except:
    if os.name == 'nt':
        print "WARNING: Matches won't be highlighted, colorama is needed"
        print "Run: pip install colorama"

ABOUT =  """GREP like application that uses minimized Deterministic Finite Automatas
            and finds matches in a set of files"""

try: 
    import colorama 
    colorama.init() 
except:
    if os.name == 'nt': 
        print "WARNING: Matches won't be highlighted, colorama is needed" 
        print "Run: pip install colorama"

def get_current_dir_files():
    return [f for f in os.listdir(os.curdir) if os.path.isfile(f)]


def search_in_file(fname, regex, queue):
    with open(fname, 'r') as f:
        text = f.read().split('\n')
        results_text = list()
        for line_index, line in enumerate(text):
            for i, j, _ in regex.search(line):
                l = line[0:i]
                c = colored(line[i:j+1], 'red', attrs=['bold', 'underline'])
                r = line[j+1:] if j + 1 != len(line) else ''
                try: results_text.append((line_index, l + c + r, l + line[i:j+1] + r))
                except Exception:
                    print l, c, r, line[i:j+1]
                    exit()
        queue.put((fname, tuple(results_text)))


class Grep:

    def __init__(self, args):
        self.files = args.files if args.files != '.' else get_current_dir_files()
        self.regex = RegularExpression(args.regex)
        self.dfa = self.regex.dfa
        self.results = list()
        if args.show_automata or args.save_automata:
            self.create_table()
        else: self.table = None
    
    def search(self):
        if self.results:
            if print_results: self.print_results()
            return
        q = Queue()
        processes = list()
        kf = 0
        for fname in self.files:
            if os.path.exists(fname):
                print "Searching file", fname
                processes.append(Process(target=search_in_file,
                                     args=(fname, self.regex, q)))
                processes[-1].start()
            else:
                kf += 1
                print "ERROR: File %s doesn't exist" % fname
        for _ in xrange(len(self.files) - kf):
            self.results.append(q.get())
        for p in processes:
            p.join()

    def get_table(self):
        if self.table:
            return 'Transition table:\n' + self.table.draw() + '\n'
        return ''

    def print_table(self, out=sys.stdout):
        print >>out, self.get_table()

    def print_results(self, out=sys.stdout):
        stdout = out == sys.stdout
        print >>out, "Search results:"
        none = True
        for fname, results in self.results:
            for i, r1, r2 in results:
                none = False
                f = fname + ':%d> ' % i
                if stdout: print >>out, f + r1
                else: print >>out, f + r2
        if not stdout: out.close()
        if none: print "None"

    def create_table(self):
        alphabet = list(self.dfa.alphabet)
        initial = self.dfa.initial_state
        get_char = lambda s: ' '.join(c for (f, c) in
            [(self.dfa.is_initial, '->'), (self.dfa.is_final, '*')] if f(s))
        self.table = Texttable()
        self.table.set_cols_dtype(['t'] + ['i'] * len(alphabet))
        self.table.set_cols_align(['c'] * (len(alphabet) + 1))
        self.table.set_cols_valign(['c'] * (len(alphabet) + 1))
        self.table.header(['d(q, a)'] + alphabet)
        for s in [initial] + list(self.dfa.states - {initial}):
            trans = lambda s, a: list(self.dfa.get_transition(s, a))
            self.table.add_row(['%s q_%d' % (get_char(s), s)] + \
                [trans(s, a)[0]  if trans(s, a) else '-' for a in alphabet])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=ABOUT)
    parser.add_argument('regex',
                        metavar='REGEX',
                        help='Regular expression. Symbols: ' + ', '.join(SYMBOLS),
                        type=str)
    parser.add_argument('files',
                        metavar='FILES',
                        help="""Files to search the pattern on. 
                                If none: all files in current directory""",
                        nargs='*',
                        default='.',
                        type=str)
    parser.add_argument('-o',
                        help='Save output to a file called "output"',
                        action='store_true',
                        dest='save_output')
    parser.add_argument('-a',
                        help='Show transition table',
                        action='store_true',
                        dest='show_automata')
    parser.add_argument('-oa',
                        help='Save automata to file',
                        action='store_true',
                        dest='save_automata')

    args = parser.parse_args()
    grep = Grep(args)
    print
    if args.show_automata: grep.print_table()
    grep.search()
    grep.print_results()
    print
    if args.save_output:
        with open('output', 'w') as f:
            grep.print_results(out=f)
    if args.save_automata:
        with open('outputt', 'w') as f:
            grep.print_table(out=f)
