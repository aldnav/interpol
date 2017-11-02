import argparse
import re
import sys

from collections import OrderedDict


def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    return re.match("(?:" + regex + r")\Z", string, flags=flags)


try:
    re.fullmatch
except Exception as e:
    setattr(re, 'fullmatch', fullmatch)
    re.fullmatch = fullmatch


class Token(object):
    def __init__(self, name, lexeme, group, *args, **kwargs):
        # the symbolic name for the token entity
        # (e.g. `if` for the keyword if, `id` for identifier)
        self.name = name
        # the literal text of the token
        self.lexeme = lexeme
        # the group name of token
        self.group = group

    def __repr__(self):
        return '<%s> %s %s' % (self.lexeme, self.name, self.group)


class SymbolTable(object):
    _table = OrderedDict()

    def insert(self, key, value):
        self._table[key] = value

    def lookup(self, key):
        return self._table.get(key, None)

    def __str__(self):
        display = (
            'SYMBOL TABLE\n' +
            '-' * 30 + '\n'
            'NAME\t\t| TYPE\n' +
            '-' * 30 + '\n'
        )
        for k, v in self._table.items():
            display += '%s\t\t| %s\n' % (k, v['type'])
        return display.expandtabs(10)


symbol_table = SymbolTable()


primitives = (
    Token('begin_program', 'CREATE', 'BEGIN'),
    Token('end_program', 'RUPTURE', 'END'),

    Token('type_string', 'DSTR', '<DATATYPE>'),
    Token('type_int', 'DINT', '<DATATYPE>'),

    Token('addition', 'PLUS', '<OPERATOR>'),
    Token('subtraction', 'MINUS', '<OPERATOR>'),
    Token('multiplication', 'TIMES', '<OPERATOR>'),
    Token('division', 'DIVBY', '<OPERATOR>'),
    Token('modulo', 'MODU', '<OPERATOR>'),
    Token('exponentiation', 'RAISE', '<OPERATOR>'),
    Token('nth_root', 'ROOT', '<OPERATOR>'),
    Token('average', 'MEAN', '<MEAN>'),
    Token('distance', 'DIST', '<DIST>'),

    Token('input', 'GIVEME?', '<IO_FUNCTION>'),
    Token('printwnl', 'GIVEYOU!!', '<IO_FUNCTION>'),
    Token('print', 'GIVEYOU!', '<IO_FUNCTION>'),
    Token('store_start', 'STORE', 'STORE'),
    Token('store_in', 'IN', 'IN'),
    Token('declare_with', 'WITH', '<WITH>'),
    Token('dist_and', 'AND', 'AND')

)


def tokenize(line):
    """Pass 1.1 Lexer spits out tokens"""
    tokens = re.split('\s+', line)

    # Go through every primitive and find a match to already known keywords
    for primitive in primitives:
        parts = []
        for token in tokens:
            if isinstance(token, Token):
                parts.append(token)
            else:
                for s in token.split(primitive.lexeme):
                    if len(s) > 0:
                        parts.append(s)
                    parts.append(primitive)
                parts.pop()
        tokens = parts

    # Identify identifiers, integer and string literals
    token_results = []
    for token in tokens:
        token_results.append(detect_token(token))

    # Purge string lexeme parts into one string token
    # Anything enclosed to "[ ]" is a string
    string_start_index = None
    string_end_index = None
    for i, token in enumerate(token_results):
        if token.name == 'string_start' and string_start_index is None:
            string_start_index = i
        elif (token.name == 'string_end' and
              string_start_index is not None and
              string_end_index is None):
            string_end_index = i
    if string_start_index is not None and string_end_index is not None:
        string_lexeme = ' '.join([
            token.lexeme for token in
            token_results[string_start_index:string_end_index + 1]])
        token_results = token_results[:string_start_index] \
            + [Token('string', string_lexeme, '<STRING>')] \
            + token_results[string_end_index + 1:]
    elif string_start_index is not None and string_end_index is None:
        # @NOTE: Improve this
        raise Exception(
            'While scanning String literal starting from "%s".\n'
            'Must be an unclosed string.'
            % token_results[string_start_index].lexeme)

    tokens = token_results
    add_to_symbol_table(tokens)  # initial pass on adding symbols to table
    return tokens


def detect_token(text):
    if isinstance(text, Token):
        return text
    elif re.fullmatch('[0-9]*', text):
        return Token('integer', text, '<INTEGER>')
    elif text.startswith('['):
        return Token('string_start', text, 'string_start')
    elif text.endswith(']'):
        return Token('string_end', text, 'string_end')
    # \[([^\]]+)]
    elif re.fullmatch('[a-zA-Z_][a-zA-Z0-9_\'\"]*', text):
        return Token('identifier', text, '<IDENTIFIER>')
    else:
        raise Exception('Invalid token!')


def add_to_symbol_table(tokens):
    """Identify a declaration statement and add a symbol to the symbol table
        Used in:  `tokenize`
    """
    for i, token in enumerate(tokens):
        next_token = None
        try:
            next_token = tokens[i + 1]
        except Exception:
            continue
        if (token.name in ['type_int', 'type_string'] and
                next_token.name == 'identifier'):
            symbol_table.insert(next_token.lexeme, {
                'name': next_token.lexeme,
                'type': token.name.replace('type_', ''),
                'reference_token': next_token
            })


interpol_dfa = {
    1: {'<IDENTIFIER>':7, '<DATATYPE>':2, 'END':3, 'BEGIN':3, '<IO_FUNCTION>':8, 'STORE':10},
    2: {'<IDENTIFIER>': 4},
    3: {},  # accepting state
    4: {'<WITH>': 5},  # accepting state
    5: {'<EXP>': 6},
    6: {},  # accepting state
    7: {'<EXP>': 6},
    8: {'<IDENTIFIER>': 9, '<EXP>': 9},
    9: {},  # accepting state
    10: {'<EXP>': 11},
    11: {'IN': 12},
    12: {'<IDENTIFIER>': 9},
    13: {}  # dead end state
}


class SyntaxChecker(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def accept_interpol(self, transitions, accepting_states):
        state = 1
        # iterate through tokens produced by lexer
        while self.tokens:
            # simulation of dfa for interpol grammar
            temp_state = transitions[state].get(self.tokens[0].group, None)
            print 'state %s %s' % (temp_state, self.tokens[0].group),
            if temp_state is None:
                # if current state is a state with transition for expression
                # and the current token is possibly an expression
                if (self.tokens[0].group in [
                        '<IDENTIFIER>', '<OPERATOR>', '<STRING>', '<INTEGER>', 
                        '<DIST>', '<MEAN>'] and
                        state in [5, 10, 7, 8]):
                    if self.accept_exp():
                        state = transitions[state].get('<EXP>', None)
                    else:
                        return 13
                else:
                	if state in [5, 10, 7, 8]:
                        return 13
                    else:
                    	return state
            else:
                state = temp_state
                self.tokens.remove(self.tokens[0])
        return state in accepting_states

    # check if tokens form an expression
    def accept_exp(self):
        is_accepted = False
        if self.tokens[0].group in ['<IDENTIFIER>', '<STRING>', '<INTEGER>']:
            self.tokens.remove(self.tokens[0])
            return True

        # for plus, minus, times, divby, modu, raise and nth root operations
        elif self.tokens[0].group == '<OPERATOR>':
            self.tokens.remove(self.tokens[0])
            if self.accept_exp():
                if self.accept_exp():
                    return True

        # for distance operation
        elif self.tokens[0].group == '<DIST>':
            self.tokens.remove(self.tokens[0])
            if self.accept_exp():
                if self.accept_exp():
                    if self.tokens[0].group == 'AND':
                        self.tokens.remove(self.tokens[0])
                        if self.accept_exp():
                            if self.accept_exp():
                                return True

        # for mean operation
        elif self.tokens[0].group == '<MEAN>':
            self.tokens.remove(self.tokens[0])
            while self.accept_exp() and self.tokens:
                is_accepted = True

        return is_accepted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename',
        help=('INTERPOL file to execute. (`*.ipol`)\n'
              'When left empty, interpreter will ask for the file.'))
    args = parser.parse_args()
    if args.filename:
        filename = args.filename
    else:
        filename = raw_input('File to execute: ')
    try:
        assert filename.split('.')[-1] == 'ipol'
    except AssertionError as e:
        print 'File not supported. INTERPOL file has the extension `.ipol`'
        sys.exit(1)
    file_content = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        print i + 1,
        tokens = tokenize(line)
        syntax_checker = SyntaxChecker(tokens)
        print tokens
        print syntax_checker.accept_interpol(interpol_dfa, [3, 4, 6, 9])
    print symbol_table
