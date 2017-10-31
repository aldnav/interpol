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

    def __init__(self, name, lexeme, *args, **kwargs):
        # the symbolic name for the token entity
        # (e.g. `if` for the keyword if, `id` for identifier)
        self.name = name
        # the literal text of the token
        self.lexeme = lexeme

    def __repr__(self):
        return '<%s> %s' % (self.name, self.lexeme)


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
    Token('begin_program', 'CREATE'),
    Token('end_program', 'RUPTURE'),

    Token('type_string', 'DSTR'),
    Token('type_int', 'DINT'),

    Token('addition', 'PLUS'),
    Token('subtraction', 'MINUS'),
    Token('multiplication', 'TIMES'),
    Token('division', 'DIVBY'),
    Token('modulo', 'MODU'),
    Token('exponentiation', 'RAISE'),
    Token('nth_root', 'ROOT'),
    Token('average', 'MEAN'),
    Token('distance', 'DIST'),

    Token('input', 'GIVEME?'),
    Token('printwnl', 'GIVEYOU!!'),
    Token('print', 'GIVEYOU!'),
    Token('store_start', 'STORE'),
    Token('store_in', 'IN'),
    Token('declare_with', 'WITH'),
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
        token_results = token_results[:string_start_index]\
            + [Token('string', string_lexeme)]\
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
        return Token('integer', text)
    elif text.startswith('['):
        return Token('string_start', text)
    elif text.endswith(']'):
        return Token('string_end', text)
    # \[([^\]]+)]
    elif re.fullmatch('[a-zA-Z_][a-zA-Z0-9_\'\"]*', text):
        return Token('identifier', text)
    else:
        raise Exception('Invalid token!')


def add_to_symbol_table(tokens):
    """Identify a declaration statement and add a symbol to the symbol table
        Used in:  `tokenize`
    """
    for i, token in enumerate(tokens):
        next_token = None
        try:
            next_token = tokens[i+1]
        except Exception:
            continue
        if (token.name in ['type_int', 'type_string'] and
                next_token.name == 'identifier'):
            symbol_table.insert(next_token.lexeme, {
                    'name': next_token.lexeme,
                    'type': token.name.replace('type_', ''),
                    'reference_token': next_token
                })

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
        print tokenize(line)
    print symbol_table
