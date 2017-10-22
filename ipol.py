import re
import sys


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
        return '%s %s' % (self.name, self.lexeme)


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
    Token('print', 'GIVEYOU!'),
    Token('printwnl', 'GIVEYOU!!'),
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
    return map(detect_token, parts)


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
    elif re.fullmatch('[a-zA-Z_!][a-zA-Z0-9_!]*', text):
        return Token('identifier', text)
    else:
        print text
        raise Exception('Invalid token!')


if __name__ == '__main__':
    _, filename = sys.argv
    file_content = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        print tokenize(line)
