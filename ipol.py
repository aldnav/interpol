import re
import sys
from collections import deque


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
        return '%s %s %s' % (self.name, self.group, self.lexeme)


primitives = (
    Token('begin_program', 'CREATE', 'begin'),
    Token('end_program', 'RUPTURE', 'end'),

    Token('type_string', 'DSTR', 'datatype'),
    Token('type_int', 'DINT', 'datatype'),

    Token('addition', 'PLUS', 'operator'),
    Token('subtraction', 'MINUS', 'operator'),
    Token('multiplication', 'TIMES', 'operator'),
    Token('division', 'DIVBY', 'operator'),
    Token('modulo', 'MODU', 'operator'),
    Token('exponentiation', 'RAISE', 'operator'),
    Token('nth_root', 'ROOT', 'operator'),
    Token('average', 'MEAN', 'mean'),
    Token('distance', 'DIST', 'dist'),

    Token('input', 'GIVEME?', 'io'),
    Token('printwnl', 'GIVEYOU!!', 'io'),
    Token('print', 'GIVEYOU!', 'io'),
    Token('store_start', 'STORE', 'assignment'),
    Token('store_in', 'IN', 'in_assignment'),
    Token('declare_with', 'WITH', 'dec_with'),
    Token('dist_and', 'AND', 'dist_and')

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
                        + [Token('string', string_lexeme, 'string')] \
                        + token_results[string_end_index + 1:]
    elif string_start_index is not None and string_end_index is None:
        # @NOTE: Improve this
        raise Exception(
            'While scanning String literal starting from "%s".\n'
            'Must be an unclosed string.'
            % token_results[string_start_index].lexeme)

    tokens = token_results
    return tokens


def detect_token(text):
    if isinstance(text, Token):
        return text
    elif re.fullmatch('[0-9]*', text):
        return Token('integer', text, 'integer')
    elif text.startswith('['):
        return Token('string_start', text, 'string_start')
    elif text.endswith(']'):
        return Token('string_end', text, 'string_end')
    # \[([^\]]+)]
    elif re.fullmatch('[a-zA-Z_][a-zA-Z0-9_\'\"]*', text):
        return Token('identifier', text, 'identifier')
    else:
        raise Exception('Invalid token!')


interpol_dfa = {
    1: {'identifier': 7, 'datatype': 2, 'end': 3, 'begin': 3, 'io': 8, 'assignment': 10},
    2: {'identifier': 4},
    3: {},  #accepting state
    4: {'dec_with': 5},  #accepting state
    5: {'exp': 6},
    6: {},  #accepting state
    7: {'exp': 6},
    8: {'identifier': 9, 'exp': 9},
    9: {},  #accepting state
    10: {'exp': 11},
    11: {'in_assignment': 12},
    12: {'identifier': 9},
    13: {}  #dead end state
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
                if self.tokens[0].group in ['identifier', 'operator', 'string', 'integer', 'dist',
                                            'mean'] and state in [5, 10, 7, 8]:
                    if self.accept_exp():
                        state = transitions[state].get('exp',None)
                    else:
                        return False
                else:
                    return False
            else:
                state = temp_state
                self.tokens.remove(self.tokens[0])
        return state in accepting_states

    #check if tokens form an expression
    def accept_exp(self):
        is_accepted = False
        if self.tokens[0].group in ['identifier', 'string', 'integer']:
            self.tokens.remove(self.tokens[0])
            return True

        #for plus, minus, times, divby, modu, raise and nth root operations
        elif self.tokens[0].group == 'operator':
            self.tokens.remove(self.tokens[0])
            if self.accept_exp():
                if self.accept_exp():
                    return True

        #for distance operation
        elif self.tokens[0].group == 'dist':
            self.tokens.remove(self.tokens[0])
            if self.accept_exp():
                if self.accept_exp():
                    if self.tokens[0].group == 'dist_and':
                        self.tokens.remove(self.tokens[0])
                        if self.accept_exp():
                            if self.accept_exp():
                                return True

        # for mean operation
        elif self.tokens[0].group == 'mean':
            self.tokens.remove(self.tokens[0])
            while self.accept_exp() and self.tokens:
                is_accepted = True

        return is_accepted


if __name__ == '__main__':
    _, filename = sys.argv
    file_content = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # print tokenize(line)
        tokens = tokenize(line)
        syntax_checker = SyntaxChecker(tokens)
        print tokens
        print syntax_checker.accept_interpol(interpol_dfa, [3, 4, 6, 9])
