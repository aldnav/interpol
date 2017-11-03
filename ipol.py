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

    def set_value(self, key, value):
        self.lookup(key)['value']= value


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
    3: {},  # dead end state
    4: {'<WITH>': 5},  # accepting state
    5: {'<EXP>': 6},
    6: {},  # accepting state
    7: {'<EXP>': 6},
    8: {'<IDENTIFIER>': 9, '<EXP>': 9},
    9: {},  # accepting state
    10: {'<EXP>': 11},
    11: {'IN': 12},
    12: {'<IDENTIFIER>': 9},
    13: {'<EXP>': None},
    14: {'<END_OF_LINE>': None}  # dead end state
}

accepting_states =  [4, 6, 9]


class SyntaxChecker(object):

    def __init__(self, tokens, symbol_table, code_began, code_ended):
        self.tokens = tokens
        self.parse_tree = ParseTree()
        self.symbol_table = symbol_table
        self.code_began = code_began
        self.code_ended = code_ended


    def accept_interpol(self, transitions):
        state = 1
        # iterate through tokens produced by lexer
        while self.tokens:
            # simulation of dfa for interpol grammar
            group_name = self.tokens[0].group
            temp_state = transitions[state].get(self.tokens[0].group, None)
            print 'state %s %s' % (temp_state, self.tokens[0].group),
            if self.tokens[0].name == 'begin_program':
                print 'Invalid begin token.'
                return None
            elif self.tokens[0].name == 'end_program':
                print 'Invalid token. Code has ended already.'
                return None
            elif code_began:
                if temp_state is None:
                    # if current state is a state with transition for expression
                    # and the current token is possibly an expression
                    if (group_name in [
                            '<IDENTIFIER>', '<OPERATOR>', '<STRING>', '<INTEGER>',
                            '<DIST>', '<MEAN>'] and
                            state in [5, 10, 7, 8]):
                        if self.accept_exp():
                            state = transitions[state].get('<EXP>', None)
                        else:
                            return 13   # send to dead end
                    else:
                        if state in [5, 10, 7, 8]:
                            return 13
                        else:
                            return state # send to dead end
                else:
                    if group_name == '<IO_FUNCTION>':
                        token_node = Node(self.tokens[0], 1)
                        self.parse_tree.add_token(token_node)
                        self.parse_tree.current = token_node
                    elif group_name in ['<DATATYPE>','STORE']:
                        token_node = Node(self.tokens[0], 2)
                        self.parse_tree.add_token(token_node)
                        self.parse_tree.current = token_node
                    elif group_name == '<IDENTIFIER>':
                        token_node = Node(self.tokens[0], 0)
                        self.parse_tree.add_token(token_node)
                    state = temp_state
                    self.tokens.remove(self.tokens[0])
            else:
                print 'Syntax Error! Expected BEGIN token'
                return None
        return state

    # check if tokens form an expression
    def accept_exp(self):
        is_accepted = False
        if self.tokens[0].group in ['<IDENTIFIER>', '<STRING>', '<INTEGER>']:
            token_node = Node(self.tokens[0],0)
            if len(self.parse_tree.current.children) == self.parse_tree.current.max_children:
                self.parse_tree.current = self.parse_tree.current.parent
            self.parse_tree.add_token(token_node)
            self.tokens.remove(self.tokens[0])
            return True

        # for plus, minus, times, divby, modu, raise and nth root operations
        elif self.tokens[0].group == '<OPERATOR>':
            token_node = Node(self.tokens[0], 2)
            self.parse_tree.add_token(token_node)
            self.parse_tree.current = token_node # self.parse_tree.current.children[-1]
            self.tokens.remove(self.tokens[0])
            if self.accept_exp():
                if self.accept_exp():
                    return True

        # for distance operation
        elif self.tokens[0].group == '<DIST>':
            token_node = Node(self.tokens[0], 4)
            self.parse_tree.add_token(token_node)
            self.parse_tree.current = token_node  # self.parse_tree.current.children[-1]
            self.tokens.remove(self.tokens[0])
            if self.accept_exp():
                if self.accept_exp():
                    if self.tokens[0].group == 'AND':
                        self.tokens.remove(self.tokens[0])
                        if self.accept_exp():
                            if self.accept_exp():
                                self.parse_tree.current = self.parse_tree.current.parent
                                return True

        # for mean operation
        elif self.tokens[0].group == '<MEAN>':
            token_node = Node(self.tokens[0], None)
            self.parse_tree.add_token(token_node)
            self.parse_tree.current = token_node  # self.parse_tree.current.children[-1]
            self.tokens.remove(self.tokens[0])
            while self.tokens[0].group not in ['<OPERATOR>','<DIST>'] and \
                    self.accept_exp() and \
                    self.tokens:
                is_accepted = True

        return is_accepted


class Node(object):
    def __init__(self, token, max_children):
        self.token = token
        self.children = []
        self.parent = None
        self.max_children = max_children


    def add_child(self, node_obj):
        self.children.append(node_obj)


class ParseTree(object):


    def __init__(self):
        self.root = None
        self.current = None


    def add_token(self, obj):
        if self.root is None:
            self.root =  obj
            self.current = obj
        else:
            obj.parent = self.current
            self.current.add_child(obj)


class PostfixEvaluator(object):

    def evaluate(self, symbol_table, parse_list):
        operandStack = Stack()

        for token in parse_list:
            if token.group in ['<OPERATOR>']:
                operand1 = operandStack.pop()
                operand2 = operandStack.pop()
                result = self.doBasicArithmetic(operand1, operand2, token.lexeme)
                operandStack.push(result)

        return operandStack.pop()



    def doBasicArithmetic(self, op1, op2, operator):
        if operator == 'PLUS':
            return op1 + op2
        elif operator == 'MINUS':
            return op1 - op2
        elif operator == 'TIMES':
            return op1 * op2
        elif operator == 'DIVBY':
            return op1 / op2
        elif operator == 'RAISE':
            return op1 ** op2
        elif operator == 'ROOT':
            return op1 ** (1/op2)


class Stack(object):
    def __init__(self):
        self.tokens = []

    def isEmpty(self):
        return self.tokens == []

    def push(self, token):
        self.tokens.insert(0, token)

    def pop(self):
        return self.items.pop(0)

    def peek(self):
        return self.items[0]

    def size(self):
        return len(self.items)


    # def doAdvArithmetic(self, operator, *args, **kwargs):
    #     if operator == 'DIST':


def visit(node):
    print node.token
    return node.token


def walk_tree_df_postorder(node, visit):
    """Depth-first post-order."""
    if node is None:
        return
    for child in node.children:
        walk_tree_df_postorder(child, visit)
    parse_list.append(visit(node))

parse_list = []



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

    code_began = False
    code_ended = False
    has_error = False

    for i, line in enumerate(lines):
        print i + 1,
        tokens = tokenize(line)

        # conditions for trapping codes outside begin and rupture
        
        if tokens[0].name == 'begin_program' and not code_began:
            code_began = True
        elif tokens[0].name == 'end_program' and not code_ended:
            code_ended = True
        elif code_ended:
            print '%s: %s Syntax Error! Code already ended' % (filename, i)
            has_error = True
            break
        else:
            syntax_checker = SyntaxChecker(tokens, symbol_table, code_began, code_ended)
            print tokens
            end_state = syntax_checker.accept_interpol(interpol_dfa)
            if end_state in accepting_states:
                print "post order"
                walk_tree_df_postorder(syntax_checker.parse_tree.root, visit)
                while parse_list:
                    # This is temporary:
                    # evaluate
                    parse_list.pop()

            else:
                if end_state is not None:
                    print "%s: %s Syntax Error! expected " % (filename, i),
                    for token in interpol_dfa[end_state].keys():
                        if token == interpol_dfa[end_state].keys()[-1]:
                            print "%s " % (token)
                        else:
                            print "%s, " % (token),
                has_error = True
                break # end interpretation if error is found
    if not has_error:
        print symbol_table