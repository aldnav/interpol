import argparse
import re
import sys
import math

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
        return '<%s> %s' % (self.lexeme, self.name)


class SymbolTable(object):
    """Ideally a symbol table would look like:
    {
        'x': { 'name': 'x', 'type': 'int', 'value': 5 },
        'prompt': { 'name': 'prompt', 'type': 'string', 'value': '[Ni hao!]' }
    }
    """
    _table = OrderedDict()

    def insert(self, key, meta):
        """Insert a key with a meta to the symbol table.
        Arguments:
            key -- identifier of the symbol
            meta -- a dict containing information about the symbol
        """
        self._table[key] = meta

    def lookup(self, key):
        """Retrieves symbol. Returns None if the key is not present"""
        return self._table.get(key, None)

    def __str__(self):
        if len(self._table.keys()) == 0:
            return '\nSYMBOL TABLE is empty\n'
        display = (
            '\nSYMBOL TABLE\n'
            'NAME\t\t| TYPE \t\t | VALUE\n'
        )
        for k, v in self._table.items():
            var_value = v.get('value', None)
            if v['type'] == 'string' and var_value is not None:
                var_value = var_value[1:-1]
            display += '%s\t\t| %s  \t | %r\n' % (
                k, v['type'], var_value)
        return display.expandtabs(10)

    def set_value(self, key, value):
        """Sets the value of the symbol meta."""
        if self.lookup(key) is not None:
            try:
                value = int(value)
            except Exception:
                pass
            self.lookup(key)['value'] = value
            return value
        return None


symbol_table = SymbolTable()
primitives = (
    Token('begin_program', 'CREATE', 'BEGIN'),
    Token('end_program', 'RUPTURE', 'END'),

    Token('type_string', 'DSTR', '<DATATYPE>'),
    Token('type_integer', 'DINT', '<DATATYPE>'),

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
            if token.lexeme.endswith(']'):
                string_end_index = i
        if (token.name == 'string_end' and
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
        raise Exception({
            'error': 'LexingError',
            'message': 'While scanning String literal starting from "%s".\n'
                       'Must be an unclosed string.'
                        % token_results[string_start_index].lexeme})

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
        raise Exception({
            'error': 'LexingError',
            'message': 'Invalid token! %s' % text})


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
        if (token.name in ['type_integer', 'type_string'] and
                next_token.name == 'identifier'):
            symbol_table.insert(next_token.lexeme, {
                'name': next_token.lexeme,
                'type': token.name.replace('type_', ''),
                'reference_token': next_token
            })


interpol_dfa = {
    1: {'<IDENTIFIER>': 7, '<DATATYPE>': 2, 'END': 3, 'BEGIN': 3, '<IO_FUNCTION>': 8, 'STORE': 10},
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
}

accepting_states = [4, 6, 9]


class SyntaxChecker(object):

    def __init__(self, tokens, symbol_table, code_began, code_ended):
        self.tokens = tokens
        self.parse_tree = ParseTree()
        self.symbol_table = symbol_table
        self.code_began = code_began
        self.code_ended = code_ended
        self.error_msg = ''
        self.token_index = 0  # tracks current token index
        self.end_state = 1  # If end state is an element of final state, line has no syntax error.
        self.operator_stack = []

    def accept_interpol(self, transitions):
        state = 1
        tokens_size = len(self.tokens)
        # iterate through tokens produced by lexer
        while self.token_index < tokens_size:
            # simulation of dfa for interpol grammar
            token = self.tokens[self.token_index]
            temp_state = transitions[state].get(token.group, None)
            if token.name == 'begin_program':
                self.error_msg = 'Invalid begin token.'
                state = 13  # set to dead end state
                break
            elif code_began:
                if temp_state is None:
                    # if current state is a state with transition for expression
                    # and the current token is possibly an expression
                    if (token.group in [
                            '<IDENTIFIER>', '<OPERATOR>', '<STRING>', '<INTEGER>',
                            '<DIST>', '<MEAN>'] and
                            state in [5, 10, 7, 8]):
                        if self.accept_exp():
                            state = transitions[state].get('<EXP>', None)
                        else:
                            if not self.error_msg:
                                self.error_msg = 'Invalid expression.'
                            state = 13  # set to dead end state
                            break   # send to dead end
                    else:
                        if state in [5, 10, 7, 8]:
                            self.error_msg = 'Expected expression.'
                            state = 13  # set to dead end state

                        else:
                            if state == 4:
                                self.error_msg = 'Expected WITH keyword.'
                                state = 13
                        break
                else:
                    if token.group == '<IO_FUNCTION>':
                        token_node = Node(token, 1)
                        self.parse_tree.add_token(token_node)
                        self.parse_tree.current = token_node
                    elif token.group in ['<DATATYPE>', 'STORE']:
                        token_node = Node(token, 2)
                        self.parse_tree.add_token(token_node)
                        self.parse_tree.current = token_node
                    elif token.group == '<IDENTIFIER>':
                        token_node = Node(token, 0)
                        self.parse_tree.add_token(token_node)
                    state = temp_state
                    self.token_index = self.token_index + 1
            else:
                self.error_msg = 'Syntax Error! Expected BEGIN token'
                state = 13  # set to dead end state
                break
        self.end_state = state

    # check if tokens form an expression
    def accept_exp(self):
        is_accepted = False
        token = self.tokens[self.token_index]
        if self.token_index != 0:
            if (self.tokens[self.token_index-1].lexeme == 'GIVEME?' and
                    token.group not in ['<IDENTIFIIER>']):
                self.error_msg = 'Syntax Error! Expected <IDENTIFIER>.'
                return False
        if token.group in ['<IDENTIFIER>', '<STRING>', '<INTEGER>']:
            token_node = Node(token, 0)
            if self.token_index != 0:
                if (self.tokens[self.token_index - 1].group in [
                        '<OPERATOR>', '<DIST>', '<MEAN>'] and
                        token.group == '<STRING>'):
                    self.error_msg = 'Type Error! String cannot be an operand.'
                    return False
            if (len(self.parse_tree.current.children) ==
                    self.parse_tree.current.max_children):
                self.parse_tree.current = self.parse_tree.current.parent
            self.parse_tree.add_token(token_node)
            self.token_index = self.token_index + 1
#            self.tokens.remove(self.tokens[0])
            return True

        # for plus, minus, times, divby, modu, raise and nth root operations
        elif token.group == '<OPERATOR>':
            token_node = Node(token, 2)
            self.parse_tree.add_token(token_node)
            self.parse_tree.current = token_node
            self.token_index = self.token_index + 1
#            self.tokens.remove(self.tokens[0])
            self.operator_stack.append(token_node)
            is_exp = self.accept_exp()
            if is_exp:
                is_exp = self.accept_exp()
                if is_exp:
                    self.operator_stack.pop()
                    return True
                elif is_exp is None:
                    return True
            elif is_exp is None:
                return True

        # for distance operation
        elif token.group == '<DIST>':
            token_node = Node(token, 4)
            self.parse_tree.add_token(token_node)
            self.parse_tree.current = token_node
            self.token_index = self.token_index + 1
#            self.tokens.remove(self.tokens[0])
            self.operator_stack.append(token_node)
            is_exp = self.accept_exp()
            if is_exp:
                is_exp = self.accept_exp()
                if is_exp:
                    token = self.tokens[self.token_index]
                    if token.group == 'AND':
                        self.token_index = self.token_index + 1
#                        self.tokens.remove(self.tokens[0])
                        is_exp = self.accept_exp()
                        if is_exp:
                            is_exp = self.accept_exp()
                            if is_exp:
                                self.parse_tree.current = self.parse_tree.current.parent
                                self.operator_stack.pop()
                                return True
                            elif is_exp is None:
                                return True
                        elif is_exp is None:
                            return True
                elif is_exp is None:
                    return True
            elif is_exp is None:
                return True

        # for mean operation
        elif token.group == '<MEAN>':
            token_node = Node(token, None)
            self.parse_tree.add_token(token_node)
            self.parse_tree.current = token_node
            self.token_index = self.token_index + 1
            while (self.token_index < len(self.tokens) and
                    token.group in ['<MEAN>', '<OPERATOR>', '<DIST>',
                                    '<INTEGER>', '<IDENTIFIER>']):
                if self.accept_exp():
                    is_accepted = True
                else:
                    is_accepted = False
            self.parse_tree.current = token_node
            operator_stack = self.operator_stack
            while operator_stack:
                top_node = operator_stack[len(operator_stack) - 1]
                # transfer operations not part of mean expression
                while len(top_node.children) < top_node.max_children:
                    operator_stack.pop()
                    if len(self.parse_tree.current.children) > 0:
                        last_child = self.parse_tree.current.children.pop()
                        top_node.add_child(last_child)
                        last_child.parent = top_node
                    else:
                        return False
                is_accepted = None
        else:
            is_accepted = False

        return is_accepted

    def print_error_msg(self):
        for token in interpol_dfa[self.end_state].keys():
            if token == interpol_dfa[end_state].keys()[-1]:
                print "%s " % token
            else:
                print "%s, " % token,


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
            self.root = obj
            self.current = obj
        else:
            obj.parent = self.current
            self.current.add_child(obj)


class PostfixEvaluator(object):

    def __init__(self, symbol_table, parse_list):
        self.symbol_table = symbol_table
        self.parse_list = parse_list
        self.error_msg = ''
        self.operand_stack = Stack()

    def evaluate(self):

        for node in self.parse_list:
            token = node.token
            if self.error_msg:
                return None
            else:
                if token.group in ['<IDENTIFIER>', '<STRING>', '<INTEGER>']:
                    self.operand_stack.push(token)
                elif token.group == '<OPERATOR>':
                    temp_operand2 = self.operand_stack.pop()
                    operand2 = self.get_integer_operand(temp_operand2)
                    if operand2 is not None:
                        temp_operand1 = self.operand_stack.pop()
                        operand1 = self.get_integer_operand(temp_operand1)
                        if operand1 is not None:
                            result = self.do_basic_arithmetic(
                                operand1, operand2, token.lexeme)
                            self.operand_stack.push(result)
                        else:
                            break
                    else:
                        break
                elif token.group in ['STORE']:
                    operand2 = self.operand_stack.pop()
                    operand1 = self.operand_stack.pop()
                    self.perform_assignment(operand1, operand2)
                elif token.group == '<DATATYPE>':
                    operand1 = self.operand_stack.pop()
                    if(self.operand_stack.isEmpty() is False):
                        # bind operand1 with datatype
                        operand2 = self.operand_stack.pop()
                        self.perform_assignment(operand1, operand2)
                elif token.lexeme == 'GIVEYOU!':
                    print '>>', self.get_token_output(self.operand_stack.pop()),
                elif token.lexeme == 'GIVEYOU!!':
                    print '>>', self.get_token_output(self.operand_stack.pop())
                elif token.lexeme == 'GIVEME?':
                    var = self.operand_stack.pop()
                    if self.find_symbol(var) is None:
                        self.error_msg = 'Input Error! '\
                            'Symbol `%s` for input not found.' % var.lexeme
                        break
                    user_input = raw_input()
                    self.get_user_input(var, user_input)
                elif token.lexeme == 'MEAN':
                    items = []
                    index = 0
                    operand_size = len(node.children)
                    while index < operand_size:
                        operand = self.get_integer_operand(
                            self.operand_stack.pop())
                        items.append(operand)
                        index = index + 1
                    self.operand_stack.push(self.get_mean(items))
                elif token.lexeme == 'DIST':
                    temp_operand4 = self.operand_stack.pop()
                    operand4 = self.get_integer_operand(temp_operand4)
                    if operand4 is not None:
                        temp_operand3 = self.operand_stack.pop()
                        operand3 = self.get_integer_operand(temp_operand3)
                        if operand3 is not None:
                            temp_operand2 = self.operand_stack.pop()
                            operand2 = self.get_integer_operand(temp_operand2)
                            if operand2 is not None:
                                temp_operand1 = self.operand_stack.pop()
                                operand1 = self.get_integer_operand(temp_operand1)
                                if operand1 is not None:
                                    result = self.get_distance(
                                        operand1, operand2, operand3, operand4)
                                    self.operand_stack.push(result)

        if not self.operand_stack.isEmpty():
            return self.operand_stack.pop()

    def get_token_output(self, token):
        if (isinstance(token, Token) and (token.name is not 'identifier')):
            return token.lexeme.replace('[', '').replace(']', '')
        else:
            symbol = self.find_symbol(token)

            if symbol is not None:
                if symbol['value']:
                    if symbol['type'] == 'string':
                        return symbol['value'].replace('[','').replace(']','')
                    else:
                        return symbol['value']
                else:
                    self.error_msg = 'Uninitialized symbol: %s' % (symbol.lexeme)

    def get_distance(self, op1, op2, op3, op4):
        dist = math.sqrt((op3 - op1) ** 2 + (op4 - op2) ** 2)
        return Token('integer', dist, '<INTEGER>')

    def get_mean(self, operands):
        result = sum(operands) / (len(operands))
        return Token('integer', result, '<INTEGER>')

    def get_integer_operand(self, token):
        if token.name == 'integer':
            return int(token.lexeme)
        else:
            operand = self.get_token_value(token)
            if operand:
                if self.get_symbol_type(token) == 'integer':
                    return int(operand)
                else:
                    return None
            else:
                return None

    def get_symbol_type(self, token):
        symbol = self.find_symbol(token)

        if symbol is not None:
            return symbol['type']

    def get_token_value(self, token):
        if (isinstance(token, Token) and (token.name is not 'identifier')):
            return token.lexeme
        else:
            symbol = self.find_symbol(token)

            if symbol is not None:
                if symbol['value']:
                    return symbol['value']
                else:
                    self.error_msg = 'Uninitialized symbol: %s' % (symbol.lexeme)

    def find_symbol(self, token):
        symbol = self.symbol_table.lookup(token.lexeme)

        if symbol is not None:
            return symbol
        else:
            self.error_msg = 'Cannot find symbol: "%s"!' % (token.lexeme)
            return None

    # op2 = varname , op1 = value
    def perform_assignment(self, op1, op2):
        """
        Arguments:
            op1 -- value
            op2 -- varname
        """
        symbol = self.find_symbol(op2)

        if symbol is not None:
            if symbol['type'] == op1.name:
                symbol_table.set_value(op2.lexeme, op1.lexeme)
            else:
                self.error_msg = 'Type Error! '\
                    '%s and %s do not have the same datatype.' % (
                        op1.lexeme, op2.lexeme)

    def do_basic_arithmetic(self, op1, op2, operator):
        try:
            op1 = int(op1)
        except Exception as e:
            raise e
        try:
            op2 = int(op2)
        except Exception as e:
            raise e
        result = 0
        if operator == 'PLUS':
            result = op1 + op2
        elif operator == 'MINUS':
            result = op1 - op2
        elif operator == 'TIMES':
            result = op1 * op2
        elif operator == 'DIVBY':
            result = op1 / op2
        elif operator == 'RAISE':
            result = op1 ** op2
        elif operator == 'ROOT':
            result = (op1 ** (1 / op2))
        return Token('integer', result, '<INTEGER>')

    def get_user_input(self, var, user_input):
        """
        Arguments:
            var -- variable to assign value
            user_input -- the user raw input
        """
        input_token = None
        if self.find_symbol(var)['type'] == 'integer':
            try:
                input_token = Token('integer', int(user_input), '<INTEGER>')
            except Exception:
                self.error_msg = 'Input Error! '\
                    'Malformed integer input for `%s`' % var.lexeme
        else:
            input_token = Token('string', '[%s]' % user_input, '<STRING>')
        if len(self.error_msg) == 0:
            self.perform_assignment(input_token, var)
        return input_token


class Stack(object):
    def __init__(self):
        self.tokens = []

    def isEmpty(self):
        return self.tokens == []

    def push(self, token):
        self.tokens.insert(0, token)

    def pop(self):
        return self.tokens.pop(0)

    def peek(self):
        return self.tokens[0]

    def size(self):
        return len(self.tokens)


def visit(node):
    parse_list.append(node)


def walk_tree_df_postorder(node, visit):
    """Depth-first post-order."""
    if node is None:
        return
    for child in node.children:
        walk_tree_df_postorder(child, visit)
    visit(node)


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
        if line.strip().startswith('#'):
            continue
        print i + 1, line,
        if line:  # skip empty lines
            tokens = []
            try:
                tokens = tokenize(line)
            except Exception as e:
                if e.message['error'] == 'LexingError':
                    print
                    print '%s: %s\n%s: %s' % (
                        e.message['error'], e.message['message'],
                        filename, i + 1)
                    print '  ->', line
                    break
            # conditions for trapping codes outside begin and rupture

            temp_tokens = tokens

            if temp_tokens[0].name == 'begin_program' and not code_began:
                code_began = True
            elif temp_tokens[0].name == 'end_program' and not code_ended:
                code_ended = True
            elif code_ended:
                print 'Syntax Error! Expected end of file.\n%s: %s ' % (
                    filename, i)
                print '  ->', line
                has_error = True
                break  # end interpretation if error is found
            else:
                syntax_checker = SyntaxChecker(
                    temp_tokens, symbol_table, code_began, code_ended)
                syntax_checker.accept_interpol(interpol_dfa)
                end_state = syntax_checker.end_state
                if end_state in accepting_states:
                    walk_tree_df_postorder(syntax_checker.parse_tree.root, visit)

                    # evaluation
                    postfix_evaluator = PostfixEvaluator(symbol_table, parse_list)
                    postfix_evaluator.evaluate()
                    evaluation_error = postfix_evaluator.error_msg
                    if evaluation_error:
                        print '%s\n%s %s' % (evaluation_error,
                            filename, i + 1)
                        print '  ->', line
                        has_error = True
                        break
                    parse_list = []
                else:
                    index = 0
                    for token in tokens:
                        if index == syntax_checker.token_index:
                            # -> points at the source of syntax error
                            print '-> %s ' % token.lexeme,
                        else:
                            print token.lexeme,
                        index = index + 1
                    print
                    if syntax_checker.error_msg:
                        print syntax_checker.error_msg
                    else:
                        syntax_checker.print_error_msg()
                    has_error = True
                    break  # end interpretation if error is found
        if not has_error:
            print symbol_table
            print 'LEXEMES AND TOKENS:',
            print tokens
            print
