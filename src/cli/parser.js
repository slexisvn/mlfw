import { tokenize, LangSyntaxError } from './tokenizer.js';

const PRECEDENCE = {
  'or': 1,
  'and': 2,
  '==': 3, '!=': 3, '<': 3, '<=': 3, '>': 3, '>=': 3,
  '+': 4, '-': 4,
  '*': 5, '/': 5, '@': 5,
  '**': 6,
};

const COMPOUND_OPS = { '+=': '+', '-=': '-', '*=': '*', '/=': '/', '**=': '**', '@=': '@' };

export function parse(source) {
  return new Parser(tokenize(source)).parseProgram();
}

class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }

  parseProgram(stop = null) {
    const body = [];
    this.skipLines();
    while (!this.at('eof') && !(stop && (this.atValue(stop) || this.at(stop)))) {
      body.push(this.parseStatement());
      this.skipLines();
    }
    return { type: 'Program', body };
  }

  parseStatement() {
    const start = this.current();
    if (this.atIdentifier('model') && this.peek(1).type === 'identifier') return this.parseModel();
    if (this.atIdentifier('forward')) return this.parseForward();
    if (this.atIdentifier('fn')) return this.parseFunctionDeclaration();
    if (this.atIdentifier('if')) return this.parseIf();
    if (this.atIdentifier('for')) return this.parseFor();
    if (this.atIdentifier('while')) return this.parseWhile();
    if (this.atIdentifier('break')) { this.next(); return this.locate({ type: 'Break' }, start); }
    if (this.atIdentifier('continue')) { this.next(); return this.locate({ type: 'Continue' }, start); }
    if (this.atIdentifier('return')) {
      this.next();
      return this.locate({ type: 'Return', value: this.parseExpression() }, start);
    }

    if (this.at('identifier')) {
      const nextVal = this.peek(1).value;
      if (nextVal === '=') {
        const name = this.next().value;
        this.expectValue('=');
        return this.locate({ type: 'Assign', name, value: this.parseExpression() }, start);
      }
      if (COMPOUND_OPS[nextVal] !== undefined) {
        const name = this.next().value;
        const opToken = this.next();
        const op = COMPOUND_OPS[opToken.value];
        return this.locate({ type: 'CompoundAssign', name, op, value: this.parseExpression() }, start);
      }
    }
    return this.locate({ type: 'ExpressionStatement', expression: this.parseExpression() }, start);
  }

  parseBlock() {
    this.expectValue(':');
    // One-line form: colon followed by statement on same line
    if (!this.at('newline') && !this.at('indent') && !this.at('eof')) {
      return [this.parseStatement()];
    }
    // Multi-line form: colon + newline + INDENT ... DEDENT
    this.skipLines();
    this.expect('indent');
    const body = this.parseProgram('dedent').body;
    this.expect('dedent');
    return body;
  }

  parseIf() {
    const start = this.expectIdentifier('if');
    const condition = this.parseExpression();
    const body = this.parseBlock();
    const elifs = [];
    this.skipLines();
    while (this.atIdentifier('elif')) {
      this.next();
      const elifCond = this.parseExpression();
      const elifBody = this.parseBlock();
      this.skipLines();
      elifs.push({ condition: elifCond, body: elifBody });
    }
    let elseBody = null;
    if (this.atIdentifier('else')) {
      this.next();
      elseBody = this.parseBlock();
    }
    return this.locate({ type: 'If', condition, body, elifs, elseBody }, start);
  }

  parseFor() {
    const start = this.expectIdentifier('for');
    const variable = this.expect('identifier').value;
    this.expectIdentifier('in');
    const iterable = this.parseExpression();
    const body = this.parseBlock();
    return this.locate({ type: 'For', variable, iterable, body }, start);
  }

  parseWhile() {
    const start = this.expectIdentifier('while');
    const condition = this.parseExpression();
    const body = this.parseBlock();
    return this.locate({ type: 'While', condition, body }, start);
  }

  parseFunctionDeclaration() {
    const start = this.expectIdentifier('fn');
    const name = this.expect('identifier').value;
    const params = this.parseNameList();
    const body = this.parseBlock();
    return this.locate({ type: 'FunctionDeclaration', name, params, body }, start);
  }

  parseModel() {
    const start = this.expectIdentifier('model');
    const name = this.expect('identifier').value;
    const params = this.parseNameList();
    const body = this.parseBlock();
    return this.locate({ type: 'ModelDeclaration', name, params, body }, start);
  }

  parseForward() {
    const start = this.expectIdentifier('forward');
    const params = [];
    while (!this.atValue(':')) {
      params.push(this.expect('identifier').value);
      if (!this.matchValue(',')) break;
      if (this.atValue(':')) break;
    }
    const body = this.parseBlock();
    return this.locate({ type: 'ForwardDeclaration', params, body }, start);
  }

  parseNameList() {
    if (!this.matchValue('(')) return [];
    const names = [];
    if (!this.atValue(')')) {
      do {
        names.push(this.expect('identifier').value);
        if (!this.matchValue(',')) break;
      } while (!this.atValue(')'));
    }
    this.expectValue(')');
    return names;
  }

  parseExpression(minPrec = 0) {
    let left = this.parsePrefix();
    while (true) {
      if (this.atValue('(')) {
        left = this.parseCall(left);
        continue;
      }
      if (this.atValue('.')) {
        const start = this.next();
        left = this.locate({ type: 'Member', object: left, property: this.expect('identifier').value }, start);
        continue;
      }
      if (this.atValue('[')) {
        left = this.parseIndex(left);
        continue;
      }
      const op = this.current().value;
      const prec = PRECEDENCE[op];
      if (prec === undefined || prec < minPrec) break;
      const start = this.next();
      const right = this.parseExpression(prec + (op === '**' ? 0 : 1));
      left = this.locate({ type: 'Binary', op, left, right }, start);
    }
    return left;
  }

  parsePrefix() {
    const token = this.current();
    if (token.value === '-' || token.value === '+') {
      this.next();
      return this.locate({ type: 'Unary', op: token.value, value: this.parseExpression(7) }, token);
    }
    if (token.type === 'identifier' && token.value === 'not') {
      this.next();
      return this.locate({ type: 'Unary', op: 'not', value: this.parseExpression(7) }, token);
    }
    if (token.type === 'number' || token.type === 'string') {
      this.next();
      return this.locate({ type: 'Literal', value: token.value }, token);
    }
    if (token.type === 'identifier') {
      this.next();
      if (token.value === 'true') return this.locate({ type: 'Literal', value: true }, token);
      if (token.value === 'false') return this.locate({ type: 'Literal', value: false }, token);
      if (token.value === 'null') return this.locate({ type: 'Literal', value: null }, token);
      return this.locate({ type: 'Identifier', name: token.value }, token);
    }
    if (this.matchValue('(')) {
      const value = this.parseExpression();
      this.expectValue(')');
      return value;
    }
    if (this.matchValue('[')) {
      const elements = [];
      if (!this.atValue(']')) {
        do {
          this.skipLines();
          if (this.atValue(']')) break;
          elements.push(this.parseExpression());
          this.skipLines();
        } while (this.matchValue(','));
      }
      this.expectValue(']');
      return this.locate({ type: 'Array', elements }, token);
    }
    throw this.error(`Expected expression, got '${token.value ?? token.type}'`);
  }

  parseCall(callee) {
    const start = this.expectValue('(');
    const args = [];
    this.skipLines();
    if (!this.atValue(')')) {
      do {
        this.skipLines();
        if (this.atValue(')')) break;
        if (this.at('identifier') && this.peek(1).value === '=') {
          const name = this.next().value;
          this.next();
          args.push({ name, value: this.parseExpression() });
        } else {
          args.push({ name: null, value: this.parseExpression() });
        }
        this.skipLines();
      } while (this.matchValue(','));
    }
    this.skipLines();
    this.expectValue(')');
    return this.locate({ type: 'Call', callee, args }, start);
  }

  parseIndex(object) {
    const start = this.expectValue('[');
    const items = [];
    if (this.atValue(']')) throw this.error('Expected index expression');
    do {
      this.skipLines();
      if (this.atValue(']')) break;
      items.push(this.parseIndexItem());
      this.skipLines();
    } while (this.matchValue(','));
    this.expectValue(']');
    return this.locate({ type: 'Index', object, items }, start);
  }

  parseIndexItem() {
    const start = this.current();
    let first = null;
    if (!this.atValue(':') && !this.atValue(',') && !this.atValue(']')) first = this.parseExpression();
    if (!this.matchValue(':')) {
      if (!first) throw this.error('Expected index expression');
      return first;
    }

    let end = null;
    let step = null;
    if (!this.atValue(':') && !this.atValue(',') && !this.atValue(']')) end = this.parseExpression();
    if (this.matchValue(':') && !this.atValue(',') && !this.atValue(']')) step = this.parseExpression();
    return this.locate({ type: 'Slice', start: first, end, step }, start);
  }

  skipLines() { while (this.at('newline')) this.next(); }
  current() { return this.tokens[this.pos]; }
  peek(n) { return this.tokens[this.pos + n] || this.tokens[this.tokens.length - 1]; }
  next() { return this.tokens[this.pos++]; }
  at(type) { return this.current().type === type; }
  atValue(value) { return this.current().value === value; }
  atIdentifier(value) { return this.at('identifier') && this.atValue(value); }
  matchValue(value) { if (this.atValue(value)) { this.next(); return true; } return false; }
  expect(type) {
    if (!this.at(type)) throw this.error(`Expected ${type}`);
    return this.next();
  }
  expectValue(value) {
    if (!this.atValue(value)) throw this.error(`Expected '${value}'`);
    return this.next();
  }
  expectIdentifier(value) {
    if (!this.atIdentifier(value)) throw this.error(`Expected '${value}'`);
    return this.next();
  }
  error(message) {
    const t = this.current();
    return new LangSyntaxError(message, t.line, t.column);
  }
  locate(node, token) {
    node.line = token.line;
    node.column = token.column;
    return node;
  }
}
