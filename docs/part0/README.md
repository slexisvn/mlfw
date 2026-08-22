# Part 0 — Orientation

Three short chapters before the real material starts. They tell you what the book is, get code running on your machine, and hand you a map you will keep coming back to.

| Chapter | Title | What you get out of it |
|---|---|---|
| [1](ch01-what-this-book-is/README.md) | What this book is, and how to read it | The contract: who it is for, how every chapter is built, how to read it in the order that suits you |
| [2](ch02-setting-up/README.md) | Setting up | A working checkout, three labs that print real IR, real timings, and real generated code |
| [3](ch03-map-of-the-codebase/README.md) | A map of the codebase | The five representations a program passes through, and which directory owns each one |

Every chapter that teaches a mechanism ends with runnable labs. Chapter 1 is not one of those — it sets out the contract and has nothing to run — so the labs in Part 0 live in `ch02-*/labs/` and `ch03-*/labs/`, five of them, and run straight from the built bundle:

```bash
npm run build
node docs/part0/ch02-setting-up/labs/01-first-look.mjs
```

## What you need to know already

The book assumes you can program and that you know roughly what a neural network is. It does **not** assume you know anything about compilers. But it is honest about the mathematics it uses, because the later parts use real mathematics and a reader who expects none will be ambushed around Part V:

| You will need | Where it starts to matter | If you are rusty |
|---|---|---|
| The chain rule, and partial derivatives of a vector-valued function | Part V — the whole part is the chain rule as a program transformation | Any single-variable calculus text plus the definition of a Jacobian |
| Matrix multiplication, transpose, and what a Jacobian is | Part V, Part VI | Enough to read `Jᵀv` and know which side is which |
| Floating-point arithmetic: rounding, `ε`, why `(a+b)+c ≠ a+(b+c)` | Part IV Chapter 20 onward, and everywhere after | Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* |
| Integer division, modulo, and the difference between flooring and truncating them | Part VI — index arithmetic is built from them | Chapter 35 defines what it uses, but assumes the idea is familiar |
| Big-O reasoning and the idea of a lattice or a fixed point | Part III, Part VI | Chapter 16 and Chapter 36 define what they use |

None of this is assumed on page one; each is introduced where it is needed. But Part V and Part VI are genuinely harder than Part I, and the step up is real rather than a failure of attention.

If you only have an hour, read Chapter 1 and run the first lab in Chapter 2. That is enough to know whether the rest of the book is for you.
