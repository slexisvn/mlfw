# Part 0 — Orientation

Three short chapters before the real material starts. They tell you what the book is, get code running on your machine, and hand you a map you will keep coming back to.

| Chapter | Title | What you get out of it |
|---|---|---|
| [1](ch01-what-this-book-is/README.md) | What this book is, and how to read it | The contract: who it is for, how every chapter is built, how to read it in the order that suits you |
| [2](ch02-setting-up/README.md) | Setting up | A working checkout, three labs that print real IR, real timings, and real generated code |
| [3](ch03-map-of-the-codebase/README.md) | A map of the codebase | The five representations a program passes through, and which directory owns each one |

Every chapter in this book ends with runnable labs. In Part 0 they live in `ch0N-*/labs/` and run straight from the built bundle:

```bash
npm run build
node docs/part0/ch02-setting-up/labs/01-first-look.mjs
```

If you only have ten minutes, read Chapter 1 and run the first lab in Chapter 2. That is enough to know whether the rest of the book is for you.
