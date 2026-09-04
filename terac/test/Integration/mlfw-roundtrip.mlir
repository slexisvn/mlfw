// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs demo --emit-mlir --out %t.demo.mlir --json %t.demo.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.demo.mlir -o %t.demo.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.demo.rt.mlir --data %t.demo.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs mlp --emit-mlir --out %t.mlp.mlir --json %t.mlp.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlp.mlir -o %t.mlp.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.mlp.rt.mlir --data %t.mlp.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs attention --emit-mlir --out %t.attn.mlir --json %t.attn.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.attn.mlir -o %t.attn.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.attn.rt.mlir --data %t.attn.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs rnn --emit-mlir --out %t.rnn.mlir --json %t.rnn.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.rnn.mlir -o %t.rnn.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rnn.rt.mlir --data %t.rnn.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs branch --emit-mlir --out %t.branch.mlir --json %t.branch.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.branch.mlir -o %t.branch.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.branch.rt.mlir --data %t.branch.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs integers --emit-mlir --out %t.ints.mlir --json %t.ints.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.ints.mlir -o %t.ints.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.ints.rt.mlir --data %t.ints.json %}

// The reader has to be able to fail. Given a module and a record for a
// different program it must say so rather than pass, or every line above is
// worth nothing.
// RUN: %if mlfw-oracle %{ not %node %tera_harness/mlfw_reader.mjs %t.demo.rt.mlir --data %t.mlp.json %}

// The bridge in the other direction: mlfw reads what MLIR wrote.
//
// mlfw-bridge.mlir and the tests beside it prove mlfw can write a module MLIR
// parses, verifies and runs. That is one of the two claims. The other -- that
// mlfw can read what MLIR writes -- has nothing to do with the first, and until
// this file nothing checked it: mlfw's parser was only ever fed mlfw's own
// printer's output, so every spelling the two share was tested and every one
// they do not was not.
//
// MLIR's own spellings are what these lines put through it, and each of them
// broke something real the first time it was run:
//
//   `%0:2 = tera.scan` and `%0#1`   a group of results and an index into it,
//                                   which mlfw never writes -- it names each
//                                   result on its own.
//   `-> tensor<f32>`                one result without parentheses. mlfw always
//                                   writes the parenthesised form.
//   `^bb0(%arg0: ...)` twice        MLIR names values per region and reuses
//                                   `%arg0` in both bodies of a branch. mlfw
//                                   read them into one flat table and reported
//                                   a value defined twice.
//   `dense<5.000000e-01>`           an exponent with a sign. mlfw's literal
//                                   scanner accepted `+` and not `-`, so this
//                                   parsed as NaN -- and since mlfw prints
//                                   small floats as `5e-7`, its own round trip
//                                   had the same hole.
//
// The last one is the shape of all four: a round trip that only ever sees one
// printer's output cannot find a disagreement between two of them. What makes
// this file able to is that nothing in it is written by hand -- the module is
// mlfw's, reprinted by MLIR, and the answer is mlfw's, computed before either
// printer ran.
