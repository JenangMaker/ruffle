Weak-keyed `Dictionary` entries are collected by `System.gc()` once their key
is otherwise unreachable, including when the value refers back to the key
(ephemeron semantics); a value is kept while its key is alive, and can keep
another key alive in turn.

Compiled from `Test.as` with `tools/asc` (`-swf Test,100,100,30`). The expected
output is Ruffle's: `System.gc()` only collects in the debug Flash Player,
and Flash does not collect key/value cycles like the `cycle` entry.
