Compiled with Ruffle's bundled `tools/asc/asc.jar` against its own
`playerglobal_import.abc`:

    java -jar asc.jar -import playerglobal_import.abc -swf Test,100,100,30 Test.as

Runs with no renderer, so the backend cannot render offscreen. `output.txt`
reflects Flash Player's behaviour that `draw()` does not throw, so the script
continues; it was not captured from Flash Player itself.
