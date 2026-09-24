Compiled with Ruffle's bundled `tools/asc/asc.jar` against its own
`playerglobal_import.abc`, not with Flash Professional:

    java -jar asc.jar -import playerglobal_import.abc -swf Child,100,100,30 Child.as
    java -jar asc.jar -import playerglobal_import.abc -swf Test,100,100,30 Test.as

`output.txt` follows Flash Player's documented behaviour for `Loader.close()`
-- an in-progress load is cancelled -- but was not captured from Flash Player
itself.
