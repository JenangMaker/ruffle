package {
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.system.System;
    import flash.utils.Dictionary;

    // Weak-keyed Dictionary entries go away once their key is otherwise
    // unreachable, including when the value refers back to the key. A value
    // is kept while its key lives, which can keep further keys alive.
    public class Test extends Sprite {
        private var weak:Dictionary = new Dictionary(true);
        private var strong:Dictionary = new Dictionary();
        private var kept:Object = {name: "kept"};
        private var anchor:Object = {name: "anchor"};
        private var frames:int = 0;

        public function Test() {
            weak[kept] = "kept value";
            weak[{name: "dropped"}] = "dropped value";

            // The value leads back to its own key, as in a per-object cache.
            var cycle:Object = {name: "cycle"};
            weak[cycle] = {owner: cycle};

            // a -> b, both reachable only through the dictionary.
            var a:Object = {name: "a"};
            var b:Object = {name: "b"};
            weak[a] = b;
            weak[b] = "b value";

            // anchor is live; its value keeps "linked" (another key) alive.
            var linked:Object = {name: "linked"};
            weak[anchor] = linked;
            weak[linked] = "linked value";

            weak["str"] = "string key";
            strong[{name: "strong"}] = 1;

            report("before");
            System.gc();
            addEventListener(Event.ENTER_FRAME, onFrame);
        }

        private function onFrame(e:Event):void {
            if (++frames == 1) {
                report("after gc");
                trace("anchor value is linked: " + (weak[anchor] === getKey("linked")));
                removeEventListener(Event.ENTER_FRAME, onFrame);
            }
        }

        private function getKey(name:String):Object {
            for (var k:* in weak) {
                if (!(k is String) && k.name == name) {
                    return k;
                }
            }
            return null;
        }

        private function report(label:String):void {
            var names:Array = [];
            for (var k:* in weak) {
                names.push(k is String ? k : k.name);
            }
            names.sort();
            var n:int = 0;
            for (var s:* in strong) {
                n++;
            }
            trace(label + ": weak [" + names.join(",") + "], strong " + n);
        }
    }
}
