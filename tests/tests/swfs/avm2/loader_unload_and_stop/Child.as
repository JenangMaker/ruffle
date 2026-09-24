package {
    import flash.display.Sprite;
    import flash.events.Event;

    // Loaded by Test.as. Counts its own per-frame events so the parent can
    // check whether they keep firing after Loader.unloadAndStop().
    public class Child extends Sprite {
        public var inner:Sprite;
        public var rootTicks:int = 0;
        public var innerTicks:int = 0;
        public var innerExitTicks:int = 0;

        public function Child() {
            inner = new Sprite();
            addChild(inner);

            addEventListener(Event.ENTER_FRAME, onRootEnterFrame);
            inner.addEventListener(Event.ENTER_FRAME, onInnerEnterFrame);
            inner.addEventListener(Event.EXIT_FRAME, onInnerExitFrame);
        }

        private function onRootEnterFrame(e:Event):void {
            rootTicks++;
        }

        private function onInnerEnterFrame(e:Event):void {
            innerTicks++;
        }

        private function onInnerExitFrame(e:Event):void {
            innerExitTicks++;
        }
    }
}
