package {
    import flash.display.Loader;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.net.URLRequest;

    // Loader.unloadAndStop() must stop the unloaded content's per-frame code,
    // not merely detach it.
    //
    // enterFrame / exitFrame are broadcast events: they reach every display
    // object with a listener, on the display list or not. So `content` is
    // deliberately kept reachable here -- exactly what a game does when it
    // caches a reference to the previous room -- and the test checks that its
    // handlers stopped anyway. A plain unload() would leave them firing.
    public class Test extends Sprite {
        private var loader:Loader = new Loader();
        private var content:Object;
        private var frame:int = 0;
        private var rootAt:int;
        private var innerAt:int;
        private var innerExitAt:int;

        public function Test() {
            loader.contentLoaderInfo.addEventListener(Event.COMPLETE, onComplete);
            loader.load(new URLRequest("child.swf"));
        }

        private function onComplete(e:Event):void {
            content = loader.content;
            trace("loaded");
            addEventListener(Event.ENTER_FRAME, onFrame);
        }

        private function onFrame(e:Event):void {
            frame++;

            if (frame == 3) {
                trace("child ran before unload: " + (content.rootTicks > 0 && content.innerTicks > 0));
                trace("before: root has enterFrame: " + content.hasEventListener(Event.ENTER_FRAME));
                trace("before: inner has enterFrame: " + content.inner.hasEventListener(Event.ENTER_FRAME));
                trace("before: inner has exitFrame: " + content.inner.hasEventListener(Event.EXIT_FRAME));

                loader.unloadAndStop();

                rootAt = content.rootTicks;
                innerAt = content.innerTicks;
                innerExitAt = content.innerExitTicks;

                trace("loader.content after: " + loader.content);
                trace("after: root has enterFrame: " + content.hasEventListener(Event.ENTER_FRAME));
                trace("after: inner has enterFrame: " + content.inner.hasEventListener(Event.ENTER_FRAME));
                trace("after: inner has exitFrame: " + content.inner.hasEventListener(Event.EXIT_FRAME));
            }

            if (frame == 7) {
                trace("root enterFrame ticks since unloadAndStop: " + (content.rootTicks - rootAt));
                trace("inner enterFrame ticks since unloadAndStop: " + (content.innerTicks - innerAt));
                trace("inner exitFrame ticks since unloadAndStop: " + (content.innerExitTicks - innerExitAt));
                removeEventListener(Event.ENTER_FRAME, onFrame);
                trace("done");
            }
        }
    }
}
