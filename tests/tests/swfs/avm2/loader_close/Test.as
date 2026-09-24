package {
    import flash.display.Loader;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.events.IOErrorEvent;
    import flash.net.URLRequest;

    // Loader.close() must cancel a load that is still in flight: the content
    // is never instantiated and no completion or error events fire.
    //
    // Games recycle asset loaders this way -- close() then unloadAndStop() --
    // when an object that requested an asset goes away before it arrives.
    public class Test extends Sprite {
        private var loader:Loader = new Loader();
        private var frame:int = 0;

        public function Test() {
            loader.contentLoaderInfo.addEventListener(Event.INIT, function(e:Event):void {
                trace("init fired");
            });
            loader.contentLoaderInfo.addEventListener(Event.COMPLETE, function(e:Event):void {
                trace("complete fired");
            });
            loader.contentLoaderInfo.addEventListener(IOErrorEvent.IO_ERROR, function(e:Event):void {
                trace("ioError fired");
            });

            loader.load(new URLRequest("child.swf"));
            loader.close();
            trace("closed");

            addEventListener(Event.ENTER_FRAME, onFrame);
        }

        private function onFrame(e:Event):void {
            frame++;
            if (frame == 8) {
                trace("loader.content: " + loader.content);
                removeEventListener(Event.ENTER_FRAME, onFrame);
                trace("done");
            }
        }
    }
}
