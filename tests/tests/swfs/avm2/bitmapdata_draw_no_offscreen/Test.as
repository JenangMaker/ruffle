package {
    import flash.display.BitmapData;
    import flash.display.Sprite;

    // BitmapData.draw() must not abort the calling script when the render
    // backend cannot render offscreen (this test runs without a renderer).
    // Flash Player never throws here; the code after the call has to run.
    public class Test extends Sprite {
        public function Test() {
            var shape:Sprite = new Sprite();
            shape.graphics.beginFill(0xFF0000);
            shape.graphics.drawRect(0, 0, 10, 10);
            shape.graphics.endFill();

            var bd:BitmapData = new BitmapData(10, 10, true, 0);
            trace("before draw");
            bd.draw(shape);
            trace("after draw");
            trace("done");
        }
    }
}
