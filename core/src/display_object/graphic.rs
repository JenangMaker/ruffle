use crate::avm1::Object as Avm1Object;
use crate::avm2::{
    Activation as Avm2Activation, Avm2, ClassObject as Avm2ClassObject,
    StageObject as Avm2StageObject,
};
use crate::context::{RenderContext, UpdateContext};
use crate::display_object::{BoundsMode, DisplayObjectBase};
use crate::drawing::Drawing;
use crate::library::MovieLibrarySource;
use crate::prelude::*;
use crate::tag_utils::{SwfMovie, SwfSlice};
use crate::tessellation_cache::TessellationCache;
use crate::vminterface::Instantiator;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use ruffle_render::backend::ShapeHandle;
use ruffle_render::commands::CommandHandler;
use std::cell::{OnceCell, RefCell, RefMut};
use std::sync::Arc;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct Graphic<'gc>(Gc<'gc, GraphicData<'gc>>);

impl<'gc> Graphic<'gc> {
    /// The data this graphic shares with every instance made from it; see
    /// `Character::liveness_handle`.
    pub fn shared_gc(self) -> Gc<'gc, ()> {
        Gc::erase(self.0.shared.get())
    }
}

impl fmt::Debug for Graphic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graphic")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct GraphicData<'gc> {
    base: DisplayObjectBase<'gc>,
    shared: Lock<Gc<'gc, GraphicShared>>,
    class: Lock<Option<Avm2ClassObject<'gc>>>,
    avm2_object: Lock<Option<Avm2StageObject<'gc>>>,
    /// This is lazily allocated on demand, to make `GraphicData` smaller in the common case.
    #[collect(require_static)]
    drawing: OnceCell<Box<RefCell<Drawing>>>,
}

impl<'gc> Graphic<'gc> {
    /// Construct a `Graphic` from it's associated `Shape` tag.
    ///
    /// VibeSkua: with `source` (the tag's data and DefineShape version) the
    /// parsed records are dropped and parsed again when first needed (to draw
    /// or hit-test the shape): kept for every shape of every loaded movie they
    /// took about 100 MB in an AQW player, mostly for shapes never shown.
    pub fn from_swf_tag(
        context: &mut UpdateContext<'gc>,
        swf_shape: swf::Shape,
        movie: Arc<SwfMovie>,
        source: Option<(SwfSlice, u8)>,
    ) -> Self {
        // VibeSkua: not tessellated here. Drawing goes through the per-scale
        // cache (get_or_retessellate_handle), which tessellates on first draw
        // anyway, so a tessellation made at load was never drawn: it only held
        // memory, for every shape of every loaded movie (about 280 MB per AQW
        // player), and a player that never draws (headless) needs none.
        let shared = GraphicShared {
            id: swf_shape.id,
            shape_bounds: swf_shape.shape_bounds,
            edge_bounds: swf_shape.edge_bounds,
            renderable: true,
            shape: if source.is_some() {
                OnceCell::new()
            } else {
                OnceCell::from(swf_shape)
            },
            source,
            movie,
            scaled_handle: RefCell::new(TessellationCache::new()),
            last_drawn_ms: Default::default(),
        };

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
            },
        ))
    }

    /// Construct an empty `Graphic`.
    pub fn empty(context: &mut UpdateContext<'gc>) -> Self {
        let shared = GraphicShared {
            id: 0,
            shape_bounds: Default::default(),
            edge_bounds: Default::default(),
            renderable: false,
            shape: OnceCell::from(empty_shape(0)),
            source: None,
            movie: context.root_swf.clone(),
            scaled_handle: RefCell::new(TessellationCache::new()),
            last_drawn_ms: Default::default(),
        };

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
            },
        ))
    }

    pub fn instantiate(self, mc: &Mutation<'gc>) -> Self {
        Self(Gc::new(mc, (*self.0).clone()))
    }

    pub fn drawing_mut(&self) -> RefMut<'_, Drawing> {
        self.0.drawing.get_or_init(Default::default).borrow_mut()
    }

    pub fn set_avm2_class(self, mc: &Mutation<'gc>, class: Avm2ClassObject<'gc>) {
        unlock!(Gc::write(mc, self.0), GraphicData, class).set(Some(class));
    }

    fn set_shared(self, mc: &Mutation<'gc>, shared: Gc<'gc, GraphicShared>) {
        unlock!(Gc::write(mc, self.0), GraphicData, shared).set(shared);
    }

    /// Drops this shape's tessellations if none was drawn in the last
    /// `idle_ms` (they are made again on the next draw); returns whether any
    /// were dropped. A shape drawn once, as AQW's avatar snapshots
    /// (BitmapData.draw) are, otherwise keeps its meshes and their GPU
    /// buffers for as long as its movie's library lives.
    pub fn expire_meshes(self, now_ms: u64, idle_ms: u64) -> bool {
        let shared = self.0.shared.get();
        now_ms.saturating_sub(shared.last_drawn_ms.get()) >= idle_ms
            && shared.scaled_handle.borrow_mut().clear()
    }

    /// Returns the best shape handle for the current scale, retessellating if necessary
    /// (`None` if the movie's library is gone).
    fn get_or_retessellate_handle(
        self,
        context: &mut RenderContext,
        current_scale: f32,
    ) -> Option<ShapeHandle> {
        // Since graphics are created from a shared shape, we may be able to reuse a
        // cached tessellation from another instance at a similar scale.
        let shared = self.0.shared.get();

        shared.last_drawn_ms.set(mesh_clock_ms());
        {
            let mut cache = shared.scaled_handle.borrow_mut();
            if let Some(handle) = cache.find_near_and_touch(current_scale) {
                // Found a cached handle at a similar scale; reuse it.
                return Some(handle);
            }
        }

        // Retessellate at the new scale
        let library = context.library.library_for_movie(shared.movie.clone());
        if let Some(library) = library {
            let new_handle = context.renderer.register_shape_with_scale(
                shared.shape().into(),
                &MovieLibrarySource { library },
                current_scale,
            );

            {
                let mut cache = shared.scaled_handle.borrow_mut();
                tracing::debug!(
                    "Graphic id={} retessellated: new_scale={:.2}, cache_size={}",
                    shared.id,
                    current_scale,
                    cache.len()
                );
                cache.insert(current_scale, new_handle.clone());
            }

            Some(new_handle)
        } else {
            None
        }
    }
}

impl<'gc> TDisplayObject<'gc> for Graphic<'gc> {
    fn base(self) -> Gc<'gc, DisplayObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    fn id(self) -> CharacterId {
        self.0.shared.get().id
    }

    fn self_bounds(self, mode: BoundsMode) -> Rectangle<Twips> {
        let include_strokes = mode.includes_strokes();

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().self_bounds(include_strokes)
        } else if include_strokes {
            self.0.shared.get().shape_bounds
        } else {
            self.0.shared.get().edge_bounds
        }
    }

    fn construct_frame(self, context: &mut UpdateContext<'gc>) {
        if self.movie().is_action_script_3() && self.object2().is_none() {
            let class_object = self
                .0
                .class
                .get()
                .unwrap_or_else(|| context.avm2.classes().shape);

            let mut activation = Avm2Activation::from_nothing(context);

            match Avm2StageObject::for_display_object_childless(
                &mut activation,
                self.into(),
                class_object,
            ) {
                Ok(object) => self.set_object2(activation.context, object),
                Err(err) => {
                    Avm2::uncaught_error(
                        &mut activation,
                        Some(self.into()),
                        err,
                        "Error running AVM2 construction for shape",
                    );
                }
            }

            self.on_construction_complete(context);
        }
    }

    fn replace_with(self, context: &mut UpdateContext<'gc>, id: CharacterId) {
        // Static assets like Graphics can replace themselves via a PlaceObject tag with PlaceObjectAction::Replace.
        // This does not create a new instance, but instead swaps out the underlying static data to point to the new art.
        if let Some(new_graphic) = context
            .library
            .library_for_movie_mut(self.movie())
            .get_graphic(id)
        {
            self.set_shared(context.gc(), new_graphic.0.shared.get());
        } else {
            tracing::warn!("PlaceObject: expected Graphic at character ID {}", id);
        }
        self.invalidate_cached_bitmap();
    }

    fn render_self(self, context: &mut RenderContext) {
        if !context.is_offscreen
            && !self
                .world_bounds(BoundsMode::Engine)
                .intersects(&context.stage.view_bounds())
        {
            // Off-screen; culled
            return;
        }

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().render(context);
        } else if self.0.shared.get().renderable {
            let transform = context.transform_stack.transform();

            // Calculate the current scale from the transform, to determine if
            // we can reuse a cached tessellation or need to retessellate.
            let matrix = &transform.matrix;
            let scale_x = f32::abs(matrix.a + matrix.c);
            let scale_y = f32::abs(matrix.b + matrix.d);
            let current_scale = ((scale_x * scale_x + scale_y * scale_y) / 2.0).sqrt();

            if let Some(handle) = self.get_or_retessellate_handle(context, current_scale) {
                context.commands.render_shape(handle, transform)
            }
        }
    }

    fn hit_test_shape(
        self,
        _context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        options: HitTestOptions,
    ) -> bool {
        // Transform point to local coordinates and test.
        if (!options.contains(HitTestOptions::SKIP_INVISIBLE) || self.visible())
            && self.world_bounds(BoundsMode::Engine).contains(point)
        {
            let Some(local_matrix) = self.global_to_local_matrix() else {
                return false;
            };
            let point = local_matrix * point;
            if let Some(drawing) = self.0.drawing.get() {
                if drawing.borrow().hit_test(point, &local_matrix) {
                    return true;
                }
            } else {
                let shared = self.0.shared.get();
                return ruffle_render::shape_utils::shape_hit_test(
                    shared.shape(),
                    point,
                    &local_matrix,
                );
            }
        }

        false
    }

    fn post_instantiation(
        self,
        context: &mut UpdateContext<'gc>,
        _init_object: Option<Avm1Object<'gc>>,
        _instantiated_by: Instantiator,
        _run_frame: bool,
    ) {
        if self.movie().is_action_script_3() {
            self.set_default_instance_name(context);
        }
    }

    fn movie(self) -> Arc<SwfMovie> {
        self.0.shared.get().movie.clone()
    }

    fn object1(self) -> Option<Avm1Object<'gc>> {
        None
    }

    fn object2(self) -> Option<Avm2StageObject<'gc>> {
        self.0.avm2_object.get()
    }

    fn set_object2(self, context: &mut UpdateContext<'gc>, to: Avm2StageObject<'gc>) {
        let mc = context.gc();
        unlock!(Gc::write(mc, self.0), GraphicData, avm2_object).set(Some(to));
    }

    fn as_drawing(&self) -> Option<RefMut<'_, Drawing>> {
        Some(self.drawing_mut())
    }
}

/// Data shared between all instances of a Graphic.
#[derive(Collect)]
#[collect(require_static)]
struct GraphicShared {
    id: CharacterId,
    /// The shape's records; empty until first needed when `source` is set.
    shape: OnceCell<swf::Shape>,
    /// The DefineShape tag's data and version, to parse `shape` from.
    source: Option<(SwfSlice, u8)>,
    /// False for the empty graphic, which draws nothing.
    renderable: bool,
    shape_bounds: Rectangle<Twips>,
    edge_bounds: Rectangle<Twips>,
    movie: Arc<SwfMovie>,
    #[collect(require_static)]
    scaled_handle: RefCell<TessellationCache>,
    /// When a tessellation was last drawn (`mesh_clock_ms`), for
    /// `expire_meshes`.
    last_drawn_ms: std::cell::Cell<u64>,
}

/// Milliseconds since the first call, for when meshes were last drawn.
pub(crate) fn mesh_clock_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

impl GraphicShared {
    /// The shape's records, parsed from the tag the first time they are needed.
    fn shape(&self) -> &swf::Shape {
        self.shape.get_or_init(|| {
            self.source
                .as_ref()
                .and_then(|(slice, version)| {
                    slice
                        .read_from(0)
                        .read_define_shape(*version)
                        .map_err(|e| tracing::warn!("Graphic {}: parsing the shape failed: {e}", self.id))
                        .ok()
                })
                .unwrap_or_else(|| empty_shape(self.id))
        })
    }
}

fn empty_shape(id: CharacterId) -> swf::Shape {
    swf::Shape {
        version: 32,
        id,
        shape_bounds: Default::default(),
        edge_bounds: Default::default(),
        flags: swf::ShapeFlag::empty(),
        styles: swf::ShapeStyles {
            fill_styles: Vec::new(),
            line_styles: Vec::new(),
        },
        shape: Vec::new(),
    }
}
