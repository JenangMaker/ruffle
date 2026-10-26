use super::target::PoolOrArcTexture;
use crate::backend::RenderTargetMode;
use crate::blend::TrivialBlend;
use crate::blend::{BlendType, ComplexBlend};
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::Surface;
use crate::surface::target::CommandTarget;
use crate::{Descriptors, MaskState, Pipelines, PosUvVertex, Transforms, as_texture};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::lines::{emulate_line, emulate_line_rect};
use ruffle_render::matrix::Matrix;
use ruffle_render::pixel_bender::PixelBenderShaderHandle;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::mem;
use swf::{BlendMode, Color, ColorTransform, Twips};
use wgpu::Backend;
use wgpu_profiler::Scope;

pub struct CommandRenderer<'encoder> {
    pipelines: &'encoder Pipelines,
    descriptors: &'encoder Descriptors,
    num_masks: u32,
    mask_state: MaskState,
    needs_stencil: bool,
    dynamic_vertex_buffer: &'encoder wgpu::Buffer,
    state: PassStateCache,
}

/// VibeSkua: what this pass last set, so the same pipeline, bind group,
/// buffers or stencil reference are not set again for the next draw. Every
/// shape draw set all of them: ~15,500 commands a frame in AQW's Battleon
/// with filters off, and wgpu checks and records each one when the command
/// buffer is finished (most of a frame's drawing time on Windows). Starts
/// empty for each pass.
#[derive(Default)]
struct PassStateCache {
    pipeline: std::cell::Cell<usize>,
    group2: std::cell::Cell<usize>,
    vertex: std::cell::Cell<(usize, u64, u64)>,
    index: std::cell::Cell<(usize, u64, u64)>,
    stencil: std::cell::Cell<Option<u32>>,
}

impl<'encoder> CommandRenderer<'encoder> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: &'encoder Pipelines,
        descriptors: &'encoder Descriptors,
        dynamic_vertex_buffer: &'encoder wgpu::Buffer,
        num_masks: u32,
        mask_state: MaskState,
        needs_stencil: bool,
    ) -> Self {
        Self {
            pipelines,
            num_masks,
            mask_state,
            descriptors,
            needs_stencil,
            dynamic_vertex_buffer,
            state: PassStateCache::default(),
        }
    }

    fn set_pipeline(&self, render_pass: &mut wgpu::RenderPass<'encoder>, pipeline: &'encoder wgpu::RenderPipeline) {
        let key = pipeline as *const _ as usize;
        if self.state.pipeline.replace(key) != key {
            render_pass.set_pipeline(pipeline);
        }
    }

    fn set_group2(&self, render_pass: &mut wgpu::RenderPass<'encoder>, bind_group: &'encoder wgpu::BindGroup) {
        let key = bind_group as *const _ as usize;
        if self.state.group2.replace(key) != key {
            render_pass.set_bind_group(2, bind_group, &[]);
        }
    }

    fn set_stencil(&self, render_pass: &mut wgpu::RenderPass<'encoder>, reference: u32) {
        if self.state.stencil.replace(Some(reference)) != Some(reference) {
            render_pass.set_stencil_reference(reference);
        }
    }

    fn set_buffers(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: &'encoder wgpu::Buffer,
        vertex_range: std::ops::Range<wgpu::BufferAddress>,
        indices: &'encoder wgpu::Buffer,
        index_range: std::ops::Range<wgpu::BufferAddress>,
    ) {
        let key = (vertices as *const _ as usize, vertex_range.start, vertex_range.end);
        if self.state.vertex.replace(key) != key {
            render_pass.set_vertex_buffer(0, vertices.slice(vertex_range));
        }
        let key = (indices as *const _ as usize, index_range.start, index_range.end);
        if self.state.index.replace(key) != key {
            render_pass.set_index_buffer(indices.slice(index_range), wgpu::IndexFormat::Uint32);
        }
    }

    pub fn execute(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        command: &'encoder DrawCommand,
    ) {
        if self.needs_stencil {
            match self.mask_state {
                MaskState::NoMask => {}
                MaskState::DrawMaskStencil => {
                    self.set_stencil(render_pass, self.num_masks - 1);
                }
                MaskState::DrawMaskedContent => {
                    self.set_stencil(render_pass, self.num_masks);
                }
                MaskState::ClearMaskStencil => {
                    self.set_stencil(render_pass, self.num_masks);
                }
            }
        }

        match command {
            DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset,
                smoothing,
                blend_mode,
                render_stage3d,
            } => self.render_bitmap(
                render_pass,
                bitmap,
                *instance_index,
                *smoothing,
                *blend_mode,
                *render_stage3d,
                *vertex_offset,
            ),
            DrawCommand::RenderTexture {
                _texture,
                binds,
                instance_index,
                blend_mode,
            } => self.render_texture(render_pass, *instance_index, binds, *blend_mode),
            DrawCommand::RenderAtlas {
                binds,
                instance_index,
                vertex_offset,
                blend_mode,
                ..
            } => {
                self.prep_bitmap(render_pass, binds, *blend_mode, false);
                self.draw(
                    render_pass,
                    self.dynamic_vertex_buffer,
                    *vertex_offset..self.dynamic_vertex_buffer.size(),
                    &self.descriptors.quad.indices,
                    0..self.descriptors.quad.indices.size(),
                    6,
                    *instance_index,
                );
            }
            DrawCommand::RenderShape {
                shape,
                instance_index,
            } => self.render_shape(render_pass, shape, *instance_index),
            DrawCommand::DrawRect { instance_index } => {
                self.draw_rect(render_pass, *instance_index)
            }
            DrawCommand::DrawLine { instance_index } => {
                self.draw_lines::<false>(render_pass, *instance_index)
            }
            DrawCommand::DrawLineRect { instance_index } => {
                self.draw_lines::<true>(render_pass, *instance_index)
            }
            DrawCommand::PushMask => self.push_mask(render_pass),
            DrawCommand::ActivateMask => self.activate_mask(render_pass),
            DrawCommand::DeactivateMask => self.deactivate_mask(render_pass),
            DrawCommand::PopMask => self.pop_mask(render_pass),
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                instance_index,
            } => self.render_alpha_mask(render_pass, maskee, mask, binds, *instance_index),
        }
    }

    pub fn prep_color(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            self.set_pipeline(render_pass, self.pipelines.color.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline(render_pass, self.pipelines.color.stencilless_pipeline());
        }
    }

    pub fn prep_lines(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            self.set_pipeline(render_pass, self.pipelines.lines.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline(render_pass, self.pipelines.lines.stencilless_pipeline());
        }
    }

    pub fn prep_gradient(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            self.set_pipeline(render_pass, self.pipelines.gradients.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline(render_pass, self.pipelines.gradients.stencilless_pipeline());
        }

        self.set_group2(render_pass, bind_group);
    }

    pub fn prep_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                self.set_pipeline(render_pass, &self.pipelines.bitmap_opaque_dummy_stencil);
            }
            (true, false) => {
                self.set_pipeline(render_pass, self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state));
            }
            (false, true) => {
                self.set_pipeline(render_pass, &self.pipelines.bitmap_opaque);
            }
            (false, false) => {
                self.set_pipeline(render_pass, self.pipelines.bitmap[blend_mode].stencilless_pipeline());
            }
        }

        self.set_group2(render_pass, bind_group);
    }

    pub fn prep_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            self.set_pipeline(render_pass, self.pipelines.alpha_mask.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline(render_pass, self.pipelines.alpha_mask.stencilless_pipeline());
        }

        self.set_group2(render_pass, bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: &'encoder wgpu::Buffer,
        vertex_range: std::ops::Range<wgpu::BufferAddress>,
        indices: &'encoder wgpu::Buffer,
        index_range: std::ops::Range<wgpu::BufferAddress>,
        num_indices: u32,
        instance_index: u32,
    ) {
        self.set_buffers(render_pass, vertices, vertex_range, indices, index_range);

        render_pass.draw_indexed(0..num_indices, 0, instance_index..(instance_index + 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bitmap: &'encoder BitmapHandle,
        instance_index: u32,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
        vertex_offset: Option<wgpu::BufferAddress>,
    ) {
        let texture = as_texture(bitmap);

        let descriptors = self.descriptors;
        let bind = texture.bind_group(
            false,
            smoothing,
            &descriptors.device,
            &descriptors.bind_layouts.bitmap,
            bitmap.clone(),
            &descriptors.bitmap_samplers,
        );
        self.prep_bitmap(render_pass, &bind.bind_group, blend_mode, render_stage3d);

        let (vertex_buffer, vertex_range) = if let Some(vertex_offset) = vertex_offset {
            (
                self.dynamic_vertex_buffer,
                vertex_offset..self.dynamic_vertex_buffer.size(),
            )
        } else {
            let quad = &self.descriptors.quad.vertices_pos_uv;
            (quad, 0..quad.size())
        };

        self.draw(
            render_pass,
            vertex_buffer,
            vertex_range,
            &self.descriptors.quad.indices,
            0..self.descriptors.quad.indices.size(),
            6,
            instance_index,
        );
    }

    pub fn render_texture(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
    ) {
        self.prep_bitmap(render_pass, bind_group, blend_mode, false);

        self.draw(
            render_pass,
            &self.descriptors.quad.vertices_pos_uv,
            0..self.descriptors.quad.vertices_pos_uv.size(),
            &self.descriptors.quad.indices,
            0..self.descriptors.quad.indices.size(),
            6,
            instance_index,
        );
    }

    pub fn render_shape(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        shape: &'encoder ShapeHandle,
        instance_index: u32,
    ) {
        let mesh = as_mesh(shape);
        for draw in &mesh.draws {
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                // Omit strokes when drawing a mask stencil.
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            match &draw.draw_type {
                DrawType::Color => {
                    self.prep_color(render_pass);
                }
                DrawType::Gradient { bind_group, .. } => {
                    self.prep_gradient(render_pass, bind_group);
                }
                DrawType::Bitmap { binds, .. } => {
                    self.prep_bitmap(render_pass, &binds.bind_group, TrivialBlend::Normal, false);
                }
            }

            self.draw(
                render_pass,
                &mesh.vertex_buffer,
                draw.vertices.clone(),
                &mesh.index_buffer,
                draw.indices.clone(),
                num_indices,
                instance_index,
            );
        }
    }

    pub fn render_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'encoder wgpu::BindGroup,
        instance_index: u32,
    ) {
        if cfg!(feature = "render_debug_labels") {
            render_pass.push_debug_group("render_alpha_mask");
        }

        self.prep_alpha_mask(render_pass, bind_group);

        self.draw(
            render_pass,
            &self.descriptors.quad.vertices_pos,
            0..self.descriptors.quad.vertices_pos.size(),
            &self.descriptors.quad.indices,
            0..self.descriptors.quad.indices.size(),
            6,
            instance_index,
        );

        if cfg!(feature = "render_debug_labels") {
            render_pass.pop_debug_group();
        }
    }

    pub fn draw_rect(&self, render_pass: &mut wgpu::RenderPass<'encoder>, instance_index: u32) {
        self.prep_color(render_pass);

        self.draw(
            render_pass,
            &self.descriptors.quad.vertices_pos_color,
            0..self.descriptors.quad.vertices_pos_color.size(),
            &self.descriptors.quad.indices,
            0..self.descriptors.quad.indices.size(),
            6,
            instance_index,
        );
    }

    pub fn draw_lines<const RECT: bool>(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
    ) {
        self.prep_lines(render_pass);

        self.draw(
            render_pass,
            &self.descriptors.quad.vertices_pos_color,
            0..self.descriptors.quad.vertices_pos_color.size(),
            if RECT {
                &self.descriptors.quad.indices_line_rect
            } else {
                &self.descriptors.quad.indices_line
            },
            0..if RECT {
                self.descriptors.quad.indices_line_rect.size()
            } else {
                self.descriptors.quad.indices_line.size()
            },
            if RECT { 5 } else { 2 },
            instance_index,
        );
    }

    pub fn push_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        self.set_stencil(render_pass, self.num_masks - 1);
    }

    pub fn activate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        self.set_stencil(render_pass, self.num_masks);
    }

    pub fn deactivate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        self.set_stencil(render_pass, self.num_masks);
    }

    pub fn pop_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        self.set_stencil(render_pass, self.num_masks);
        if self.num_masks == 0 {
            self.mask_state = MaskState::NoMask;
        } else {
            self.mask_state = MaskState::DrawMaskedContent;
        };
    }

    pub fn num_masks(&self) -> u32 {
        self.num_masks
    }

    pub fn mask_state(&self) -> MaskState {
        self.mask_state
    }
}

pub enum Chunk {
    Draw {
        chunk: Vec<DrawCommand>,
        needs_stencil: bool,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
    },
    Blend {
        texture: PoolOrArcTexture,
        blend_mode: ChunkBlendMode,
        needs_stencil: bool,
        /// The area the blend can change; None for all of the target.
        rect: Option<BlendRect>,
    },
}

/// VibeSkua: the part of a target a blend works on, in its pixels.
///
/// Each blended object (Multiply, Overlay, Screen, Add, ...) was drawn into a
/// fresh texture the size of the whole target, and a complex blend copied all
/// of the parent and ran its shader over all of it. AQW's Battleon blends
/// about 240 small objects a frame on a 1000 x 666 stage: ~330 full-size
/// passes, 100-200 ms of an integrated GPU a frame (5 fps). Now the object is
/// drawn into a texture of its own bounds, shifted there, and composited back
/// over that area only. Outside the object the result is the parent as it
/// was (transparent pixels add nothing, and complex blend shaders discard
/// them), so the picture is the same.
#[derive(Clone, Copy, Debug)]
pub struct BlendRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Screen-space bounds of drawn commands, in target pixels.
struct CommandBounds {
    min: (f32, f32),
    max: (f32, f32),
}

impl CommandBounds {
    fn new() -> Self {
        Self {
            min: (f32::INFINITY, f32::INFINITY),
            max: (f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    fn add_rect(&mut self, matrix: &Matrix, [x0, y0, x1, y1]: [f32; 4]) {
        let (tx, ty) = (matrix.tx.to_pixels() as f32, matrix.ty.to_pixels() as f32);
        for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let px = matrix.a * x + matrix.c * y + tx;
            let py = matrix.b * x + matrix.d * y + ty;
            self.min = (self.min.0.min(px), self.min.1.min(py));
            self.max = (self.max.0.max(px), self.max.1.max(py));
        }
    }

    /// Adds what `commands` draw; false if that is not known (3D, Stage3D).
    fn add(&mut self, commands: &[Command]) -> bool {
        for command in commands {
            let known = match command {
                Command::RenderShape { shape, transform } => {
                    transform.perspective_projection.is_none()
                        && as_mesh(shape).bounds.is_none_or(|bounds| {
                            self.add_rect(&transform.matrix, bounds);
                            true
                        })
                }
                Command::RenderBitmap {
                    transform, region, ..
                } => {
                    let matrix = transform.matrix
                        * Matrix::scale(region.width() as f32, region.height() as f32);
                    self.add_rect(&matrix, [0.0, 0.0, 1.0, 1.0]);
                    transform.perspective_projection.is_none()
                }
                Command::RenderStage3D { .. } => false,
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => self.add(&maskee_commands.commands) && self.add(&mask_commands.commands),
                Command::DrawRect { matrix, .. }
                | Command::DrawLine { matrix, .. }
                | Command::DrawLineRect { matrix, .. } => {
                    self.add_rect(matrix, [0.0, 0.0, 1.0, 1.0]);
                    true
                }
                Command::PushMask
                | Command::ActivateMask
                | Command::DeactivateMask
                | Command::PopMask => true,
                Command::Blend(commands, _) => self.add(&commands.commands),
            };
            if !known {
                return false;
            }
        }
        true
    }
}

/// VibeSkua: blend groups drawn together on one atlas page.
///
/// Each Add, Screen, Subtract or Multiply group was drawn into a texture of
/// its own in a render pass of its own, then composited onto its parent: about
/// 200 passes a frame in AQW's Battleon with filters on. wgpu's cost to record
/// a frame grows with its render passes (~40 us each on Windows, ~600 passes,
/// most of a drawn frame). Groups are now placed side by side on a shared page
/// (rows, in the 64-pixel steps their sizes already have) and all drawn in one
/// pass, right before the chunk that composites them from their places. The
/// picture is the same: each group is drawn alone on transparent pixels, as on
/// its own texture.
///
/// Complex blends (Overlay, Hard Light, Lighten, ...) read their parent, so
/// they still composite one by one, but their groups are drawn on the page too
/// and copied from it into textures of their own (copies are not passes; the
/// blend shaders read the same pixels as before). The chunks that composite
/// them wait (see `WgpuCommandHandler::held`) until the page is drawn.
const ATLAS_SIZE: u32 = 2048;

/// The most texture memory the waiting complex blends may hold before the page
/// is drawn early (their textures stay out of the pool until then).
const ATLAS_HELD_BYTES: u64 = 64 << 20;

struct AtlasBatch {
    page: std::sync::Arc<PoolOrArcTexture>,
    binds: wgpu::BindGroup,
    commands: Vec<Command>,
    /// Complex blend groups to copy out once the page is drawn: their place
    /// (x, y, width, height) and their own texture.
    copies: Vec<(u32, u32, u32, u32, wgpu::Texture)>,
    copied_bytes: u64,
    // Rows: the next place in the current row, and the row's height.
    x: u32,
    y: u32,
    row_height: u32,
}

impl AtlasBatch {
    fn new(descriptors: &Descriptors, texture_pool: &mut TexturePool) -> Self {
        let texture = texture_pool.get_texture(
            descriptors,
            wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
        );
        let page = std::sync::Arc::new(PoolOrArcTexture::Pool(texture));
        let binds = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &descriptors.bind_layouts.bitmap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(page.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
                label: None,
            });
        Self {
            page,
            binds,
            commands: Vec::new(),
            copies: Vec::new(),
            copied_bytes: 0,
            x: 0,
            y: 0,
            row_height: 0,
        }
    }

    /// Where a group of this size would go, if it still fits.
    fn place(&self, width: u32, height: u32) -> Option<(u32, u32)> {
        if self.x + width <= ATLAS_SIZE && self.y + height <= ATLAS_SIZE {
            Some((self.x, self.y))
        } else {
            let y = self.y + self.row_height;
            (width <= ATLAS_SIZE && y + height <= ATLAS_SIZE).then_some((0, y))
        }
    }

    /// Takes the place `place` gave.
    fn take(&mut self, width: u32, height: u32) {
        if self.x + width > ATLAS_SIZE || self.y + height > ATLAS_SIZE {
            self.y += self.row_height;
            self.x = 0;
            self.row_height = 0;
        }
        self.x += width;
        self.row_height = self.row_height.max(height);
    }
}

/// Whether a group draws a blend of its own that needs its target's pixels
/// (it could not share a page with others then).
fn has_complex_blend(commands: &[Command]) -> bool {
    commands.iter().any(|command| match command {
        Command::Blend(_, mode) => !matches!(BlendType::from(mode.clone()), BlendType::Trivial(_)),
        Command::RenderAlphaMask { .. } => true,
        _ => false,
    })
}

/// Moves what `commands` draw by (-dx, -dy), onto a blend texture whose
/// top left corner is at (dx, dy) of its parent.
fn translate_commands(commands: &mut [Command], dx: Twips, dy: Twips) {
    let shift = |matrix: &mut Matrix| {
        matrix.tx = Twips::new(matrix.tx.get() - dx.get());
        matrix.ty = Twips::new(matrix.ty.get() - dy.get());
    };
    for command in commands {
        match command {
            Command::RenderBitmap { transform, .. }
            | Command::RenderStage3D { transform, .. }
            | Command::RenderShape { transform, .. } => shift(&mut transform.matrix),
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => {
                translate_commands(&mut maskee_commands.commands, dx, dy);
                translate_commands(&mut mask_commands.commands, dx, dy);
            }
            Command::DrawRect { matrix, .. }
            | Command::DrawLine { matrix, .. }
            | Command::DrawLineRect { matrix, .. } => shift(matrix),
            Command::PushMask
            | Command::ActivateMask
            | Command::DeactivateMask
            | Command::PopMask => {}
            Command::Blend(commands, _) => translate_commands(&mut commands.commands, dx, dy),
        }
    }
}

/// The area of a `width` x `height` target that `commands` can change, with a
/// margin for pixel snapping, line widths and smoothing. None: all of it (the
/// bounds are not known, or the area is most of the target anyway).
/// Some with a zero size: they draw nothing on the target.
fn blend_rect(commands: &CommandList, width: u32, height: u32) -> Option<BlendRect> {
    let mut bounds = CommandBounds::new();
    if !bounds.add(&commands.commands) {
        return None;
    }
    const MARGIN: f32 = 2.0;
    let clamp = |v: f32, max: u32| (v.max(0.0) as u32).min(max);
    let x0 = clamp((bounds.min.0 - MARGIN).floor(), width);
    let y0 = clamp((bounds.min.1 - MARGIN).floor(), height);
    let x1 = clamp((bounds.max.0 + MARGIN).ceil(), width);
    let y1 = clamp((bounds.max.1 + MARGIN).ceil(), height);
    if !(bounds.min.0 <= bounds.max.0 && bounds.min.1 <= bounds.max.1) || x1 <= x0 || y1 <= y0 {
        return Some(BlendRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        });
    }
    // Sizes in steps of 64 (or the whole width or height), so the pooled
    // textures are reused rather than made for every object size.
    let step = |size: u32, max: u32| size.div_ceil(64).saturating_mul(64).min(max);
    let (rect_width, rect_height) = (step(x1 - x0, width), step(y1 - y0, height));
    let rect = BlendRect {
        x: x0.min(width - rect_width),
        y: y0.min(height - rect_height),
        width: rect_width,
        height: rect_height,
    };
    // Close to the whole target: not worth a partial copy.
    if rect.width as u64 * rect.height as u64 * 4 > width as u64 * height as u64 * 3 {
        return None;
    }
    Some(rect)
}

/// Whether all that `commands` draw (and its anti-aliasing margin) lies inside
/// `rect`: false when part of it is off the target, or its size is not known.
fn drawn_within(commands: &CommandList, rect: BlendRect) -> bool {
    let mut bounds = CommandBounds::new();
    if !bounds.add(&commands.commands)
        || bounds.min.0 > bounds.max.0
        || bounds.min.1 > bounds.max.1
    {
        return false;
    }
    const MARGIN: f32 = 2.0;
    bounds.min.0 - MARGIN >= rect.x as f32
        && bounds.min.1 - MARGIN >= rect.y as f32
        && bounds.max.0 + MARGIN <= (rect.x + rect.width) as f32
        && bounds.max.1 + MARGIN <= (rect.y + rect.height) as f32
}

#[derive(Debug)]
pub enum ChunkBlendMode {
    Complex(ComplexBlend),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug)]
pub enum DrawCommand {
    RenderBitmap {
        bitmap: BitmapHandle,
        instance_index: u32,
        vertex_offset: Option<wgpu::BufferAddress>,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    },
    RenderTexture {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
        blend_mode: TrivialBlend,
    },
    /// VibeSkua: a blend group drawn from its place on an atlas page.
    RenderAtlas {
        _page: std::sync::Arc<PoolOrArcTexture>,
        binds: wgpu::BindGroup,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: TrivialBlend,
    },
    RenderAlphaMask {
        maskee: PoolOrArcTexture,
        mask: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
    },
    RenderShape {
        shape: ShapeHandle,
        instance_index: u32,
    },
    DrawRect {
        instance_index: u32,
    },
    DrawLine {
        instance_index: u32,
    },
    DrawLineRect {
        instance_index: u32,
    },
    PushMask,
    ActivateMask,
    DeactivateMask,
    PopMask,
}

impl DrawCommand {
    pub fn name(&self) -> &'static str {
        match self {
            DrawCommand::RenderBitmap { .. } => "render bitmap",
            DrawCommand::RenderShape { .. } => "render shape",
            DrawCommand::RenderTexture { .. } => "render texture",
            DrawCommand::RenderAtlas { .. } => "render atlas",
            DrawCommand::DrawRect { .. } => "draw rect",
            DrawCommand::DrawLine { .. } => "draw line",
            DrawCommand::DrawLineRect { .. } => "draw line rect",
            DrawCommand::PushMask => "push mask",
            DrawCommand::ActivateMask => "activate mask",
            DrawCommand::DeactivateMask => "deactivate mask",
            DrawCommand::PopMask => "pop mask",
            DrawCommand::RenderAlphaMask { .. } => "render alpha mask",
        }
    }
}

#[derive(Copy, Clone)]
pub enum LayerRef<'a> {
    None,
    Current,
    Parent(&'a CommandTarget),
}

/// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
/// Every complex blend will be its own item, but every other draw will be chunked together
#[expect(clippy::too_many_arguments)]
/// VibeSkua: where `chunk_blends` hands each chunk as soon as it is made, with
/// the encoder state to execute it (see `Surface::draw_commands`).
pub type ChunkSink<'a, 'global> = &'a mut dyn FnMut(
    Chunk,
    &mut wgpu::util::StagingBelt,
    &mut Scope<'global, wgpu::CommandEncoder>,
    &mut TexturePool,
);

pub fn chunk_blends<'encoder, 'global: 'encoder>(
    commands: CommandList,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    nearest_layer: LayerRef<'encoder>,
    texture_pool: &'encoder mut TexturePool,
    sink: Option<ChunkSink<'encoder, 'global>>,
    opaque: bool,
) -> Vec<Chunk> {
    let mut handler = WgpuCommandHandler::new(
        descriptors,
        staging_belt,
        dynamic_transforms,
        draw_encoder,
        meshes,
        quality,
        width,
        height,
        nearest_layer,
        texture_pool,
        sink,
    );
    handler.opaque = opaque;
    handler.chunk_blends(commands)
}

struct WgpuCommandHandler<'encoder, 'global: 'encoder> {
    descriptors: &'encoder Descriptors,
    quality: StageQuality,
    width: u32,
    height: u32,
    nearest_layer: LayerRef<'encoder>,
    meshes: &'encoder Vec<Mesh>,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    texture_pool: &'encoder mut TexturePool,
    emulate_lines: bool,
    sink: Option<ChunkSink<'encoder, 'global>>,
    /// VibeSkua: the target was cleared to an opaque colour (the stage), so
    /// it stays opaque: Multiply is then a plain hardware blend.
    opaque: bool,
    /// VibeSkua: blend groups waiting to be drawn together (see `AtlasBatch`).
    atlas: Option<AtlasBatch>,
    /// VibeSkua: chunks made while the atlas page has groups not drawn yet:
    /// they composite from it, so they run, in order, once it is drawn.
    held: Vec<Chunk>,

    result: Vec<Chunk>,
    current: Vec<DrawCommand>,
    transforms: BufferBuilder,
    vertices: BufferBuilder,
    needs_stencil: bool,
    num_masks: i32,
}

impl<'encoder, 'global: 'encoder> WgpuCommandHandler<'encoder, 'global> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        descriptors: &'encoder Descriptors,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'encoder Vec<Mesh>,
        quality: StageQuality,
        width: u32,
        height: u32,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
        sink: Option<ChunkSink<'encoder, 'global>>,
    ) -> Self {
        let transforms = Self::new_transforms(descriptors, dynamic_transforms);
        let vertices = Self::new_vertices(descriptors, dynamic_transforms);

        // DirectX does support drawing lines, but it's very inconsistent.
        // With MSAA, lines have 1.4px thickness, which makes them too thick.
        // Without MSAA, lines have 1px thickness, but their placement is sometimes off.
        let emulate_lines = descriptors.backend == Backend::Dx12;

        Self {
            descriptors,
            quality,
            width,
            height,
            nearest_layer,
            meshes,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            texture_pool,
            emulate_lines,
            sink,
            opaque: false,
            atlas: None,
            held: Vec::new(),

            result: vec![],
            current: vec![],
            transforms,
            vertices,
            needs_stencil: false,
            num_masks: 0,
        }
    }

    /// VibeSkua: every blend group's content is drawn into its own texture
    /// the size of the stage. Kept in a list until the whole command list was
    /// gone through, they were all alive at once: AQW's Battleon draws ~590
    /// Multiply, Overlay and Add groups a frame, 590 stage-sized textures
    /// (1.3 GB at 930x610) held by the pool for good. Executed as soon as it is
    /// made, a chunk gives its texture back for the next group to reuse; the
    /// GPU work is in the same order on the same encoder.
    fn push_chunk(&mut self, chunk: Chunk) {
        if self.atlas.as_ref().is_some_and(|a| !a.commands.is_empty()) {
            self.held.push(chunk);
            return;
        }
        self.run_chunk(chunk);
    }

    fn run_chunk(&mut self, chunk: Chunk) {
        match &mut self.sink {
            Some(sink) => sink(
                chunk,
                self.staging_belt,
                self.draw_encoder,
                self.texture_pool,
            ),
            None => self.result.push(chunk),
        }
    }

    /// Draws the waiting blend groups onto their atlas page, in one pass,
    /// copies the complex blends' groups out of it, then runs the chunks that
    /// waited for it.
    fn flush_atlas(&mut self) {
        let Some(batch) = self.atlas.take() else {
            return;
        };
        if batch.commands.is_empty() {
            return;
        }
        let copies = batch.copies;
        let mut list = CommandList::new();
        list.commands = batch.commands;
        let surface = Surface::new(
            self.descriptors,
            self.quality,
            ATLAS_SIZE,
            ATLAS_SIZE,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let target = surface.draw_commands(
            RenderTargetMode::ExistingWithColor(batch.page.texture().clone(), wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            list,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            self.nearest_layer,
            self.texture_pool,
        );
        target.ensure_cleared(self.draw_encoder);
        for (x, y, width, height, texture) in copies {
            self.draw_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: batch.page.texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        for chunk in mem::take(&mut self.held) {
            self.run_chunk(chunk);
        }
    }

    fn new_transforms(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut transforms = BufferBuilder::new(&descriptors.limits, 0);
        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
        transforms
    }

    fn new_vertices(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut vertices = BufferBuilder::new(&descriptors.limits, 0);
        vertices.set_buffer_limit(dynamic_transforms.vertex_buffer.size());
        vertices
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(&mut self, commands: CommandList) -> Vec<Chunk> {
        commands.execute(self);
        self.flush_atlas();

        let current = mem::take(&mut self.current);
        let mut result = mem::take(&mut self.result);
        let needs_stencil = mem::take(&mut self.needs_stencil);
        let transforms = mem::replace(
            &mut self.transforms,
            Self::new_transforms(self.descriptors, self.dynamic_transforms),
        );
        let vertices = mem::replace(
            &mut self.vertices,
            Self::new_vertices(self.descriptors, self.dynamic_transforms),
        );

        if !current.is_empty() {
            let chunk = Chunk::Draw {
                chunk: current,
                needs_stencil,
                transforms,
                vertices,
            };
            match &mut self.sink {
                Some(sink) => sink(
                    chunk,
                    self.staging_belt,
                    self.draw_encoder,
                    self.texture_pool,
                ),
                None => result.push(chunk),
            }
        }

        result
    }

    fn add_to_current(
        &mut self,
        matrix: Matrix,
        tz: f64,
        color_transform: ColorTransform,
        command_builder: impl FnOnce(u32) -> DrawCommand,
    ) {
        self.add_to_current_with_vertices(matrix, tz, color_transform, None, |instance_index, _| {
            command_builder(instance_index)
        })
    }

    fn add_to_current_with_vertices(
        &mut self,
        matrix: Matrix,
        tz: f64,
        color_transform: ColorTransform,
        vertices: Option<&[PosUvVertex]>,
        command_builder: impl FnOnce(u32, Option<wgpu::BufferAddress>) -> DrawCommand,
    ) {
        let transform = Transforms {
            world_matrix: [
                [matrix.a, matrix.b, 0.0, 0.0],
                [matrix.c, matrix.d, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    matrix.tx.to_pixels() as f32,
                    matrix.ty.to_pixels() as f32,
                    tz as f32,
                    1.0,
                ],
            ],
            mult_color: color_transform.mult_rgba_normalized(),
            add_color: color_transform.add_rgba_normalized(),
        };
        if let (Ok(transform_range), Ok(vertices_range)) = (
            self.transforms.add(&[transform]),
            vertices.map(|v| self.vertices.add(v)).transpose(),
        ) {
            self.current.push(command_builder(
                (transform_range.start as usize / size_of::<Transforms>()) as u32,
                vertices_range.map(|v| v.start),
            ));
        } else {
            let chunk = Chunk::Draw {
                chunk: mem::take(&mut self.current),
                needs_stencil: self.needs_stencil,
                transforms: mem::replace(
                    &mut self.transforms,
                    Self::new_transforms(self.descriptors, self.dynamic_transforms),
                ),
                vertices: mem::replace(
                    &mut self.vertices,
                    Self::new_vertices(self.descriptors, self.dynamic_transforms),
                ),
            };
            self.push_chunk(chunk);
            let transform_range = self
                .transforms
                .add(&[transform])
                .expect("Buffer must be able to fit a new thing, it was just emptied");
            let vertices_range = vertices.map(|v| {
                self.vertices
                    .add(v)
                    .expect("Buffer must be able to fit a new thing, it was just emptied")
            });
            self.current.push(command_builder(
                (transform_range.start as usize / size_of::<Transforms>()) as u32,
                vertices_range.map(|v| v.start),
            ));
        }
    }
}

impl CommandHandler for WgpuCommandHandler<'_, '_> {
    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode) {
        // VibeSkua: with RUFFLE_FILTERS=off, blended content is drawn in place
        // (as normal), not through a fresh full-size texture per blend group.
        if !crate::backend::offscreen_effects() {
            let _ = blend_mode;
            commands.execute(self);
            return;
        }
        // Alpha and Erase change their layer outside the object too.
        let rect = match BlendType::from(blend_mode.clone()) {
            BlendType::Complex(ComplexBlend::Alpha | ComplexBlend::Erase) => None,
            BlendType::Complex(_) | BlendType::Trivial(_) => {
                blend_rect(&commands, self.width, self.height)
            }
            BlendType::Shader(_) => None,
        };
        // VibeSkua: only a group drawn wholly inside its rect goes on an atlas
        // page. The rect is clipped to the target (and its own texture clips
        // the rest away), but on a page what is drawn past it lands on the
        // other groups' places: Yulgar's room-sized Overlay lights, partly
        // off screen, showed as light and dark boxes in other blends.
        let on_page = rect.is_some_and(|rect| drawn_within(&commands, rect));
        let mut commands = commands;
        if let Some(rect) = rect {
            if rect.width == 0 {
                // Off the target or empty: blending nothing changes nothing.
                return;
            }
            translate_commands(
                &mut commands.commands,
                Twips::from_pixels_i32(rect.x as i32),
                Twips::from_pixels_i32(rect.y as i32),
            );
        }
        let surface = Surface::new(
            self.descriptors,
            self.quality,
            rect.map_or(self.width, |rect| rect.width),
            rect.map_or(self.height, |rect| rect.height),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let target_layer = if let RenderBlendMode::Builtin(BlendMode::Layer) = &blend_mode {
            LayerRef::Current
        } else {
            self.nearest_layer
        };
        let blend_type = match BlendType::from(blend_mode) {
            BlendType::Complex(ComplexBlend::Multiply) if self.opaque => {
                BlendType::Trivial(TrivialBlend::Multiply)
            }
            blend_type => blend_type,
        };
        if let (Some(rect), BlendType::Trivial(trivial)) = (rect, &blend_type)
            && !matches!(trivial, TrivialBlend::Normal)
            && on_page
            && rect.width <= ATLAS_SIZE
            && rect.height <= ATLAS_SIZE
            && !has_complex_blend(&commands.commands)
        {
            let trivial = *trivial;
            if self.atlas.as_ref().is_some_and(|a| a.place(rect.width, rect.height).is_none()) {
                self.flush_atlas();
            }
            let atlas = self
                .atlas
                .get_or_insert_with(|| AtlasBatch::new(self.descriptors, self.texture_pool));
            let (ax, ay) = atlas
                .place(rect.width, rect.height)
                .expect("a group fits an empty page");
            atlas.take(rect.width, rect.height);
            // Already moved to the group's own origin; now onto its place.
            translate_commands(
                &mut commands.commands,
                Twips::from_pixels_i32(-(ax as i32)),
                Twips::from_pixels_i32(-(ay as i32)),
            );
            atlas.commands.append(&mut commands.commands);
            let size = ATLAS_SIZE as f32;
            let (u0, v0) = (ax as f32 / size, ay as f32 / size);
            let (u1, v1) = (
                (ax + rect.width) as f32 / size,
                (ay + rect.height) as f32 / size,
            );
            let vertices = [
                PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
            ];
            let mut matrix = Matrix::scale(rect.width as f32, rect.height as f32);
            matrix.tx = Twips::from_pixels_i32(rect.x as i32);
            matrix.ty = Twips::from_pixels_i32(rect.y as i32);
            let page = atlas.page.clone();
            let binds = atlas.binds.clone();
            self.add_to_current_with_vertices(
                matrix,
                0.0,
                Default::default(),
                Some(&vertices),
                |instance_index, vertex_offset| DrawCommand::RenderAtlas {
                    _page: page,
                    binds,
                    instance_index,
                    vertex_offset: vertex_offset.expect("vertices were given"),
                    blend_mode: trivial,
                },
            );
            return;
        }
        if let (Some(rect), BlendType::Complex(complex)) = (rect, &blend_type)
            && on_page
            && rect.width <= ATLAS_SIZE
            && rect.height <= ATLAS_SIZE
            && !has_complex_blend(&commands.commands)
        {
            let complex = *complex;
            let bytes = u64::from(rect.width) * u64::from(rect.height) * 4;
            if self.atlas.as_ref().is_some_and(|a| {
                a.place(rect.width, rect.height).is_none()
                    || a.copied_bytes + bytes > ATLAS_HELD_BYTES
            }) {
                self.flush_atlas();
            }
            let atlas = self
                .atlas
                .get_or_insert_with(|| AtlasBatch::new(self.descriptors, self.texture_pool));
            let (ax, ay) = atlas
                .place(rect.width, rect.height)
                .expect("a group fits an empty page");
            atlas.take(rect.width, rect.height);
            translate_commands(
                &mut commands.commands,
                Twips::from_pixels_i32(-(ax as i32)),
                Twips::from_pixels_i32(-(ay as i32)),
            );
            atlas.commands.append(&mut commands.commands);
            // The group's own texture, as the blend reads it (filled from the
            // page by flush_atlas).
            let texture = self.texture_pool.get_texture(
                self.descriptors,
                wgpu::Extent3d {
                    width: rect.width,
                    height: rect.height,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                wgpu::TextureFormat::Rgba8Unorm,
                1,
            );
            let atlas = self.atlas.as_mut().expect("made above");
            atlas
                .copies
                .push((ax, ay, rect.width, rect.height, texture.0.clone()));
            atlas.copied_bytes += bytes;
            if !self.current.is_empty() {
                let chunk = Chunk::Draw {
                    chunk: mem::take(&mut self.current),
                    needs_stencil: self.needs_stencil,
                    transforms: mem::replace(
                        &mut self.transforms,
                        Self::new_transforms(self.descriptors, self.dynamic_transforms),
                    ),
                    vertices: mem::replace(
                        &mut self.vertices,
                        Self::new_vertices(self.descriptors, self.dynamic_transforms),
                    ),
                };
                self.push_chunk(chunk);
            }
            let chunk = Chunk::Blend {
                texture: PoolOrArcTexture::Pool(texture),
                blend_mode: ChunkBlendMode::Complex(complex),
                needs_stencil: self.num_masks > 0,
                rect: Some(rect),
            };
            self.push_chunk(chunk);
            return;
        }
        let clear_color = blend_type.default_color();
        let target = surface.draw_commands(
            RenderTargetMode::FreshWithColor(clear_color),
            self.descriptors,
            self.meshes,
            commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            target_layer,
            self.texture_pool,
        );
        target.ensure_cleared(self.draw_encoder);

        // We currently do not support shader blends in masks. In order not to
        // break other parts of the scene, we just fall back to a normal blend.
        //
        // TODO Add support for shader blends in masks.
        let is_shader_blend_in_mask =
            self.num_masks > 0 && matches!(blend_type, BlendType::Shader(_));
        let blend_type = if is_shader_blend_in_mask {
            BlendType::Trivial(TrivialBlend::Normal)
        } else {
            blend_type
        };

        match blend_type {
            BlendType::Trivial(blend_mode) => {
                let mut matrix = Matrix::scale(target.width() as f32, target.height() as f32);
                if let Some(rect) = rect {
                    matrix.tx = Twips::from_pixels_i32(rect.x as i32);
                    matrix.ty = Twips::from_pixels_i32(rect.y as i32);
                }
                let transform = Transform {
                    matrix,
                    tz: 0.0,
                    color_transform: Default::default(),
                    perspective_projection: None,
                };
                let texture = target.take_color_texture();
                let bind_group =
                    self.descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.descriptors.bind_layouts.bitmap,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(texture.view()),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(
                                        self.descriptors.bitmap_samplers.get_sampler(false, false),
                                    ),
                                },
                            ],
                            label: None,
                        });
                self.add_to_current(
                    transform.matrix,
                    transform.tz,
                    transform.color_transform,
                    |instance_index| DrawCommand::RenderTexture {
                        _texture: texture,
                        binds: bind_group,
                        instance_index,
                        blend_mode,
                    },
                );
            }
            blend_type => {
                if !self.current.is_empty() {
                    let chunk = Chunk::Draw {
                        chunk: mem::take(&mut self.current),
                        needs_stencil: self.needs_stencil,
                        transforms: mem::replace(
                            &mut self.transforms,
                            Self::new_transforms(self.descriptors, self.dynamic_transforms),
                        ),
                        vertices: mem::replace(
                            &mut self.vertices,
                            Self::new_vertices(self.descriptors, self.dynamic_transforms),
                        ),
                    };
                    self.push_chunk(chunk);
                }
                let chunk_blend_mode = match blend_type {
                    BlendType::Complex(complex) => ChunkBlendMode::Complex(complex),
                    BlendType::Shader(shader) => ChunkBlendMode::Shader(shader),
                    _ => unreachable!(),
                };
                let chunk = Chunk::Blend {
                    texture: target.take_color_texture(),
                    blend_mode: chunk_blend_mode,
                    needs_stencil: self.num_masks > 0,
                    rect,
                };
                self.push_chunk(chunk);
                self.needs_stencil = self.num_masks > 0;
            }
        }
    }

    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    ) {
        let texture = as_texture(&bitmap);

        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(region.width() as f32, region.height() as f32);

        let vertices: &[PosUvVertex] = {
            let (u0, u1, v0, v1) = (
                region.x_min as f32 / texture.texture.width() as f32,
                region.x_max as f32 / texture.texture.width() as f32,
                region.y_min as f32 / texture.texture.height() as f32,
                region.y_max as f32 / texture.texture.height() as f32,
            );
            &[
                PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
            ]
        };

        self.add_to_current_with_vertices(
            matrix,
            transform.tz,
            transform.color_transform,
            Some(vertices),
            |instance_index, vertex_offset| DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset,
                smoothing,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: false,
            },
        );
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        let mut matrix = transform.matrix;
        {
            let texture = as_texture(&bitmap);
            matrix *= Matrix::scale(
                texture.texture.width() as f32,
                texture.texture.height() as f32,
            );
        }
        self.add_to_current(
            matrix,
            transform.tz,
            transform.color_transform,
            |instance_index| DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset: None,
                smoothing: false,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: true,
            },
        );
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        self.add_to_current(
            transform.matrix,
            transform.tz,
            transform.color_transform,
            |instance_index| DrawCommand::RenderShape {
                shape,
                instance_index,
            },
        );
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.add_to_current(
            matrix,
            0.0,
            ColorTransform::multiply_from(color),
            |instance_index| DrawCommand::DrawRect { instance_index },
        );
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                0.0,
                ColorTransform::multiply_from(color),
                |instance_index| DrawCommand::DrawLine { instance_index },
            );
        }
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line_rect(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                0.0,
                ColorTransform::multiply_from(color),
                |instance_index| DrawCommand::DrawLineRect { instance_index },
            );
        }
    }

    fn push_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks += 1;
        self.current.push(DrawCommand::PushMask);
    }

    fn activate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::ActivateMask);
    }

    fn deactivate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::DeactivateMask);
    }

    fn pop_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks -= 1;
        self.current.push(DrawCommand::PopMask);
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        let surface = Surface::new(
            self.descriptors,
            self.quality,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let maskee = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            maskee_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        maskee.ensure_cleared(self.draw_encoder);
        let matrix = Matrix::scale(maskee.width() as f32, maskee.height() as f32);
        let maskee = maskee.take_color_texture();

        let mask = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            mask_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        mask.ensure_cleared(self.draw_encoder);
        let mask = mask.take_color_texture();

        let binds = self
            .descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.descriptors.bind_layouts.alpha_mask,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(maskee.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mask.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            self.descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
                label: None,
            });

        self.add_to_current(matrix, 0.0, Default::default(), |instance_index| {
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                instance_index,
            }
        });
    }
}
