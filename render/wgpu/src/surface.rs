mod commands;
pub mod target;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::commands::{BlendRect, Chunk, CommandRenderer, chunk_blends};
use crate::surface::target::{BlendBuffer, PoolOrArcTexture};
use crate::utils::run_copy_pipeline;
use crate::utils::supported_sample_count;
use crate::{Descriptors, MaskState, Pipelines};
use ruffle_render::commands::CommandList;
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use target::CommandTarget;
use tracing::instrument;
use wgpu_profiler::Scope;

pub use crate::surface::commands::LayerRef;

use self::commands::ChunkBlendMode;

/// VibeSkua: a complex blend (Multiply, Overlay, ...) waiting to be drawn.
///
/// Each one had a render pass of its own on the parent target, after the copy
/// of the parent it reads, and the draws after it then started another pass:
/// two full-size passes per blend, ~0.3 ms each on an integrated GPU, and AQW
/// has ~180 of these blends a frame in Battleon. It is now the first draw of
/// the next pass (a pass of its own only when another copy of the parent
/// comes first, its stencil use differs, or nothing follows).
///
/// Blends waiting together form a group: a blend whose rect overlaps none of
/// the waiting ones reads parent pixels they do not change, so its parent copy
/// can be taken before they are drawn and all of them share one pass. A full
/// Battleon drew ~120 blends a frame in passes of their own before this.
struct PendingBlend<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    rect: Option<BlendRect>,
    needs_stencil: bool,
    stencil_reference: Option<u32>,
    // Kept until drawn, so the pools do not hand them out again first.
    _texture: PoolOrArcTexture,
    _parent_copy: Option<BlendBuffer>,
}

impl PendingBlend<'_> {
    fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        descriptors: &Descriptors,
        target: &CommandTarget,
    ) {
        render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
        if let Some(reference) = self.stencil_reference {
            render_pass.set_stencil_reference(reference);
        }
        render_pass.set_pipeline(self.pipeline);
        render_pass.set_bind_group(2, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, descriptors.quad.vertices_pos.slice(..));
        render_pass.set_index_buffer(descriptors.quad.indices.slice(..), wgpu::IndexFormat::Uint32);
        if let Some(rect) = self.rect {
            // The quad covers the viewport: the rect, where both textures
            // (the object's and the parent copy) start.
            render_pass.set_viewport(
                rect.x as f32,
                rect.y as f32,
                rect.width as f32,
                rect.height as f32,
                0.0,
                1.0,
            );
        }
        render_pass.draw_indexed(0..6, 0, 0..1);
        if self.rect.is_some() {
            render_pass.set_viewport(
                0.0,
                0.0,
                target.width() as f32,
                target.height() as f32,
                0.0,
                1.0,
            );
        }
    }

    /// Whether this blend can join `group`: same stencil use, and its rect
    /// overlaps none of theirs (a blend draws only inside its rect).
    fn fits(rect: Option<BlendRect>, needs_stencil: bool, group: &[Self]) -> bool {
        let Some(rect) = rect else {
            return false;
        };
        group.iter().all(|other| {
            other.needs_stencil == needs_stencil
                && other.rect.is_some_and(|o| {
                    rect.x >= o.x + o.width
                        || o.x >= rect.x + rect.width
                        || rect.y >= o.y + o.height
                        || o.y >= rect.y + rect.height
                })
        })
    }

    /// Draws the waiting group in one pass of its own.
    fn draw_alone<'global>(
        group: &mut Vec<Self>,
        descriptors: &Descriptors,
        target: &CommandTarget,
        draw_encoder: &mut Scope<'global, wgpu::CommandEncoder>,
        texture_pool: &mut TexturePool,
    ) {
        let Some(first) = group.first() else {
            return;
        };
        let mut render_pass = draw_encoder.scoped_render_pass(
            "Complex blend",
            wgpu::RenderPassDescriptor {
                color_attachments: &[target.color_attachments()],
                depth_stencil_attachment: if first.needs_stencil {
                    target.stencil_attachment(descriptors, texture_pool)
                } else {
                    None
                },
                ..Default::default()
            },
        );
        for blend in group.drain(..) {
            blend.draw(&mut render_pass, descriptors, target);
        }
    }
}

#[derive(Debug)]
pub struct Surface {
    size: wgpu::Extent3d,
    quality: StageQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
}

impl Surface {
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        width: u32,
        height: u32,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // VibeSkua: with edge smoothing (RUFFLE_EDGE_AA) shapes get edge
        // fringes instead of multisampling, above Low quality (Flash's Low
        // smoothed nothing).
        let edge_aa = crate::backend::edge_aa().is_some();
        let sample_count = if edge_aa {
            1
        } else {
            supported_sample_count(
                &descriptors.adapter,
                quality.sample_count(),
                frame_buffer_format,
            )
        };
        let pipelines = descriptors.pipelines(
            sample_count,
            frame_buffer_format,
            edge_aa && quality != StageQuality::Low,
        );
        Self {
            size,
            quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
        }
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands_and_copy_to<'encoder, 'global: 'encoder>(
        &self,
        frame_view: &wgpu::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        layer: LayerRef<'encoder>,
        texture_pool: &'global mut TexturePool,
    ) {
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            layer,
            texture_pool,
        );

        run_copy_pipeline(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            target.globals(),
            1,
            draw_encoder,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'encoder, 'global: 'encoder>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'encoder Descriptors,
        meshes: &'encoder Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
    ) -> CommandTarget {
        let opaque = matches!(&render_target_mode, RenderTargetMode::FreshWithColor(color) if color.a >= 1.0);
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            self.size,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        );

        let mut num_masks = 0;
        let mut mask_state = MaskState::NoMask;
        // VibeSkua: a complex blend is drawn at the start of the next draw
        // pass instead of in a pass of its own (see PendingBlend).
        let mut pending: Vec<PendingBlend<'_>> = Vec::new();
        // VibeSkua: each chunk runs as soon as chunk_blends makes it (see
        // WgpuCommandHandler::push_chunk).
        let mut execute = |chunk: Chunk,
                           staging_belt: &mut wgpu::util::StagingBelt,
                           draw_encoder: &mut Scope<'global, wgpu::CommandEncoder>,
                           texture_pool: &mut TexturePool| {
            match chunk {
                Chunk::Draw {
                    chunk,
                    needs_stencil,
                    transforms,
                    vertices,
                } => {
                    // Between passes: the frame's encoder may be cut here.
                    crate::split_encoder::cut(&descriptors.device, &mut **draw_encoder);
                    // Its pipeline must match this pass's stencil attachment.
                    if pending.first().is_some_and(|b| b.needs_stencil != needs_stencil) {
                        PendingBlend::draw_alone(
                            &mut pending,
                            descriptors,
                            &target,
                            draw_encoder,
                            texture_pool,
                        );
                    }
                    let blends_first = std::mem::take(&mut pending);
                    transforms.copy_to(staging_belt, draw_encoder, &dynamic_transforms.buffer);
                    vertices.copy_to(
                        staging_belt,
                        draw_encoder,
                        &dynamic_transforms.vertex_buffer,
                    );
                    let mut render_pass = draw_encoder.scoped_render_pass(
                        format!(
                            "Chunked draw calls {}",
                            if needs_stencil {
                                "(with stencil)"
                            } else {
                                "(Stencilless)"
                            }
                        ),
                        wgpu::RenderPassDescriptor {
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        },
                    );
                    for blend in blends_first {
                        blend.draw(&mut render_pass, descriptors, &target);
                    }
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
                    render_pass.set_bind_group(1, &dynamic_transforms.bind_group, &[]);
                    let mut renderer = CommandRenderer::new(
                        &self.pipelines,
                        descriptors,
                        &dynamic_transforms.vertex_buffer,
                        num_masks,
                        mask_state,
                        needs_stencil,
                    );

                    for command in &chunk {
                        renderer.execute(&mut render_pass.scope(command.name()), command);
                    }

                    num_masks = renderer.num_masks();
                    mask_state = renderer.mask_state();
                }
                Chunk::Blend {
                    texture,
                    blend_mode: ChunkBlendMode::Shader(shader),
                    needs_stencil,
                    ..
                } => {
                    assert!(!needs_stencil, "Shader blend mode not implemented in masks");
                    PendingBlend::draw_alone(
                        &mut pending,
                        descriptors,
                        &target,
                        draw_encoder,
                        texture_pool,
                    );
                    let parent_blend_buffer =
                        target.update_blend_buffer(descriptors, texture_pool, draw_encoder);
                    run_pixelbender_shader_impl(
                        descriptors,
                        shader,
                        ShaderMode::Filter,
                        &[
                            PixelBenderShaderArgument::ImageInput {
                                index: 0,
                                channels: 0xFF,
                                name: "background".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(
                                    parent_blend_buffer.texture(),
                                )),
                            },
                            PixelBenderShaderArgument::ImageInput {
                                index: 1,
                                channels: 0xff,
                                name: "foreground".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                            },
                        ],
                        parent_blend_buffer.texture(),
                        draw_encoder,
                        target.color_attachments(),
                        target.sample_count(),
                        &FilterSource::for_entire_texture(texture.texture()),
                    )
                    .expect("Failed to run PixelBender blend mode");
                }
                Chunk::Blend {
                    texture,
                    blend_mode: ChunkBlendMode::Complex(blend_mode),
                    needs_stencil,
                    rect,
                } => {
                    let parent = match blend_mode {
                        ComplexBlend::Alpha | ComplexBlend::Erase => {
                            match nearest_layer {
                                LayerRef::None => {
                                    // An Alpha or Erase with no Layer above it should be ignored
                                    return;
                                }
                                LayerRef::Current => &target,
                                LayerRef::Parent(layer) => layer,
                            }
                        }
                        _ => &target,
                    };

                    // The copy below must include an earlier blend's result,
                    // unless that blend draws outside the copied rect.
                    if !(std::ptr::eq(parent, &target)
                        && PendingBlend::fits(rect, needs_stencil, &pending))
                    {
                        PendingBlend::draw_alone(
                            &mut pending,
                            descriptors,
                            &target,
                            draw_encoder,
                            texture_pool,
                        );
                    }
                    let mut copy_scope = draw_encoder.scope("Copy parent for blend");
                    let parent_copy = rect.map(|rect| {
                        parent.blend_buffer_for_rect(descriptors, texture_pool, &mut copy_scope, rect)
                    });
                    let parent_blend_buffer = match &parent_copy {
                        Some(buffer) => buffer,
                        None => parent.update_blend_buffer(descriptors, texture_pool, &mut copy_scope),
                    };
                    drop(copy_scope);

                    let bind_group =
                        descriptors
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: create_debug_label!(
                                    "Complex blend binds {:?} {}",
                                    blend_mode,
                                    if needs_stencil {
                                        "(with stencil)"
                                    } else {
                                        "(Stencilless)"
                                    }
                                )
                                .as_deref(),
                                layout: &descriptors.bind_layouts.blend,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            parent_blend_buffer.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(
                                            texture.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(
                                            descriptors.bitmap_samplers.get_sampler(false, false),
                                        ),
                                    },
                                ],
                            });

                    let (pipeline, stencil_reference) = if needs_stencil {
                        let reference = match mask_state {
                            MaskState::NoMask => None,
                            MaskState::DrawMaskStencil => Some(num_masks - 1),
                            MaskState::DrawMaskedContent | MaskState::ClearMaskStencil => {
                                Some(num_masks)
                            }
                        };
                        (
                            self.pipelines.complex_blends[blend_mode].pipeline_for(mask_state),
                            reference,
                        )
                    } else {
                        (
                            self.pipelines.complex_blends[blend_mode].stencilless_pipeline(),
                            None,
                        )
                    };
                    pending.push(PendingBlend {
                        pipeline,
                        bind_group,
                        rect,
                        needs_stencil,
                        stencil_reference,
                        _texture: texture,
                        _parent_copy: parent_copy,
                    });
                }
            }
        };
        let chunks = chunk_blends(
            commands,
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            self.quality,
            target.width(),
            target.height(),
            match nearest_layer {
                LayerRef::Current => LayerRef::Parent(&target),
                layer => layer,
            },
            texture_pool,
            Some(&mut execute),
            opaque,
        );
        for chunk in chunks {
            execute(chunk, staging_belt, draw_encoder, texture_pool);
        }
        PendingBlend::draw_alone(
            &mut pending,
            descriptors,
            &target,
            draw_encoder,
            texture_pool,
        );

        // If nothing happened, ensure it's cleared so we don't operate on garbage data
        target.ensure_cleared(draw_encoder);

        target
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}
