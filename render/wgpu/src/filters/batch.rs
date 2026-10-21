//! VibeSkua: the blurs and glows of many cached bitmaps drawn together.
//!
//! Each filtered cached bitmap (AQW's avatars and items glow) took a render
//! pass per blur step and one for the glow: 60-140 passes a frame in a busy
//! room. On built-in graphics a frame's cost is mostly wgpu's CPU work per
//! render pass (~44 us each, measured), not the GPU's. Here the bitmaps are
//! copied side by side onto shared pages and every blur step, and the glows,
//! are drawn for all of them in one pass each, then copied back.
//!
//! The picture is the same as drawing each alone: every blur sample is clamped
//! to the bitmap's own place on the page, as the sampler clamped it to the
//! edges of its own texture; glows have no offset here, so they only read
//! their own place; and each place is written in full, without blending.
//! Only Blur and Glow filters are batched; a bitmap with any other filter, or
//! drawn with multisampling, goes the old way.

use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::{Texture, as_texture};
use bytemuck::{Pod, Zeroable};
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::filters::Filter;
use std::sync::OnceLock;
use swf::BlurFilter as BlurFilterArgs;
use wgpu::util::DeviceExt;
use wgpu::vertex_attr_array;

const PAGE_WIDTH: u32 = 2048;
const PAGE_MAX_HEIGHT: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BlurVertex {
    position: [f32; 2],
    uv: [f32; 2],
    clamp_rect: [f32; 4],
    dir_size: [f32; 4],
    weights: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GlowVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    args: [f32; 4],
}

const BLUR_ATTRIBUTES: [wgpu::VertexAttribute; 5] = vertex_attr_array![
    0 => Float32x2,
    1 => Float32x2,
    2 => Float32x4,
    3 => Float32x4,
    4 => Float32x4,
];

const GLOW_ATTRIBUTES: [wgpu::VertexAttribute; 4] = vertex_attr_array![
    0 => Float32x2,
    1 => Float32x2,
    2 => Float32x4,
    3 => Float32x4,
];

pub struct BatchPipelines {
    blur_layout: wgpu::BindGroupLayout,
    glow_layout: wgpu::BindGroupLayout,
    blur_pipeline_layout: wgpu::PipelineLayout,
    glow_pipeline_layout: wgpu::PipelineLayout,
    blur: OnceLock<wgpu::RenderPipeline>,
    glow: OnceLock<wgpu::RenderPipeline>,
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl BatchPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[texture_entry(0), sampler_entry(1)],
            label: create_debug_label!("Batched blur binds").as_deref(),
        });
        let glow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[texture_entry(0), sampler_entry(1), texture_entry(2)],
            label: create_debug_label!("Batched glow binds").as_deref(),
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&blur_layout)],
            immediate_size: 0,
        });
        let glow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&glow_layout)],
            immediate_size: 0,
        });
        Self {
            blur_layout,
            glow_layout,
            blur_pipeline_layout,
            glow_pipeline_layout,
            blur: OnceLock::new(),
            glow: OnceLock::new(),
        }
    }

    fn pipeline(
        descriptors: &Descriptors,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        stride: u64,
        attributes: &[wgpu::VertexAttribute],
    ) -> wgpu::RenderPipeline {
        descriptors
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: create_debug_label!("Batched filter").as_deref(),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("main_vertex"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes,
                    })],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::default(),
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("main_fragment"),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn blur(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.blur.get_or_init(|| {
            Self::pipeline(
                descriptors,
                &self.blur_pipeline_layout,
                &descriptors.shaders.blur_batch_filter,
                std::mem::size_of::<BlurVertex>() as u64,
                &BLUR_ATTRIBUTES,
            )
        })
    }

    fn glow(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.glow.get_or_init(|| {
            Self::pipeline(
                descriptors,
                &self.glow_pipeline_layout,
                &descriptors.shaders.glow_batch_filter,
                std::mem::size_of::<GlowVertex>() as u64,
                &GLOW_ATTRIBUTES,
            )
        })
    }
}

/// The steps of a blur, as `BlurFilter::apply` runs them: for each, the
/// direction (in pixels: 1 = one pixel) and blur.wgsl's arguments full_size,
/// m, m2, first_weight, last_offset and last_weight.
fn blur_steps(filter: &BlurFilterArgs) -> Vec<(bool, [f32; 6])> {
    let mut steps = Vec::new();
    for _ in 0..(filter.num_passes() as usize) {
        for horizontal in [true, false] {
            let strength = if horizontal {
                filter.blur_x.to_f32()
            } else {
                filter.blur_y.to_f32()
            };
            let full_size = strength.min(255.0);
            if full_size <= 1.0 {
                continue;
            }
            let radius = (full_size - 1.0) / 2.0;
            let m = radius.ceil() - 1.0;
            let alpha = ((radius - m) * 255.0).floor() / 255.0;
            let last_offset = 1.0 / ((1.0 / alpha) + 1.0);
            let last_weight = alpha + 1.0;
            steps.push((
                horizontal,
                [full_size, m, m * 2.0, alpha, last_offset, last_weight],
            ));
        }
    }
    steps
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Page {
    Input,
    A,
    B,
    Glow,
}

impl Page {
    fn index(self) -> usize {
        match self {
            Page::Input => 0,
            Page::A => 1,
            Page::B => 2,
            Page::Glow => 3,
        }
    }
}

struct Item {
    texture: wgpu::Texture,
    filters: Vec<Filter>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

enum Step {
    CopyIn(usize),
    Blur {
        source: Page,
        target: Page,
        vertices: std::ops::Range<u64>,
    },
    Glow {
        blurred: Page,
        vertices: std::ops::Range<u64>,
    },
    /// An item's result, from a page, into the input page (more filters to
    /// come) or its own texture (done).
    CopyOut {
        item: usize,
        from: Page,
        done: bool,
    },
}

#[derive(Default)]
pub struct FilterBatch {
    items: Vec<Item>,
    // Rows: the next place in the current row, and the row's height.
    x: u32,
    y: u32,
    row_height: u32,
    /// The right edge of the widest row (the pages are no wider).
    max_x: u32,
}

impl FilterBatch {
    /// Whether a cached bitmap with these filters can be batched.
    pub fn accepts(filters: &[Filter], texture: &wgpu::Texture) -> bool {
        texture.sample_count() == 1
            && texture.format() == wgpu::TextureFormat::Rgba8Unorm
            && texture.width() <= PAGE_WIDTH
            && texture.height() <= PAGE_MAX_HEIGHT
            && !filters.is_empty()
            && filters.iter().all(|filter| match filter {
                // A blur with no step is drawn through an identity color
                // matrix the old way, which can change pixels slightly.
                Filter::BlurFilter(blur) => !blur_steps(blur).is_empty(),
                Filter::GlowFilter(_) => true,
                _ => false,
            })
    }

    /// Adds a cached bitmap (already drawn) and its filters; gives the filters
    /// back if the page has no room left (run the batch, then add it again).
    pub fn push(&mut self, texture: &wgpu::Texture, filters: Vec<Filter>) -> Result<(), Vec<Filter>> {
        let (width, height) = (texture.width(), texture.height());
        let (x, y) = if self.x + width <= PAGE_WIDTH {
            (self.x, self.y)
        } else {
            (0, self.y + self.row_height)
        };
        if y + height > PAGE_MAX_HEIGHT {
            return Err(filters);
        }
        if x == 0 && self.x != 0 {
            self.row_height = 0;
        }
        self.x = x + width;
        self.max_x = self.max_x.max(self.x);
        self.y = y;
        self.row_height = self.row_height.max(height);
        self.items.push(Item {
            texture: texture.clone(),
            filters,
            x,
            y,
            width,
            height,
        });
        Ok(())
    }

    /// Whether these commands draw a bitmap still waiting in the batch (a
    /// cached child of the object they draw): the batch must run first.
    pub fn is_used_by(&self, commands: &CommandList) -> bool {
        !self.items.is_empty() && self.used_in(&commands.commands)
    }

    fn used_in(&self, commands: &[Command]) -> bool {
        commands.iter().any(|command| match command {
            Command::RenderBitmap { bitmap, .. } => {
                let texture: &Texture = as_texture(bitmap);
                self.items.iter().any(|item| item.texture == texture.texture)
            }
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => self.used_in(&maskee_commands.commands) || self.used_in(&mask_commands.commands),
            Command::Blend(commands, _) => self.used_in(&commands.commands),
            _ => false,
        })
    }

    /// Draws every waiting bitmap's filters and empties the batch.
    pub fn run(
        &mut self,
        descriptors: &Descriptors,
        encoder: &mut wgpu::CommandEncoder,
        texture_pool: &mut TexturePool,
    ) {
        if self.items.is_empty() {
            return;
        }
        let items = std::mem::take(&mut self.items);
        // Pages only as large as the places used (in 256-pixel steps, so that
        // the texture pool can reuse them from frame to frame).
        let page_width = self.max_x.next_multiple_of(256).min(PAGE_WIDTH);
        let page_height = (self.y + self.row_height).next_multiple_of(256).min(PAGE_MAX_HEIGHT);
        (self.x, self.y, self.row_height, self.max_x) = (0, 0, 0, 0);
        let (page_w, page_h) = (page_width as f32, page_height as f32);

        // Plan every step first, so that all vertices go into one buffer.
        let mut steps = Vec::new();
        let mut blur_vertices: Vec<BlurVertex> = Vec::new();
        let mut glow_vertices: Vec<GlowVertex> = Vec::new();
        steps.extend((0..items.len()).map(Step::CopyIn));

        let quad = |item: &Item| {
            let (x0, y0) = (item.x as f32 / page_w, item.y as f32 / page_h);
            let x1 = (item.x + item.width) as f32 / page_w;
            let y1 = (item.y + item.height) as f32 / page_h;
            [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]
        };

        let rounds = items.iter().map(|item| item.filters.len()).max().unwrap_or(0);
        for round in 0..rounds {
            let blurs: Vec<Vec<(bool, [f32; 6])>> = items
                .iter()
                .map(|item| match item.filters.get(round) {
                    Some(Filter::BlurFilter(blur)) => blur_steps(blur),
                    Some(Filter::GlowFilter(glow)) => blur_steps(&glow.inner_blur_filter()),
                    _ => Vec::new(),
                })
                .collect();
            // Where each item's latest blur result is.
            let mut at = vec![Page::Input; items.len()];
            let max_steps = blurs.iter().map(Vec::len).max().unwrap_or(0);
            for step in 0..max_steps {
                // Where each item is at the start of this step (an item moved by
                // the first pass must not be moved again by the third).
                let before = at.clone();
                for (source, target) in [
                    (Page::Input, Page::A),
                    (Page::B, Page::A),
                    (Page::A, Page::B),
                ] {
                    let start = blur_vertices.len();
                    for (i, item) in items.iter().enumerate() {
                        let Some((horizontal, args)) = blurs[i].get(step) else {
                            continue;
                        };
                        if before[i] != source {
                            continue;
                        }
                        at[i] = target;
                        let clamp_rect = [
                            (item.x as f32 + 0.5) / page_w,
                            (item.y as f32 + 0.5) / page_h,
                            ((item.x + item.width) as f32 - 0.5) / page_w,
                            ((item.y + item.height) as f32 - 0.5) / page_h,
                        ];
                        let direction = if *horizontal {
                            [1.0 / page_w, 0.0]
                        } else {
                            [0.0, 1.0 / page_h]
                        };
                        for position in quad(item) {
                            blur_vertices.push(BlurVertex {
                                position,
                                uv: position,
                                clamp_rect,
                                dir_size: [direction[0], direction[1], args[0], args[1]],
                                weights: [args[2], args[3], args[4], args[5]],
                            });
                        }
                    }
                    if blur_vertices.len() > start {
                        steps.push(Step::Blur {
                            source,
                            target,
                            vertices: start as u64..blur_vertices.len() as u64,
                        });
                    }
                }
            }
            for blurred in [Page::Input, Page::A, Page::B] {
                let start = glow_vertices.len();
                for (i, item) in items.iter().enumerate() {
                    let Some(Filter::GlowFilter(glow)) = item.filters.get(round) else {
                        continue;
                    };
                    if at[i] != blurred {
                        continue;
                    }
                    let color = [
                        f32::from(glow.color.r) / 255.0,
                        f32::from(glow.color.g) / 255.0,
                        f32::from(glow.color.b) / 255.0,
                        f32::from(glow.color.a) / 255.0,
                    ];
                    let flag = |on: bool| if on { 1.0 } else { 0.0 };
                    let args = [
                        glow.strength.to_f32(),
                        flag(glow.is_inner()),
                        flag(glow.is_knockout()),
                        flag(glow.composite_source()),
                    ];
                    for position in quad(item) {
                        glow_vertices.push(GlowVertex {
                            position,
                            uv: position,
                            color,
                            args,
                        });
                    }
                }
                if glow_vertices.len() > start {
                    steps.push(Step::Glow {
                        blurred,
                        vertices: start as u64..glow_vertices.len() as u64,
                    });
                }
            }
            for (i, item) in items.iter().enumerate() {
                let from = match item.filters.get(round) {
                    Some(Filter::GlowFilter(_)) => Page::Glow,
                    Some(_) => at[i],
                    None => continue,
                };
                let done = round + 1 == item.filters.len();
                if done || from != Page::Input {
                    steps.push(Step::CopyOut {
                        item: i,
                        from,
                        done,
                    });
                }
            }
        }

        let page_size = wgpu::Extent3d {
            width: page_width,
            height: page_height,
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let pages: Vec<_> = (0..4)
            .map(|_| {
                texture_pool.get_texture(
                    descriptors,
                    page_size,
                    usage,
                    wgpu::TextureFormat::Rgba8Unorm,
                    1,
                )
            })
            .collect();
        let device = &descriptors.device;
        let blur_buffer = (!blur_vertices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: create_debug_label!("Batched blur vertices").as_deref(),
                contents: bytemuck::cast_slice(&blur_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let glow_buffer = (!glow_vertices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: create_debug_label!("Batched glow vertices").as_deref(),
                contents: bytemuck::cast_slice(&glow_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let batch = &descriptors.filters.batch;

        for step in steps {
            match step {
                Step::CopyIn(i) => {
                    let item = &items[i];
                    encoder.copy_texture_to_texture(
                        item.texture.as_image_copy(),
                        place(&pages[0].0, item),
                        wgpu::Extent3d {
                            width: item.width,
                            height: item.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                Step::Blur {
                    source,
                    target,
                    vertices,
                } => {
                    let Some(buffer) = &blur_buffer else { continue };
                    let binds = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: create_debug_label!("Batched blur group").as_deref(),
                        layout: &batch.blur_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &pages[source.index()].1,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    descriptors.bitmap_samplers.get_sampler(false, true),
                                ),
                            },
                        ],
                    });
                    let stride = std::mem::size_of::<BlurVertex>() as u64;
                    let mut render_pass = pass(encoder, &pages[target.index()].1);
                    render_pass.set_pipeline(batch.blur(descriptors));
                    render_pass.set_bind_group(0, &binds, &[]);
                    render_pass.set_vertex_buffer(
                        0,
                        buffer.slice(vertices.start * stride..vertices.end * stride),
                    );
                    render_pass.draw(0..(vertices.end - vertices.start) as u32, 0..1);
                }
                Step::Glow { blurred, vertices } => {
                    let Some(buffer) = &glow_buffer else { continue };
                    let binds = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: create_debug_label!("Batched glow group").as_deref(),
                        layout: &batch.glow_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&pages[0].1),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    descriptors.bitmap_samplers.get_sampler(false, false),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &pages[blurred.index()].1,
                                ),
                            },
                        ],
                    });
                    let stride = std::mem::size_of::<GlowVertex>() as u64;
                    let mut render_pass = pass(encoder, &pages[Page::Glow.index()].1);
                    render_pass.set_pipeline(batch.glow(descriptors));
                    render_pass.set_bind_group(0, &binds, &[]);
                    render_pass.set_vertex_buffer(
                        0,
                        buffer.slice(vertices.start * stride..vertices.end * stride),
                    );
                    render_pass.draw(0..(vertices.end - vertices.start) as u32, 0..1);
                }
                Step::CopyOut { item, from, done } => {
                    let item = &items[item];
                    let size = wgpu::Extent3d {
                        width: item.width,
                        height: item.height,
                        depth_or_array_layers: 1,
                    };
                    let source = place(&pages[from.index()].0, item);
                    if done {
                        encoder.copy_texture_to_texture(source, item.texture.as_image_copy(), size);
                    } else {
                        encoder.copy_texture_to_texture(source, place(&pages[0].0, item), size);
                    }
                }
            }
        }
    }
}

/// The item's place on a page, for a texture copy.
fn place<'a>(page: &'a wgpu::Texture, item: &Item) -> wgpu::TexelCopyTextureInfo<'a> {
    wgpu::TexelCopyTextureInfo {
        texture: page,
        mip_level: 0,
        origin: wgpu::Origin3d {
            x: item.x,
            y: item.y,
            z: 0,
        },
        aspect: wgpu::TextureAspect::All,
    }
}

fn pass<'a>(encoder: &'a mut wgpu::CommandEncoder, target: &wgpu::TextureView) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: create_debug_label!("Batched filter").as_deref(),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            // Other places on the page hold other bitmaps' results.
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    })
}
