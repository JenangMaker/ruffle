/// VibeSkua: blur.wgsl for many blurs side by side on one page (filters/batch.rs).
/// Each one's arguments come with its vertices instead of a uniform, and its
/// samples are clamped to its own place on the page, as the sampler clamped them
/// to the edges of its own texture.

struct VertexInput {
    /// The position on the page (topleft 0,0, bottomright 1,1)
    @location(0) position: vec2<f32>,
    /// The coordinate to sample on the page the blur reads (the same place)
    @location(1) uv: vec2<f32>,
    /// The centers of the place's first and last pixels: min x, min y, max x, max y
    @location(2) clamp_rect: vec4<f32>,
    /// dir_x, dir_y, full_size and m, as in blur.wgsl's Filter
    @location(3) dir_size: vec4<f32>,
    /// m2, first_weight, last_offset and last_weight, as in blur.wgsl's Filter
    @location(4) weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) clamp_rect: vec4<f32>,
    @location(2) @interpolate(flat) dir_size: vec4<f32>,
    @location(3) @interpolate(flat) weights: vec4<f32>,
};

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = vec4<f32>((in.position.x * 2.0 - 1.0), (1.0 - in.position.y * 2.0), 0.0, 1.0);
    let direction = in.dir_size.xy;
    // As blur.wgsl: the first trivially sampled pixel's center at 0.0.
    let uv = in.uv - direction * in.dir_size.w;
    return VertexOutput(pos, uv, in.clamp_rect, in.dir_size, in.weights);
}

// The page has one mip level, so sampling level 0 is what textureSample does;
// the loop's bound differs between blurs, which textureSample does not allow.
fn sample_at(in: VertexOutput, coord: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(texture, texture_sampler, clamp(coord, in.clamp_rect.xy, in.clamp_rect.zw), 0.0);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let direction = in.dir_size.xy;
    let full_size = in.dir_size.z;
    let m2 = in.weights.x;
    let first_weight = in.weights.y;
    let last_offset = in.weights.z;
    let last_weight = in.weights.w;

    var total = vec4<f32>(0.0);
    total += sample_at(in, in.uv - direction) * first_weight;

    var center = vec4<f32>();
    for (var i = 0.5; i < m2; i += 2.0) {
        center += sample_at(in, in.uv + direction * i);
    }
    total += center * 2.0;

    let last_location = in.uv + direction * (m2 + last_offset);
    total += sample_at(in, last_location) * last_weight;

    let result = total / full_size;
    return floor(result * 255.0) / 255.0;
}
