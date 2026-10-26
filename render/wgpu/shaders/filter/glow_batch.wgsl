/// VibeSkua: glow.wgsl for many glows side by side on one page (filters/batch.rs).
/// Each one's arguments come with its vertices instead of a uniform. Only glows
/// without an offset are batched, so the blur is sampled where the source is,
/// always inside the glow's own place.

struct VertexInput {
    /// The position on the page (topleft 0,0, bottomright 1,1)
    @location(0) position: vec2<f32>,
    /// The coordinate to sample on the source and blurred pages (the same place)
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    /// strength, inner, knockout and composite_source (0.0 or 1.0), as in glow.wgsl's Filter
    @location(3) args: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) args: vec4<f32>,
};

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var blurred: texture_2d<f32>;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = vec4<f32>((in.position.x * 2.0 - 1.0), (1.0 - in.position.y * 2.0), 0.0, 1.0);
    return VertexOutput(pos, in.uv, in.color, in.args);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let strength = in.args.x;
    let inner = in.args.y > 0.5;
    let knockout = in.args.z > 0.5;
    let composite_source = in.args.w > 0.5;
    let blur = textureSampleLevel(blurred, texture_sampler, in.uv, 0.0).a;
    let dest = textureSampleLevel(texture, texture_sampler, in.uv, 0.0);

    var color = vec4<f32>(in.color.r, in.color.g, in.color.b, 1.0);
    if (inner) {
        let alpha = in.color.a * saturate((1.0 - blur) * strength);
        if (knockout) {
            color = color * alpha * dest.a;
        } else if (composite_source) {
            color = color * alpha * dest.a + dest * (1.0 - alpha);
        } else {
            color = color * alpha * dest.a;
        }
    } else {
        let alpha = in.color.a * saturate(blur * strength);
        if (knockout) {
            color = color * alpha * (1.0 - dest.a);
        } else if (composite_source) {
            color = color * alpha * (1.0 - dest.a) + dest;
        } else {
            color = color * alpha;
        }
    }

    return color;
}
