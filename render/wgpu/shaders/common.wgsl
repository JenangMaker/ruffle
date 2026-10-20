/// Common WGSL shared among many Ruffle shaders.
/// This is prepended to a lot of shaders at runtime before compiling them.
/// The `common__` identifier prefix serves as pseudo-namespacing.

/// Global uniforms that are constant throughout a frame.
struct common__Globals {
    // The view matrix determined by the viewport and stage.
    global_matrix: mat4x4<f32>,

    /// VibeSkua: half the viewport's width and height in pixels; zw unused.
    viewport: vec4<f32>,
};

/// Transform uniforms that are changed per object.
struct common__Transforms {
    /// The world matrix that transforms this object into stage space.
    world_matrix: mat4x4<f32>,

    /// The multiplicative color transform of this object.
    mult_color: vec4<f32>,

    /// The additive color transform of this object.
    add_color: vec4<f32>,
};

/// The vertex format shared among most shaders.
struct common__VertexInput {
    /// The position of the vertex in object space.
    @location(0) position: vec2<f32>,
};

struct common__VertexInputUv {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec3<f32>,
};

/// Common uniform layout shared by all shaders.
@group(0) @binding(0) var<uniform> common__globals: common__Globals;

/// Converts a color from linear to sRGB color space.
fn common__linear_to_srgb(linear_: vec4<f32>) -> vec4<f32> {
    var rgb: vec3<f32> = linear_.rgb;
    if (linear_.a > 0.0) {
        rgb = rgb / linear_.a;
    }
    let a = 12.92 * rgb;
    let b = 1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - 0.055;
    let c = step(vec3<f32>(0.0031308), rgb);
    return vec4<f32>(mix(a, b, c) * linear_.a, linear_.a);
}

/// Converts a color from sRGB to linear color space.
fn common__srgb_to_linear(srgb: vec4<f32>) -> vec4<f32> {
    var rgb: vec3<f32> = srgb.rgb;
    if (srgb.a > 0.0) {
        rgb = rgb / srgb.a;
    }
    let a = rgb / 12.92;
    let b = pow((rgb + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    let c = step(vec3<f32>(0.04045), rgb);
    return vec4<f32>(mix(a, b, c) * srgb.a, srgb.a);
}

/// VibeSkua: edge smoothing without multisampling (see the tessellator's
/// `add_edge_fringe`). The width, in pixels, of the fringe drawn outside each
/// edge; 0 leaves fringes as lines of no area (nothing drawn), as when a
/// shape is drawn into a mask's stencil.
override common__edge_width: f32 = 0.0;

/// VibeSkua: how much of the color a fringe has at the edge; it fades to
/// none at its outer side.
override common__edge_inner: f32 = 0.5;

/// The vertex input of shape fills drawn with a texture (gradients, bitmaps).
struct common__VertexInputUvAa {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec3<f32>,
    @location(2) aa: vec4<f32>,
};

/// VibeSkua: a vertex's clip-space position `pos`, moved out of the shape by
/// the fringe width if it is a fringe's outer vertex. `aa` is the vertex's
/// fringe data: the edge's outward normal in shape space, the side (1:
/// outer) and whether it is a fringe vertex at all. The width is kept in
/// pixels whatever the shape's scale, rotation or skew.
fn common__edge_position(pos: vec4<f32>, world_matrix: mat4x4<f32>, aa: vec4<f32>) -> vec4<f32> {
    if (common__edge_width <= 0.0 || aa.w <= 0.0 || aa.z <= 0.0) {
        return pos;
    }
    let matrix = common__globals.global_matrix * world_matrix;
    let px = common__globals.viewport.xy;
    // The edge's direction and its outward normal on screen, in pixels.
    let edge = (matrix * vec4<f32>(-aa.y, aa.x, 0.0, 0.0)).xy * px;
    let outward = (matrix * vec4<f32>(aa.x, aa.y, 0.0, 0.0)).xy * px;
    let edge_length = length(edge);
    if (edge_length < 1.0e-6) {
        return pos;
    }
    // Perpendicular to the edge on screen, on the outward side.
    var normal = vec2<f32>(edge.y, -edge.x) / edge_length;
    if (dot(normal, outward) < 0.0) {
        normal = -normal;
    }
    let offset = normal * (common__edge_width * aa.z) / px * pos.w;
    return vec4<f32>(pos.xy + offset, pos.zw);
}

/// VibeSkua: how much of the color a vertex has: 1 in the shape, the inner
/// coverage on a fringe's edge side and 0 on its outer side.
fn common__edge_coverage(aa: vec4<f32>) -> f32 {
    if (aa.w <= 0.0) {
        return 1.0;
    }
    return common__edge_inner * (1.0 - aa.z);
}
