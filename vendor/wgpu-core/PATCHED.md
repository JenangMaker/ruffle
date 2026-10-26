# Patched wgpu-core 30.0.1

Unmodified copy of `wgpu-core` 30.0.1 from crates.io, used through
`[patch.crates-io]` in the workspace `Cargo.toml`, with two changes. The
second, command pool recycling, is described at the end.

`src/track/metadata.rs`, `ResourceMetadata::clear`: reset only the owned slots
and keep the tables' length, instead of `resources.clear()`.

## Why

Every render/compute pass takes a pooled `UsageScope`, calls
`set_size(<highest buffer/texture index ever allocated>)` on it, and clears it
on drop. With `resources.clear()`, the clear drops every `Option<Arc<_>>` slot
and the next `set_size` refills all of them, so each pass costs
O(all live buffers + textures) regardless of what it touched.

Ruffle issues many passes per frame (masks, filters, cacheAsBitmap), and AQW
keeps a lot of buffers alive. Profiling the web build in a 10-player Battleon
put ~40% of the main thread in `drop_glue<UsageScope>`,
`Vec<Option<Arc<Buffer>>>::resize` and `BufferUsageScope::set_size`, at about
2 frames per second.

## Measured

Web build, `wgpu-webgl`, same Battleon with 10 players, Intel HD P530 via
ANGLE/Mesa, 5 s main-thread profiles:

| | before | after |
| :- | :- | :- |
| `drop_glue<UsageScope>` | 36.4% | 1.1% |
| page process CPU (Draw Max) | 65% | 37% |
| frames/s (Draw Max) | 2.2 | 2.2 |

The patch removes the CPU cost but not the frame rate: after it, the main
thread mostly waits in WebGL framebuffer calls, and halving the render
resolution raises the frame rate to 6.2 fps. The remaining limit is GPU work
in Ruffle's wgpu render path, not wgpu-core's tracking.

## Command pool recycling

`src/command/allocator.rs`: the free list of command encoders (one driver
command pool each) is capped at 8, one release in 64 destroys its pool instead
of keeping it, and `wgpu_core::command::release_free_command_pools()` drops
them all (Ruffle calls it when drawing has been paused for 10 s).

Resetting a pool (`reset_all`) keeps the memory of everything it recorded,
and the free list was never trimmed. With filters on, an AQW frame records
hundreds of passes, so on Windows (AMD iGPU, Vulkan) the pools held 0.5 to
2 GB of shared GPU memory, which is process RAM on an integrated GPU, and
never gave it back. wgpu's own allocations were 30 to 130 MB.

Measured, same login and scene (Battleon), AMD Radeon iGPU, Vulkan:

| | RAM | GPU memory |
| :- | :- | :- |
| pools kept (before) | 1.3 to 1.9 GB | 0.6 to 1.0 GB |
| every pool destroyed after use | 0.9 GB | 0.21 GB |
| same, drawing paused | 0.66 GB | 0.11 GB |

Destroying every pool halved the frame rate (17 to 7 fps), hence the 1 in 64.

## Updating

When bumping wgpu, drop this directory and the `[patch.crates-io]` entry if
upstream has fixed it; otherwise re-copy the new version and re-apply the
change (`git diff` this directory against the pristine crate to see it).
