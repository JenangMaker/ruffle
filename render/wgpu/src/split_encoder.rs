//! VibeSkua: a frame's commands recorded into several encoders, finished on
//! several threads.
//!
//! wgpu (30) only records commands as they are made; `CommandEncoder::finish`
//! then validates them, tracks the resources they use and encodes them for
//! the driver. A crowded AQW frame is a few hundred passes and thousands of
//! draws, and that finish took 16-55 ms on one core of a Ryzen 5500U while the
//! other cores idled: most of a frame's drawing time.
//!
//! While a frame is drawn (`begin`/`end`), `cut` points between passes swap the
//! encoder being recorded into for a fresh one and keep the full one. At submit
//! `finish_all` finishes all of them at once, one thread per group, and returns
//! the command buffers in recording order for a single `queue.submit`: the
//! GPU gets the same commands in the same order.
//!
//! Only the frame's own encoder may be cut (`render_offscreen` and the like
//! record into encoders submitted on their own), so cuts happen only between
//! `begin` and `end`, and every submit of the frame's encoder goes through
//! `finish_all`. `RUFFLE_SPLIT_ENCODER=off` keeps one encoder.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

/// Cut points between cuts: a few dozen passes per encoder, enough to keep the
/// cost of each extra encoder small and few enough that a heavy frame spreads
/// over the cores. `RUFFLE_SPLIT_EVERY` sets another count (1 cuts at every
/// point, as the image tests are run to check the split).
fn cut_every() -> u32 {
    static EVERY: OnceLock<u32> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("RUFFLE_SPLIT_EVERY")
            .ok()
            .and_then(|n| n.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(16)
    })
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static SINCE_CUT: Cell<u32> = const { Cell::new(0) };
    static FULL: RefCell<Vec<wgpu::CommandEncoder>> = const { RefCell::new(Vec::new()) };
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !cfg!(target_family = "wasm")
            && std::env::var("RUFFLE_SPLIT_ENCODER").as_deref() != Ok("off")
    })
}

/// The frame's encoder may be cut from now on.
pub fn begin() {
    ACTIVE.with(|active| active.set(enabled()));
    SINCE_CUT.with(|since| since.set(0));
}

/// No more cuts: other encoders are recorded after this.
pub fn end() {
    ACTIVE.with(|active| active.set(false));
}

/// A point between passes of the frame's encoder: every `cut_every()` of them
/// the encoder is swapped for a fresh one.
pub fn cut(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
    if !ACTIVE.with(Cell::get) {
        return;
    }
    let since = SINCE_CUT.with(|since| {
        let n = since.get() + 1;
        since.set(n);
        n
    });
    if since < cut_every() {
        return;
    }
    SINCE_CUT.with(|since| since.set(0));
    let full = std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
    FULL.with(|encoders| encoders.borrow_mut().push(full));
}

/// The command buffers of everything recorded since the last submit: the
/// encoders cut off so far, then `last`, in that order.
pub fn finish_all(last: wgpu::CommandEncoder) -> Vec<wgpu::CommandBuffer> {
    let mut encoders = FULL.with(|encoders| std::mem::take(&mut *encoders.borrow_mut()));
    SINCE_CUT.with(|since| since.set(0));
    encoders.push(last);
    if encoders.len() == 1 {
        return encoders.into_iter().map(|e| e.finish()).collect();
    }
    finish_parallel(encoders)
}

#[cfg(not(target_family = "wasm"))]
fn finish_parallel(encoders: Vec<wgpu::CommandEncoder>) -> Vec<wgpu::CommandBuffer> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
        .min(encoders.len());
    // Consecutive encoders per thread; the last group runs on this thread.
    let per_thread = encoders.len().div_ceil(threads);
    let mut groups: Vec<Vec<wgpu::CommandEncoder>> = Vec::with_capacity(threads);
    let mut rest = encoders.into_iter();
    loop {
        let group: Vec<_> = rest.by_ref().take(per_thread).collect();
        if group.is_empty() {
            break;
        }
        groups.push(group);
    }
    let last = groups.pop().unwrap_or_default();
    std::thread::scope(|scope| {
        let handles: Vec<_> = groups
            .into_iter()
            .map(|group| {
                scope.spawn(move || group.into_iter().map(|e| e.finish()).collect::<Vec<_>>())
            })
            .collect();
        let last: Vec<_> = last.into_iter().map(|e| e.finish()).collect();
        let mut buffers = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(group) => buffers.extend(group),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        buffers.extend(last);
        buffers
    })
}

#[cfg(target_family = "wasm")]
fn finish_parallel(encoders: Vec<wgpu::CommandEncoder>) -> Vec<wgpu::CommandBuffer> {
    encoders.into_iter().map(|e| e.finish()).collect()
}
