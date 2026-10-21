use alloc::{boxed::Box, vec::Vec};

use crate::lock::{rank, Mutex};

/// A pool of free [`wgpu_hal::CommandEncoder`]s, owned by a `Device`.
///
/// Each encoder in this list is in the "closed" state.
///
/// Since a raw [`CommandEncoder`][ce] is itself a pool for allocating
/// raw [`CommandBuffer`][cb]s, this is a pool of pools.
///
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
/// [ce]: hal::CommandEncoder
/// [cb]: hal::Api::CommandBuffer
pub(crate) struct CommandAllocator {
    free_encoders: Mutex<Vec<Box<dyn hal::DynCommandEncoder>>>,
}

/// VibeSkua: each free encoder is a driver command pool, and resetting one
/// (reset_all) keeps the memory of everything it ever recorded. With filters
/// on, an AQW frame records hundreds of passes, so on the AMD and Intel
/// Windows drivers the few pools in rotation held 0.5 to 2 GB of shared GPU
/// memory, which is the process's RAM on an integrated GPU, and nothing ever
/// gave it back: not idle, not headless. Measured on a 512 MB AMD iGPU, with
/// pools destroyed after use: 0.2 GB instead of 0.6 to 1 GB while drawing.
/// Destroying every pool halved the frame rate, so one release in
/// RECYCLE_EVERY destroys its pool instead (a fresh one is made when needed),
/// and release_free_command_pools() drops all of them at once.
/// A frame split over several encoders (ruffle's split_encoder.rs) uses a
/// pool for each part and each part's barriers at submit, a few dozen a
/// frame, each holding a share of the frame's commands: room for that many,
/// recycled about as often as when a frame used two or three.
const MAX_FREE_ENCODERS: usize = 48;
const RECYCLE_EVERY: usize = 512;

static RELEASES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static RELEASE_ALL: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// VibeSkua: destroy every free command pool (and the driver memory it
/// keeps) the next time an encoder is acquired or released. For an idle or
/// headless player.
pub fn release_free_command_pools() {
    RELEASE_ALL.store(true, core::sync::atomic::Ordering::Relaxed);
}

impl CommandAllocator {
    pub(crate) fn new() -> Self {
        Self {
            free_encoders: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Vec::new()),
        }
    }

    /// Return a fresh [`wgpu_hal::CommandEncoder`] in the "closed" state.
    ///
    /// If we have free encoders in the pool, take one of those. Otherwise,
    /// create a new one on `device`.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) fn acquire_encoder(
        &self,
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Box<dyn hal::DynCommandEncoder>, hal::DeviceError> {
        let mut free_encoders = self.free_encoders.lock();
        if RELEASE_ALL.swap(false, core::sync::atomic::Ordering::Relaxed) {
            free_encoders.clear();
        }
        match free_encoders.pop() {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        }
    }

    /// Add `encoder` back to the free pool.
    pub(crate) fn release_encoder(&self, encoder: Box<dyn hal::DynCommandEncoder>) {
        let mut free_encoders = self.free_encoders.lock();
        if RELEASE_ALL.swap(false, core::sync::atomic::Ordering::Relaxed) {
            free_encoders.clear();
        }
        let n = RELEASES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if free_encoders.len() < MAX_FREE_ENCODERS && n % RECYCLE_EVERY != 0 {
            free_encoders.push(encoder);
        }
        // Otherwise the encoder is dropped here, which destroys its command
        // pool and the memory the driver kept for it.
    }
}
