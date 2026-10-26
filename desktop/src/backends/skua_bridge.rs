//! The Skua bridge for the desktop player: lets Skua (VibeSkua's
//! Skua.Ruffle/RuffleBridge.cs) drive skua.swf over a WebSocket, as
//! web/public/skua-bridge.js does in VibeSkua's browser build. Same protocol:
//!
//!   Skua -> player  {"id":1,"fn":"getGameObject","args":["world.strMapName"]}
//!   player -> Skua  {"id":1,"ok":true,"value":...} | {"id":1,"ok":false,"error":"..."}
//!   player -> Skua  {"ev":"packetFromServer","args":[...]}
//!
//! Skua's calls run the SWF's ExternalInterface callbacks; the SWF's own
//! ExternalInterface.call()s are forwarded to Skua as events. As the page
//! does, `requestLoadGame` (skua.swf asking for the game) is answered here by
//! calling the SWF's `loadClient`, whether or not Skua is connected.
//!
//! The player lives on the event loop's thread, so the bridge thread only
//! owns the socket: each call goes to the event loop as a
//! [`RuffleEvent::SkuaBridgeCall`], which runs it ([`handle_call`]) and queues
//! the reply for the bridge thread to send, as the SWF's events are queued.
//!
//! Enabled by SKUA_BRIDGE_URL (e.g. ws://127.0.0.1:8790/).

use crate::backends::DesktopExternalInterfaceProvider;
use crate::custom_event::RuffleEvent;
use ruffle_core::Player;
use ruffle_core::context::UpdateContext;
use ruffle_core::events::{
    KeyDescriptor, KeyLocation, LogicalKey, NamedKey, PhysicalKey, PlayerEvent, TextControlCode,
};
use ruffle_core::external::{ExternalInterfaceProvider, Value as ExternalValue};
use serde_json::{Map, Value as Json, json};
use std::collections::{BTreeMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
#[cfg(windows)]
use std::net::TcpStream as WakeStream;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::net::UnixStream as WakeStream;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};
use tungstenite::{Message, WebSocket};
use winit::event_loop::EventLoopProxy;

/// Events kept while Skua is not connected, as the page keeps them.
const MAX_QUEUED: usize = 2000;
/// The call the bridge makes itself when skua.swf asks for the game.
const LOAD_CLIENT: &str = r#"{"id":-1,"fn":"loadClient","args":[]}"#;

/// Drawing control (the page's maxRenderFps): paused draws nothing; the cap
/// is frames drawn per second, 0 for no cap. The game runs either way.
static DRAWING_PAUSED: AtomicBool = AtomicBool::new(false);
/// When drawing was paused (`Instant` of the process start + ms), 0 while
/// drawing; and whether that pause's resources were released (release_due).
static PAUSED_AT_MS: AtomicU64 = AtomicU64::new(0);
static PAUSE_RELEASED: AtomicBool = AtomicBool::new(false);

fn clock_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

fn set_paused(paused: bool) {
    let was = DRAWING_PAUSED.swap(paused, Ordering::Relaxed);
    if paused && !was {
        PAUSED_AT_MS.store(clock_ms(), Ordering::Relaxed);
        PAUSE_RELEASED.store(false, Ordering::Relaxed);
    } else if !paused {
        PAUSED_AT_MS.store(0, Ordering::Relaxed);
    }
}

/// True once per pause, when drawing has been paused (a hidden or headless
/// tab) for 10 s: then the player hands back what only drawing needs
/// (`Player::release_render_resources`). Not at once: switching between tabs
/// pauses and resumes them, and everything released is made again on the
/// first frame drawn.
pub fn release_due() -> bool {
    let at = PAUSED_AT_MS.load(Ordering::Relaxed);
    at != 0
        && clock_ms().saturating_sub(at) >= 10_000
        && !PAUSE_RELEASED.swap(true, Ordering::Relaxed)
}
static RENDER_CAP_FPS: AtomicU32 = AtomicU32::new(0);
static LAST_RENDER: Mutex<Option<Instant>> = Mutex::new(None);
/// The game's current frame rate (stage.frameRate) in hundredths of a frame a
/// second, kept up to date by the window before each draw (see may_render).
static GAME_FPS_CENTI: AtomicU32 = AtomicU32::new(0);
/// How long the last frame took to draw, in microseconds (see rendered()).
static RENDER_COST_US: AtomicU64 = AtomicU64::new(0);
/// Drawing's share of the main thread (see may_render): when it was last
/// topped up, and the drawing time left, in ms (spent by rendered()).
static DRAW_CREDIT: Mutex<(Option<Instant>, f64)> = Mutex::new((None, 0.0));
/// At most this much drawing time is saved up, in ms (a few frames' worth).
const DRAW_CREDIT_MAX_MS: f64 = 100.0;

/// RUFFLE_DRAW_SHARE (0.2-0.9, default 0.6): the share of the main thread
/// drawing may use on average.
fn draw_share() -> f64 {
    static SHARE: OnceLock<f64> = OnceLock::new();
    *SHARE.get_or_init(|| {
        std::env::var("RUFFLE_DRAW_SHARE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| (0.2..=0.9).contains(v))
            .unwrap_or(0.6)
    })
}

/// The player window's X11 id (on Windows its HWND), for Skua to embed it
/// (page.windowId), as main.js's /game-window gave the Electron window's.
static WINDOW_ID: OnceLock<u64> = OnceLock::new();

pub fn set_window(window: &winit::window::Window) {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let id = match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Xlib(h)) => Some(h.window as u64),
        Ok(RawWindowHandle::Xcb(h)) => Some(h.window.get() as u64),
        Ok(RawWindowHandle::Win32(h)) => Some(h.hwnd.get() as u64),
        _ => None,
    };
    match id {
        Some(id) => {
            let _ = WINDOW_ID.set(id);
        }
        None => tracing::warn!("skua bridge: not an X11 or Win32 window; Skua cannot embed it"),
    }
}

/// Windows: gives the keyboard to the player window when it is embedded (a
/// child of Skua's window, in another process). A click there activates the
/// top-level window, which keeps the focus: nothing moves it to the child, so
/// the game got no keys. Parenting joined the two processes' input, so the
/// child may take the focus itself; called on each mouse press.
#[cfg(windows)]
pub fn take_focus_if_embedded(window: &winit::window::Window) {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetFocus, SetFocus};
    use windows_sys::Win32::UI::WindowsAndMessaging::{GWL_STYLE, GetWindowLongPtrW, WS_CHILD};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let Ok(RawWindowHandle::Win32(handle)) = window.window_handle().map(|h| h.as_raw()) else {
        return;
    };
    let hwnd = handle.hwnd.get() as windows_sys::Win32::Foundation::HWND;
    // SAFETY: plain Win32 calls on our own window's handle.
    unsafe {
        let child = GetWindowLongPtrW(hwnd, GWL_STYLE) as u32 & WS_CHILD != 0;
        if child && GetFocus() != hwnd {
            SetFocus(hwnd);
        }
    }
}

/// Whether drawing is paused: the window then asks for no redraws at all.
pub fn drawing_paused() -> bool {
    DRAWING_PAUSED.load(Ordering::Relaxed)
}

/// The game's frame rate, for may_render (stage.frameRate: 24 in AQW, 30
/// while a script runs with Skua's default FPS option).
pub fn set_game_frame_rate(fps: f64) {
    if fps.is_finite() && fps > 0.0 {
        GAME_FPS_CENTI.store((fps * 100.0) as u32, Ordering::Relaxed);
    }
}

/// Whether the window may draw a frame now (asked before each draw).
///
/// Skua's calls arrive one at a time, each waiting for the last one's answer,
/// and the event loop draws between them whenever the movie has changed: with
/// nothing to space the frames out, every call waited for a whole frame to be
/// drawn, so a Skua status read of ~20 calls took 9-15 s and Headless Mode
/// timed out. Now:
/// - at most one picture per game frame, as Flash draws (and the original
///   VibeSkua: 24 a second, 30 with a script running), or the setRender cap if
///   lower; more pictures than game frames only repeat themselves;
/// - drawing gets at most RUFFLE_DRAW_SHARE (60%) of the main thread, on
///   average: each millisecond adds that share to a drawing allowance (up to
///   DRAW_CREDIT_MAX_MS), each frame drawn spends its time, and none is drawn
///   while it is spent. This was "the next frame no sooner than twice the last
///   one's time after it": checked frame by frame, any frame a little slower
///   than usual (17 ms at the game's 30 frames a second) made the next game
///   frame too early and skipped its picture, so a crowded room showed 19.5
///   pictures of 28.9 game frames, unevenly.
/// A frame may start a little early (3/4 of the frame time), as the game's
/// frames do not come exactly on time and a strict wait would skip every other.
pub fn may_render() -> bool {
    if DRAWING_PAUSED.load(Ordering::Relaxed) {
        return false;
    }
    let Ok(mut last) = LAST_RENDER.lock() else {
        return true;
    };
    let now = Instant::now();
    let mut fps = 60.0_f64;
    let game = GAME_FPS_CENTI.load(Ordering::Relaxed);
    if game > 0 {
        fps = fps.min(game as f64 / 100.0);
    }
    let cap = RENDER_CAP_FPS.load(Ordering::Relaxed);
    if cap > 0 {
        fps = fps.min(cap as f64);
    }
    let frame = Duration::from_secs_f64(0.75 / fps.max(1.0));
    if let Some(previous) = *last
        && now.duration_since(previous) < frame
    {
        return false;
    }
    if let Ok(mut credit) = DRAW_CREDIT.lock() {
        if let Some(topped) = credit.0 {
            let gained = now.duration_since(topped).as_secs_f64() * 1000.0 * draw_share();
            credit.1 = (credit.1 + gained).min(DRAW_CREDIT_MAX_MS);
        }
        credit.0 = Some(now);
        if credit.1 < 0.0 {
            return false;
        }
    }
    *last = Some(now);
    true
}

/// Parts of drawing a frame, timed for the drawing summary (see rendered).
#[derive(Clone, Copy)]
pub enum Stage {
    /// The movie drawn into its offscreen texture (player.render()).
    Movie = 0,
    /// Getting the window's next buffer.
    Acquire = 1,
    /// Handing the frame's commands to the GPU.
    Submit = 2,
    /// Presenting the buffer to the window (the X server).
    Present = 3,
}

/// Drawing totals since the last summary: frames, total and worst frame time,
/// and the time per stage, in microseconds.
struct DrawStats {
    since: Instant,
    frames: u64,
    total_us: u64,
    max_us: u64,
    stage_us: [u64; 4],
    /// CPU time the main thread itself spent presenting (see note_present_cpu).
    present_cpu_us: u64,
}

static DRAW_STATS: Mutex<Option<DrawStats>> = Mutex::new(None);

/// Adds the time one stage of the current frame took (see Stage).
pub fn note_stage(stage: Stage, took: Duration) {
    if let Ok(mut guard) = DRAW_STATS.lock() {
        guard.get_or_insert_with(DrawStats::new).stage_us[stage as usize] +=
            took.as_micros() as u64;
    }
}

impl DrawStats {
    fn new() -> Self {
        DrawStats {
            since: Instant::now(),
            frames: 0,
            total_us: 0,
            max_us: 0,
            stage_us: [0; 4],
            present_cpu_us: 0,
        }
    }
}

/// CPU time used by the calling thread so far.
#[cfg(unix)]
pub fn thread_cpu_time() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime only writes the timespec it is given.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// CPU time used by the calling thread so far.
#[cfg(windows)]
pub fn thread_cpu_time() -> Duration {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = || FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut created, mut exited, mut kernel, mut user) = (zero(), zero(), zero(), zero());
    // SAFETY: GetThreadTimes only writes the four FILETIMEs it is given.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) };
    let ticks = |t: FILETIME| ((t.dwHighDateTime as u64) << 32) | t.dwLowDateTime as u64;
    // In 100 ns units.
    Duration::from_nanos((ticks(kernel) + ticks(user)) * 100)
}

/// Adds the CPU time the main thread spent presenting a frame: compared with
/// the present stage's wall time, it tells working from waiting.
pub fn note_present_cpu(took: Duration) {
    if let Ok(mut guard) = DRAW_STATS.lock() {
        guard.get_or_insert_with(DrawStats::new).present_cpu_us += took.as_micros() as u64;
    }
}

/// Each thread's CPU time (clock ticks) at the last summary, by thread id.
static THREAD_TICKS: Mutex<Option<std::collections::HashMap<u32, u64>>> = Mutex::new(None);

/// The player's threads' CPU use since the last call, summed by thread name,
/// busiest first, as "name 12%" (of one core over `span`). Linux only (from
/// /proc); elsewhere empty.
fn thread_cpu_summary(span: Duration) -> String {
    #[cfg(unix)]
    let hz = (unsafe { libc::sysconf(libc::_SC_CLK_TCK) }).max(1) as f64;
    #[cfg(not(unix))]
    let hz = 100.0;
    let mut now = std::collections::HashMap::new();
    let mut by_name: Vec<(String, u64)> = Vec::new();
    let Ok(mut guard) = THREAD_TICKS.lock() else {
        return String::new();
    };
    let before = guard.take().unwrap_or_default();
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let Ok(tid) = task.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else {
                continue;
            };
            // "tid (name) state ... utime stime ...": fields counted after the name.
            let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else {
                continue;
            };
            let name = stat[open + 1..close].to_string();
            let fields: Vec<&str> = stat[close + 2..].split(' ').collect();
            let ticks = fields
                .get(11)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
                + fields
                    .get(12)
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0);
            now.insert(tid, ticks);
            let used = ticks.saturating_sub(*before.get(&tid).unwrap_or(&0));
            match by_name.iter_mut().find(|(n, _)| *n == name) {
                Some((_, sum)) => *sum += used,
                None => by_name.push((name, used)),
            }
        }
    }
    *guard = Some(now);
    if before.is_empty() {
        return "(first sample)".to_string();
    }
    by_name.sort_by(|a, b| b.1.cmp(&a.1));
    by_name
        .iter()
        .filter(|(_, used)| *used > 0)
        .take(8)
        .map(|(name, used)| {
            format!(
                "{name} {:.0}%",
                *used as f64 / hz / span.as_secs_f64() * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Called after each frame is drawn with how long it took (see may_render).
/// While a tab draws, logs a summary every minute: frames per second, the
/// average and worst frame time, and where the time went.
pub fn rendered(cost: Duration) {
    note_time("draw", cost);
    RENDER_COST_US.store(cost.as_micros() as u64, Ordering::Relaxed);
    // One very slow frame (a map loading) must not pause drawing for long:
    // the allowance goes no lower than its maximum below zero.
    if let Ok(mut credit) = DRAW_CREDIT.lock() {
        credit.1 = (credit.1 - cost.as_secs_f64() * 1000.0).max(-DRAW_CREDIT_MAX_MS);
    }
    if let Ok(mut guard) = DRAW_STATS.lock() {
        let stats = guard.get_or_insert_with(DrawStats::new);
        let us = cost.as_micros() as u64;
        stats.frames += 1;
        stats.total_us += us;
        stats.max_us = stats.max_us.max(us);
        let span = stats.since.elapsed();
        if span >= Duration::from_secs(60) {
            let n = stats.frames.max(1) as f64;
            let ms = |us: u64| us as f64 / 1000.0 / n;
            tracing::warn!(
                "skua bridge: drew {} frames in {:.0} s ({:.1} fps), {:.1} ms a frame (worst {:.0} ms): movie {:.1}, acquire {:.1}, submit {:.1}, present {:.1} ms",
                stats.frames,
                span.as_secs_f64(),
                stats.frames as f64 / span.as_secs_f64(),
                ms(stats.total_us),
                stats.max_us as f64 / 1000.0,
                ms(stats.stage_us[0]),
                ms(stats.stage_us[1]),
                ms(stats.stage_us[2]),
                ms(stats.stage_us[3]),
            );
            tracing::warn!(
                "skua bridge: presenting used {:.1} ms of the main thread's CPU a frame; player threads: {}",
                ms(stats.present_cpu_us),
                thread_cpu_summary(span),
            );
            {
                use ruffle_core::skua_stats as g;
                use std::sync::atomic::Ordering::Relaxed;
                let secs = span.as_secs_f64();
                // Game frames run in the same minute: more than the pictures
                // drawn means a tick ran several frames before one picture.
                let frames_run = {
                    static LAST: AtomicU64 = AtomicU64::new(0);
                    let now = g::FRAMES.load(Relaxed);
                    now - LAST.swap(now, Relaxed)
                };
                tracing::warn!(
                    "skua bridge: game frames run: {:.1} a second ({:.1} pictures drawn)",
                    frames_run as f64 / secs,
                    stats.frames as f64 / span.as_secs_f64(),
                );
                tracing::warn!(
                    "skua bridge: gotos a second: {:.0} no-op, {:.0} inner passes taking {:.0} ms a second; {} orphans",
                    g::NOOP_GOTOS.swap(0, Relaxed) as f64 / secs,
                    g::INNER_GOTOS.swap(0, Relaxed) as f64 / secs,
                    g::INNER_GOTO_US.swap(0, Relaxed) as f64 / 1000.0 / secs,
                    g::ORPHANS.load(Relaxed),
                );
            }
            *guard = None;
        }
    }
    if cost >= Duration::from_millis(500) {
        tracing::warn!("skua bridge: a frame took {} ms to draw", cost.as_millis());
    }
}

/// Skua modules switched off or on once the game has loaded, as the page does
/// (DISABLE_MODULES / ENABLE_MODULES, comma separated). QuestRequirementWiki
/// and QuestItemRates throw on every frame when no game UI is up.
fn module_calls() -> Vec<String> {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let disable = list("DISABLE_MODULES", "QuestRequirementWiki,QuestItemRates");
    let enable = list("ENABLE_MODULES", "");
    disable
        .iter()
        .map(|m| json!({ "id": -1, "fn": "modDisable", "args": [m] }).to_string())
        .chain(
            enable
                .iter()
                .map(|m| json!({ "id": -1, "fn": "modEnable", "args": [m] }).to_string()),
        )
        .collect()
}

/// SKUA_BRIDGE_PROFILE=1: logs, every 200 calls, the average time a call
/// waits for the event loop, runs, and waits to be sent.
struct Profile {
    received: VecDeque<Instant>,
    replied: VecDeque<Instant>,
    sums: [f64; 3],
    count: u32,
}
static PROFILE: Mutex<Option<Profile>> = Mutex::new(None);

fn profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SKUA_BRIDGE_PROFILE").is_ok_and(|v| v == "1"))
}

fn profile(f: impl FnOnce(&mut Profile)) {
    if !profiling() {
        return;
    }
    if let Ok(mut p) = PROFILE.lock() {
        f(p.get_or_insert_with(|| Profile {
            received: VecDeque::new(),
            replied: VecDeque::new(),
            sums: [0.0; 3],
            count: 0,
        }));
    }
}

/// Messages for Skua (replies and the SWF's events), and a pipe that wakes the
/// bridge thread when one is queued. The thread sleeps in poll() on the socket
/// and this pipe: a timed read instead held each reply for up to a kernel
/// timer tick (7 ms measured), most of a call's time.
static OUTGOING: OnceLock<(mpsc::Sender<String>, WakeStream)> = OnceLock::new();

/// The wake-up pipe: a socket pair on Unix. Windows has no socket pairs, and
/// WSAPoll waits only on sockets, so there two loopback TCP sockets.
#[cfg(unix)]
fn wake_pair() -> std::io::Result<(WakeStream, WakeStream)> {
    WakeStream::pair()
}

#[cfg(windows)]
fn wake_pair() -> std::io::Result<(WakeStream, WakeStream)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let a = TcpStream::connect(listener.local_addr()?)?;
    let (b, from) = listener.accept()?;
    // Some other local program could have connected first.
    if from != a.local_addr()? {
        return Err(std::io::Error::other("unexpected connection to the wake-up socket"));
    }
    a.set_nodelay(true)?;
    Ok((a, b))
}

/// Whether the Skua that started this player is still there. On Unix the
/// player is re-parented when its parent dies; Windows does not track parents,
/// so there Skua passes its process id (SKUA_PARENT_PID) and this checks that
/// process (without it, always true).
#[cfg(unix)]
fn parent_watch() -> impl Fn() -> bool {
    // SAFETY: getppid has no preconditions.
    let parent = unsafe { libc::getppid() };
    move || (unsafe { libc::getppid() }) == parent
}

#[cfg(windows)]
fn parent_watch() -> impl Fn() -> bool {
    use windows_sys::Win32::Foundation::WAIT_TIMEOUT;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };
    let handle = std::env::var("SKUA_PARENT_PID")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        // SAFETY: OpenProcess has no preconditions; a null handle means it failed.
        .map(|pid| unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) } as usize)
        .filter(|h| *h != 0);
    // The handle stays open for the player's life, so the id cannot be reused.
    move || match handle {
        // SAFETY: the handle is a process handle opened above, never closed.
        Some(h) => (unsafe { WaitForSingleObject(h as _, 0) }) == WAIT_TIMEOUT,
        None => true,
    }
}

/// Sleeps until the wake-up pipe or the socket can be read (or the socket
/// written, with `write`), or `timeout_ms` passes (-1: no limit).
#[cfg(unix)]
fn wait(wake: &WakeStream, socket: Option<&TcpStream>, write: bool, timeout_ms: i32) {
    let mut fds = vec![libc::pollfd {
        fd: wake.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    }];
    if let Some(socket) = socket {
        let events = libc::POLLIN | if write { libc::POLLOUT } else { 0 };
        fds.push(libc::pollfd {
            fd: socket.as_raw_fd(),
            events,
            revents: 0,
        });
    }
    // SAFETY: fds is a valid, initialised pollfd array for the whole call.
    unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout_ms) };
}

#[cfg(windows)]
fn wait(wake: &WakeStream, socket: Option<&TcpStream>, write: bool, timeout_ms: i32) {
    use windows_sys::Win32::Networking::WinSock::{POLLRDNORM, POLLWRNORM, WSAPOLLFD, WSAPoll};
    let mut fds = vec![WSAPOLLFD {
        fd: wake.as_raw_socket() as _,
        events: POLLRDNORM,
        revents: 0,
    }];
    if let Some(socket) = socket {
        fds.push(WSAPOLLFD {
            fd: socket.as_raw_socket() as _,
            events: POLLRDNORM | if write { POLLWRNORM } else { 0 },
            revents: 0,
        });
    }
    // SAFETY: fds is a valid, initialised WSAPOLLFD array for the whole call.
    unsafe { WSAPoll(fds.as_mut_ptr(), fds.len() as u32, timeout_ms) };
}

fn send_to_skua(text: String) {
    if let Some((tx, wake)) = OUTGOING.get()
        && tx.send(text).is_ok()
    {
        // A full pipe already has a wake-up pending.
        let _ = (&*wake).write(&[1]);
    }
}

pub struct SkuaBridgeProvider {
    event_loop: EventLoopProxy<RuffleEvent>,
    fallback: DesktopExternalInterfaceProvider,
}

impl ExternalInterfaceProvider for SkuaBridgeProvider {
    fn call_method(
        &self,
        context: &mut UpdateContext<'_>,
        name: &str,
        args: &[ExternalValue],
    ) -> ExternalValue {
        // Location lookups and eval keep the desktop player's answers.
        if name == "eval" || name.ends_with(".toString") {
            return self.fallback.call_method(context, name, args);
        }
        let args: Vec<Json> = args.iter().map(to_json).collect();
        send_to_skua(json!({ "ev": name, "args": args }).to_string());
        if name == "requestLoadGame" {
            // The SWF is running right now; load the game once it is done.
            tracing::info!("skua bridge: skua.swf is ready, loading the game client");
            let _ = self
                .event_loop
                .send_event(RuffleEvent::SkuaBridgeCall(LOAD_CLIENT.to_string()));
        }
        if name == "loaded" {
            for call in module_calls() {
                let _ = self
                    .event_loop
                    .send_event(RuffleEvent::SkuaBridgeCall(call));
            }
        }
        ExternalValue::Bool(true)
    }

    fn on_callback_available(&self, _name: &str) {}

    fn get_id(&self) -> Option<String> {
        None
    }
}

/// Creates the ExternalInterface provider and starts the bridge thread.
pub fn start(
    url: String,
    event_loop: EventLoopProxy<RuffleEvent>,
    fallback: DesktopExternalInterfaceProvider,
) -> SkuaBridgeProvider {
    // RUFFLE_MAX_FPS (1-60): a starting cap on the pictures drawn a second,
    // until Skua sets another (page.setRender). Unset, one per game frame.
    if let Some(fps) = std::env::var("RUFFLE_MAX_FPS")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|fps| (1..=60).contains(fps))
    {
        RENDER_CAP_FPS.store(fps, Ordering::Relaxed);
    }
    let (tx, rx) = mpsc::channel();
    let (wake_tx, wake_rx) = wake_pair().expect("could not create the Skua bridge wake-up pipe");
    let _ = wake_tx.set_nonblocking(true);
    let _ = wake_rx.set_nonblocking(true);
    if OUTGOING.set((tx, wake_tx)).is_err() {
        tracing::warn!("skua bridge: already running; the new player sends nothing");
    }
    let proxy = event_loop.clone();
    std::thread::Builder::new()
        .name("skua-bridge".into())
        .spawn(move || run(&url, rx, wake_rx, proxy))
        .expect("could not start the Skua bridge thread");
    SkuaBridgeProvider {
        event_loop,
        fallback,
    }
}

fn run(
    url: &str,
    rx: mpsc::Receiver<String>,
    mut wake: WakeStream,
    event_loop: EventLoopProxy<RuffleEvent>,
) {
    let mut socket: Option<WebSocket<TcpStream>> = None;
    let mut queued: VecDeque<String> = VecDeque::new();
    let mut backoff = Duration::from_secs(1);
    let mut next_try = Instant::now();
    // Connecting, failing and losing the connection are logged as warnings:
    // VibeSkua records only those from here, and a player that never reached
    // Skua (GitHub issue #2: the game ran, Skua could not drive it) said
    // nothing about why.
    let drop_socket = |socket: &mut Option<WebSocket<TcpStream>>, why: String| {
        if socket.take().is_some() {
            tracing::warn!("skua bridge disconnected ({why}); retrying");
        }
    };
    let mut failed_tries = 0u32;
    // Skua started this player; if Skua goes away without ending it (killed outright),
    // nothing drives the player any more: exit.
    let parent_alive = parent_watch();
    let would_block = |e: &tungstenite::Error| matches!(e, tungstenite::Error::Io(e) if e.kind() == ErrorKind::WouldBlock);
    loop {
        // Clear the wake-ups; the channel holds the messages.
        let mut buf = [0u8; 64];
        while matches!(wake.read(&mut buf), Ok(n) if n > 0) {}

        if socket.is_none() && !parent_alive() {
            tracing::info!("skua bridge: Skua is gone; exiting");
            std::process::exit(0);
        }

        // Replies and events since the last pass.
        loop {
            match rx.try_recv() {
                Ok(text) => match socket.as_mut() {
                    Some(ws) => {
                        if text.starts_with("{\"id\"") {
                            profile_sent();
                        }
                        // A full socket buffer keeps the frame for flush() below.
                        if let Err(e) = ws.write(Message::text(text))
                            && !would_block(&e)
                        {
                            drop_socket(&mut socket, e.to_string());
                        }
                    }
                    // Replies to a lost connection are dropped with it; events
                    // wait, as the page keeps them.
                    None if text.starts_with("{\"ev\"") && queued.len() < MAX_QUEUED => {
                        queued.push_back(text)
                    }
                    None => {}
                },
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }

        if socket.is_none() && Instant::now() >= next_try {
            match connect(url) {
                Ok(mut ws) => {
                    if failed_tries > 0 {
                        tracing::warn!("skua bridge connected: {url}, after {failed_tries} failed tries");
                    } else {
                        tracing::warn!("skua bridge connected: {url}");
                    }
                    failed_tries = 0;
                    backoff = Duration::from_secs(1);
                    let mut ok = true;
                    while let Some(text) = queued.pop_front() {
                        if let Err(e) = ws.write(Message::text(text))
                            && !would_block(&e)
                        {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        socket = Some(ws);
                    }
                }
                Err(e) => {
                    // The first failure, then about one a minute (tries are
                    // at most 15 s apart).
                    if failed_tries % 4 == 0 {
                        tracing::warn!(
                            "skua bridge: {url} not reachable ({e}); retrying ({} failed so far)",
                            failed_tries + 1
                        );
                    }
                    failed_tries += 1;
                    next_try = Instant::now() + backoff;
                    backoff = (backoff * 2).min(Duration::from_secs(15));
                }
            }
        }

        // Read every complete frame there is, then send what is buffered.
        if let Some(ws) = socket.as_mut() {
            loop {
                match ws.read() {
                    Ok(Message::Text(text)) => {
                        profile(|p| p.received.push_back(Instant::now()));
                        if event_loop
                            .send_event(RuffleEvent::SkuaBridgeCall(text.as_str().to_string()))
                            .is_err()
                        {
                            return; // the player has closed
                        }
                    }
                    Ok(Message::Close(_)) => {
                        drop_socket(&mut socket, "closed by Skua".into());
                        break;
                    }
                    Ok(_) => {}
                    Err(e) if would_block(&e) => break,
                    Err(e) => {
                        drop_socket(&mut socket, e.to_string());
                        break;
                    }
                }
            }
        }
        let mut wants_write = false;
        if let Some(ws) = socket.as_mut() {
            match ws.flush() {
                Ok(()) => {}
                Err(e) if would_block(&e) => wants_write = true,
                Err(e) => drop_socket(&mut socket, e.to_string()),
            }
        }

        // Sleep until Skua sends something, a message is queued, or (while
        // disconnected) the next connection attempt is due.
        let timeout_ms: i32 = match &socket {
            Some(_) => -1,
            None => next_try
                .saturating_duration_since(Instant::now())
                .as_millis()
                .clamp(1, 15_000) as i32,
        };
        wait(
            &wake,
            socket.as_ref().map(|ws| ws.get_ref()),
            wants_write,
            timeout_ms,
        );
    }
}

/// Profiling: a reply is being sent (see `profile`).
fn profile_sent() {
    profile(|p| {
        if let Some(at) = p.replied.pop_front() {
            p.sums[2] += at.elapsed().as_secs_f64() * 1000.0;
            p.count += 1;
            if p.count == 200 {
                let n = p.count as f64;
                tracing::info!(
                    "skua bridge: per call {:.2} ms to the event loop, {:.2} ms running, {:.2} ms to the socket",
                    p.sums[0] / n,
                    p.sums[1] / n,
                    p.sums[2] / n
                );
                p.sums = [0.0; 3];
                p.count = 0;
            }
        }
    });
}

fn connect(url: &str) -> Result<WebSocket<TcpStream>, String> {
    let parsed = url::Url::parse(url).map_err(|e| e.to_string())?;
    let host = parsed.host_str().ok_or("no host")?;
    let port = parsed.port_or_known_default().ok_or("no port")?;
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| e.to_string())?
        .next()
        .ok_or("no address")?;
    let stream =
        TcpStream::connect_timeout(&addr, Duration::from_secs(2)).map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    // The handshake reads a blocking socket: without a limit, a listener that
    // took the connection and never answered (one stuck after a player
    // crashed while connecting, or another program on the port) held this
    // thread for good, and the player never reached Skua or said why.
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    let (ws, _) = tungstenite::client::client(url, stream).map_err(|e| e.to_string())?;
    ws.get_ref()
        .set_read_timeout(None)
        .map_err(|e| e.to_string())?;
    ws.get_ref()
        .set_write_timeout(None)
        .map_err(|e| e.to_string())?;
    ws.get_ref()
        .set_nonblocking(true)
        .map_err(|e| e.to_string())?;
    Ok(ws)
}

/// Runs one call from Skua on the event loop's thread and queues the reply.
pub fn handle_call(player: &mut Player, text: &str) {
    let started = Instant::now();
    let Ok(message) = serde_json::from_str::<Json>(text) else {
        return;
    };
    let (Some(id), Some(name)) = (
        message.get("id").and_then(Json::as_f64),
        message.get("fn").and_then(Json::as_str),
    ) else {
        return;
    };
    let args: Vec<ExternalValue> = match message.get("args") {
        Some(Json::Array(args)) => args.iter().map(from_json).collect(),
        _ => Vec::new(),
    };
    // A paused (hidden or headless) window has no pointer over it. Ruffle runs a mouse
    // pick (a hit test of the whole display tree) after every call into the movie
    // while it thinks the pointer is over the stage, and only a pointer leaving the
    // window tells it otherwise: a window parked under a resting pointer made every
    // call cost a pick (14 ms in Battleon on a software renderer).
    if name == "page.pauseDrawing" && matches!(args.first(), Some(ExternalValue::Bool(true))) {
        player.set_mouse_in_stage(false);
        player.handle_event(ruffle_core::events::PlayerEvent::MouseLeave);
    }
    let result = match name.strip_prefix("page.") {
        Some(page) => page_call(page, &args, player),
        None if player.has_internal_interface(name) => {
            Ok(to_json(&player.call_internal_interface(name, args)))
        }
        None => Err(format!("no SWF callback named {name}")),
    };
    if id < 0.0 {
        return; // the bridge's own calls (loadClient, modules)
    }
    profile(|p| {
        if let Some(at) = p.received.pop_front() {
            p.sums[0] += started.duration_since(at).as_secs_f64() * 1000.0;
        }
        p.sums[1] += started.elapsed().as_secs_f64() * 1000.0;
        p.replied.push_back(Instant::now());
    });
    let reply = match result {
        Ok(value) => json!({ "id": id as i64, "ok": true, "value": value }),
        Err(error) => json!({ "id": id as i64, "ok": false, "error": error }),
    };
    let text = reply.to_string();
    let key = reply_key(name, message.get("args").and_then(|a| a.get(0)).and_then(Json::as_str));
    note_reply(&key, text.len());
    note_time(&key, started.elapsed());
    send_to_skua(text);
}

fn reply_key(name: &str, first: Option<&str>) -> String {
    match first {
        Some(arg) if arg.len() <= 60 => format!("{name}({arg})"),
        _ => name.to_string(),
    }
}

/// With SKUA_MEMORY_STATS on: where the game thread's time goes, per minute:
/// each Skua call (by function and first argument), the game's own ticks
/// ("tick") and drawing ("draw").
static TIMES: Mutex<Option<std::collections::HashMap<String, (u64, Duration)>>> = Mutex::new(None);

pub fn note_time(key: &str, took: Duration) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("SKUA_MEMORY_STATS").is_some_and(|v| v != "0")) {
        return;
    }
    if let Ok(mut map) = TIMES.lock() {
        let entry = map.get_or_insert_with(Default::default).entry(key.to_string()).or_default();
        entry.0 += 1;
        entry.1 += took;
    }
}

/// This process's CPU time (all threads), from /proc/self/stat.
fn process_cpu() -> Duration {
    let ticks = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let rest = s.rsplit_once(')')?.1.split_whitespace().collect::<Vec<_>>();
            Some(rest.get(11)?.parse::<u64>().ok()? + rest.get(12)?.parse::<u64>().ok()?)
        })
        .unwrap_or(0);
    Duration::from_millis(ticks * 10)
}

/// The game's frames over the last minute: how many, ms per second in them by
/// phase, in goto passes, timers and socket data, and the listener counts.
fn frame_breakdown() -> String {
    use ruffle_core::skua_stats as st;
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: Mutex<Option<[u64; 11]>> = Mutex::new(None);
    let now = [
        st::FRAMES.load(Relaxed),
        st::FRAME_US.load(Relaxed),
        st::PHASE_US[0].load(Relaxed),
        st::PHASE_US[1].load(Relaxed),
        st::PHASE_US[2].load(Relaxed),
        st::PHASE_US[3].load(Relaxed),
        st::INNER_GOTOS.load(Relaxed),
        st::INNER_GOTO_US.load(Relaxed),
        st::NOOP_GOTOS.load(Relaxed),
        st::TIMERS_US.load(Relaxed),
        st::SOCKETS_US.load(Relaxed),
    ];
    let was = LAST.lock().ok().and_then(|mut l| l.replace(now)).unwrap_or([0; 11]);
    let d: Vec<u64> = now.iter().zip(was.iter()).map(|(a, b)| a.saturating_sub(*b)).collect();
    let ms = |us: u64| us as f64 / 1000.0 / 60.0;
    format!(
        "frames {} ({:.0} ms/s: enter {:.0}, construct {:.0}, scripts {:.0}, exit {:.0}), goto passes {} ({:.0} ms/s, {} no-op), timers {:.0} ms/s, sockets {:.0} ms/s, orphans {}, listeners enterFrame {} exitFrame {} frameConstructed {}",
        d[0], ms(d[1]), ms(d[2]), ms(d[3]), ms(d[4]), ms(d[5]), d[6], ms(d[7]), d[8], ms(d[9]), ms(d[10]),
        st::ORPHANS.load(Relaxed),
        st::LISTENERS[0].load(Relaxed), st::LISTENERS[1].load(Relaxed), st::LISTENERS[2].load(Relaxed),
    )
}

/// The last minute: process CPU, then the game thread's busy time by kind,
/// calls by total time.
fn take_times() -> String {
    static LAST: Mutex<Option<(Instant, Duration)>> = Mutex::new(None);
    let now = (Instant::now(), process_cpu());
    let cpu = match LAST.lock().ok().and_then(|mut l| l.replace(now)) {
        Some((at, was)) => format!("{:.0}%", (now.1.saturating_sub(was)).as_secs_f64() * 100.0 / at.elapsed().as_secs_f64().max(1.0)),
        None => "?".into(),
    };
    let Some(map) = TIMES.lock().ok().and_then(|mut m| m.take()) else {
        return format!("process cpu {cpu}");
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let total = |pred: &dyn Fn(&str) -> bool| map.iter().filter(|(k, _)| pred(k)).map(|(_, v)| ms(v.1)).sum::<f64>();
    let ticks = total(&|k| k == "tick");
    let draw = total(&|k| k == "draw");
    let calls = total(&|k| k != "tick" && k != "draw");
    let ncalls: u64 = map.iter().filter(|(k, _)| *k != "tick" && *k != "draw").map(|(_, v)| v.0).sum();
    let mut top: Vec<_> = map.iter().filter(|(k, _)| *k != "tick" && *k != "draw").collect();
    top.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    let top: Vec<String> = top.iter().take(6).map(|(k, (n, d))| format!("{k} {n}x {:.0} ms", ms(*d))).collect();
    let frame = frame_breakdown();
    format!(
        "process cpu {cpu}; game thread busy {:.0} ms of each 1000: calls {:.0} ({} calls), ticks {:.0}, draw {:.0}; {frame}; top calls: {}",
        (ticks + draw + calls) / 60.0,
        calls / 60.0,
        ncalls,
        ticks / 60.0,
        draw / 60.0,
        top.join(", ")
    )
}

/// With SKUA_MEMORY_STATS on: calls and reply bytes per function (and first
/// argument, the object path of getGameObject), for the memory line. A reply
/// is a string the game built and dropped.
static REPLIES: Mutex<Option<std::collections::HashMap<String, (u64, u64)>>> = Mutex::new(None);

fn note_reply(key: &str, bytes: usize) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("SKUA_MEMORY_STATS").is_some_and(|v| v != "0")) {
        return;
    }
    if let Ok(mut map) = REPLIES.lock() {
        let entry = map.get_or_insert_with(Default::default).entry(key.to_string()).or_default();
        entry.0 += 1;
        entry.1 += bytes as u64;
    }
}

/// The last minute's replies: total, then the five largest by bytes.
fn take_replies() -> String {
    let Some(map) = REPLIES.lock().ok().and_then(|mut m| m.take()) else {
        return "-".into();
    };
    let calls: u64 = map.values().map(|v| v.0).sum();
    let bytes: u64 = map.values().map(|v| v.1).sum();
    let mut top: Vec<_> = map.into_iter().collect();
    top.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    let top: Vec<String> = top
        .iter()
        .take(5)
        .map(|(k, (n, b))| format!("{k} {n}x {:.1} MB", *b as f64 / 1048576.0))
        .collect();
    format!("{calls} calls, {:.1} MB: {}", bytes as f64 / 1048576.0, top.join(", "))
}

/// The page's own functions (window.vibeskua in the browser build):
/// pauseDrawing(paused), setRender({fps}) and getRender(). TabThrottle
/// pauses drawing for hidden and headless tabs. VibeSkua's desktop host also
/// uses `keyEvent` and `textInput` to forward keyboard events it caught on
/// its own window into the player, side-stepping the GNOME/Mutter XWayland
/// keyboard-routing limitation across clients (see GameEmbed).
fn page_call(name: &str, args: &[ExternalValue], player: &mut Player) -> Result<Json, String> {
    match name {
        "pauseDrawing" => {
            let paused = matches!(args.first(), Some(ExternalValue::Bool(true)));
            set_paused(paused);
            Ok(Json::Bool(true))
        }
        "setRender" => {
            if let Some(ExternalValue::Object(options)) = args.first() {
                match options.get("fps") {
                    Some(ExternalValue::Number(fps)) if fps.is_finite() && *fps >= 1.0 => {
                        RENDER_CAP_FPS.store(*fps as u32, Ordering::Relaxed)
                    }
                    Some(ExternalValue::Number(fps)) if *fps <= 0.0 => {
                        set_paused(true)
                    }
                    Some(_) => RENDER_CAP_FPS.store(0, Ordering::Relaxed),
                    None => {}
                }
            }
            Ok(Json::Bool(true))
        }
        "windowId" => Ok(WINDOW_ID
            .get()
            .map_or(Json::Null, |id| json!(id.to_string()))),
        "getRender" => {
            let cap = RENDER_CAP_FPS.load(Ordering::Relaxed);
            let fps = if DRAWING_PAUSED.load(Ordering::Relaxed) {
                json!(0)
            } else if cap == 0 {
                Json::Null
            } else {
                json!(cap)
            };
            Ok(json!({ "fps": fps, "scale": 1, "paused": DRAWING_PAUSED.load(Ordering::Relaxed) }))
        }
        "keyEvent" => handle_key_event(args, player),
        "textInput" => handle_text_input(args, player),
        _ => Err(format!("no page function named {name}")),
    }
}

/// Forwards a key event sent by VibeSkua's host from its own window into the
/// player. GNOME/Mutter routes keyboard events by Wayland surface, which
/// belongs to the host (one X11 toplevel = one Wayland surface), so an X11
/// sub-window owned by another client (us) never sees them directly even when
/// X11 input focus is on it. The host subscribes to KeyDown/KeyUp on its
/// Avalonia window and relays each one here while the pointer is over the
/// game area. Expected args[0]:
///   { state: "down"|"up", code: "<PhysicalKey name>", key: "<char or NamedKey>",
///     location: "standard"|"left"|"right"|"numpad" }
/// The strings follow the DOM UIEvents spec (same spellings as our enums).
fn handle_key_event(args: &[ExternalValue], player: &mut Player) -> Result<Json, String> {
    let Some(ExternalValue::Object(obj)) = args.first() else {
        return Err("expected an object".into());
    };
    let get_str = |k: &str| match obj.get(k) {
        Some(ExternalValue::String(s)) => Some(s.as_str()),
        _ => None,
    };
    let state = get_str("state").ok_or_else(|| "missing state".to_string())?;
    let physical_key = parse_physical_key(get_str("code").unwrap_or(""));
    let logical_key = parse_logical_key(get_str("key").unwrap_or(""));
    let key_location = match get_str("location").unwrap_or("standard") {
        "left" => KeyLocation::Left,
        "right" => KeyLocation::Right,
        "numpad" => KeyLocation::Numpad,
        _ => KeyLocation::Standard,
    };
    let key = KeyDescriptor { physical_key, logical_key, key_location };
    match state {
        "down" => {
            let _ = player.handle_event(PlayerEvent::KeyDown { key });
            if let Some(code) = text_control_from_key(&key, obj) {
                let _ = player.handle_event(PlayerEvent::TextControl { code });
            }
        }
        "up" => { let _ = player.handle_event(PlayerEvent::KeyUp { key }); }
        _ => return Err(format!("state must be \"down\" or \"up\", not {state:?}")),
    }
    Ok(Json::Bool(true))
}

/// Pairs with KeyDown for printable characters, as app.rs::winit_to_ruffle_text_control
/// does: Ctrl+A/C/V/X, cursor moves inside text fields.
fn text_control_from_key(key: &KeyDescriptor, modifiers: &BTreeMap<String, ExternalValue>) -> Option<TextControlCode> {
    let is = |k: &str| matches!(modifiers.get(k), Some(ExternalValue::Bool(true)));
    let shift = is("shift");
    let ctrl_cmd = is("ctrl") || (is("meta") && cfg!(target_os = "macos"));
    match key.logical_key {
        LogicalKey::Named(NamedKey::Enter) => Some(TextControlCode::Enter),
        LogicalKey::Character('a') | LogicalKey::Character('A') if ctrl_cmd => Some(TextControlCode::SelectAll),
        LogicalKey::Character('c') | LogicalKey::Character('C') if ctrl_cmd => Some(TextControlCode::Copy),
        LogicalKey::Character('v') | LogicalKey::Character('V') if ctrl_cmd => Some(TextControlCode::Paste),
        LogicalKey::Character('x') | LogicalKey::Character('X') if ctrl_cmd => Some(TextControlCode::Cut),
        LogicalKey::Named(NamedKey::Backspace) if ctrl_cmd => Some(TextControlCode::BackspaceWord),
        LogicalKey::Named(NamedKey::Backspace) => Some(TextControlCode::Backspace),
        LogicalKey::Named(NamedKey::Delete) if ctrl_cmd => Some(TextControlCode::DeleteWord),
        LogicalKey::Named(NamedKey::Delete) => Some(TextControlCode::Delete),
        LogicalKey::Named(NamedKey::ArrowLeft) if ctrl_cmd && shift => Some(TextControlCode::SelectLeftWord),
        LogicalKey::Named(NamedKey::ArrowLeft) if ctrl_cmd => Some(TextControlCode::MoveLeftWord),
        LogicalKey::Named(NamedKey::ArrowLeft) if shift => Some(TextControlCode::SelectLeft),
        LogicalKey::Named(NamedKey::ArrowLeft) => Some(TextControlCode::MoveLeft),
        LogicalKey::Named(NamedKey::ArrowRight) if ctrl_cmd && shift => Some(TextControlCode::SelectRightWord),
        LogicalKey::Named(NamedKey::ArrowRight) if ctrl_cmd => Some(TextControlCode::MoveRightWord),
        LogicalKey::Named(NamedKey::ArrowRight) if shift => Some(TextControlCode::SelectRight),
        LogicalKey::Named(NamedKey::ArrowRight) => Some(TextControlCode::MoveRight),
        LogicalKey::Named(NamedKey::Home) => Some(TextControlCode::MoveLeftLine),
        LogicalKey::Named(NamedKey::End) => Some(TextControlCode::MoveRightLine),
        _ => None,
    }
}

/// Each character goes to the player as a `TextInput` event, as the browser
/// build's keyboard input does.
fn handle_text_input(args: &[ExternalValue], player: &mut Player) -> Result<Json, String> {
    let Some(ExternalValue::String(text)) = args.first() else {
        return Err("expected a string".into());
    };
    for codepoint in text.chars() {
        let _ = player.handle_event(PlayerEvent::TextInput { codepoint });
    }
    Ok(Json::Bool(true))
}

/// DOM `KeyboardEvent.code` string (e.g. "KeyA", "Digit1", "Enter",
/// "ShiftLeft") to our PhysicalKey. Variant names follow the same spec, so
/// this is mostly a stringify-the-enum mapping. Unknown codes map to
/// Unknown, which the player tolerates: logical_key still carries the text.
fn parse_physical_key(code: &str) -> PhysicalKey {
    match code {
        "Backquote" => PhysicalKey::Backquote,
        "Digit0" => PhysicalKey::Digit0, "Digit1" => PhysicalKey::Digit1,
        "Digit2" => PhysicalKey::Digit2, "Digit3" => PhysicalKey::Digit3,
        "Digit4" => PhysicalKey::Digit4, "Digit5" => PhysicalKey::Digit5,
        "Digit6" => PhysicalKey::Digit6, "Digit7" => PhysicalKey::Digit7,
        "Digit8" => PhysicalKey::Digit8, "Digit9" => PhysicalKey::Digit9,
        "Minus" => PhysicalKey::Minus, "Equal" => PhysicalKey::Equal,
        "KeyA" => PhysicalKey::KeyA, "KeyB" => PhysicalKey::KeyB,
        "KeyC" => PhysicalKey::KeyC, "KeyD" => PhysicalKey::KeyD,
        "KeyE" => PhysicalKey::KeyE, "KeyF" => PhysicalKey::KeyF,
        "KeyG" => PhysicalKey::KeyG, "KeyH" => PhysicalKey::KeyH,
        "KeyI" => PhysicalKey::KeyI, "KeyJ" => PhysicalKey::KeyJ,
        "KeyK" => PhysicalKey::KeyK, "KeyL" => PhysicalKey::KeyL,
        "KeyM" => PhysicalKey::KeyM, "KeyN" => PhysicalKey::KeyN,
        "KeyO" => PhysicalKey::KeyO, "KeyP" => PhysicalKey::KeyP,
        "KeyQ" => PhysicalKey::KeyQ, "KeyR" => PhysicalKey::KeyR,
        "KeyS" => PhysicalKey::KeyS, "KeyT" => PhysicalKey::KeyT,
        "KeyU" => PhysicalKey::KeyU, "KeyV" => PhysicalKey::KeyV,
        "KeyW" => PhysicalKey::KeyW, "KeyX" => PhysicalKey::KeyX,
        "KeyY" => PhysicalKey::KeyY, "KeyZ" => PhysicalKey::KeyZ,
        "BracketLeft" => PhysicalKey::BracketLeft, "BracketRight" => PhysicalKey::BracketRight,
        "Backslash" => PhysicalKey::Backslash, "Semicolon" => PhysicalKey::Semicolon,
        "Quote" => PhysicalKey::Quote, "Comma" => PhysicalKey::Comma,
        "Period" => PhysicalKey::Period, "Slash" => PhysicalKey::Slash,
        "Backspace" => PhysicalKey::Backspace, "Tab" => PhysicalKey::Tab,
        "CapsLock" => PhysicalKey::CapsLock, "Enter" => PhysicalKey::Enter,
        "ShiftLeft" => PhysicalKey::ShiftLeft, "ShiftRight" => PhysicalKey::ShiftRight,
        "ControlLeft" => PhysicalKey::ControlLeft, "ControlRight" => PhysicalKey::ControlRight,
        "AltLeft" => PhysicalKey::AltLeft, "AltRight" => PhysicalKey::AltRight,
        "SuperLeft" | "MetaLeft" | "OSLeft" => PhysicalKey::SuperLeft,
        "SuperRight" | "MetaRight" | "OSRight" => PhysicalKey::SuperRight,
        "Space" => PhysicalKey::Space, "ContextMenu" => PhysicalKey::ContextMenu,
        "Insert" => PhysicalKey::Insert, "Delete" => PhysicalKey::Delete,
        "Home" => PhysicalKey::Home, "End" => PhysicalKey::End,
        "PageUp" => PhysicalKey::PageUp, "PageDown" => PhysicalKey::PageDown,
        "ArrowUp" => PhysicalKey::ArrowUp, "ArrowDown" => PhysicalKey::ArrowDown,
        "ArrowLeft" => PhysicalKey::ArrowLeft, "ArrowRight" => PhysicalKey::ArrowRight,
        "NumLock" => PhysicalKey::NumLock,
        "NumpadDivide" => PhysicalKey::NumpadDivide,
        "NumpadMultiply" => PhysicalKey::NumpadMultiply,
        "NumpadSubtract" => PhysicalKey::NumpadSubtract,
        "NumpadAdd" => PhysicalKey::NumpadAdd,
        "NumpadEnter" => PhysicalKey::NumpadEnter,
        "NumpadDecimal" => PhysicalKey::NumpadDecimal,
        "Numpad0" => PhysicalKey::Numpad0, "Numpad1" => PhysicalKey::Numpad1,
        "Numpad2" => PhysicalKey::Numpad2, "Numpad3" => PhysicalKey::Numpad3,
        "Numpad4" => PhysicalKey::Numpad4, "Numpad5" => PhysicalKey::Numpad5,
        "Numpad6" => PhysicalKey::Numpad6, "Numpad7" => PhysicalKey::Numpad7,
        "Numpad8" => PhysicalKey::Numpad8, "Numpad9" => PhysicalKey::Numpad9,
        "Escape" => PhysicalKey::Escape,
        "F1" => PhysicalKey::F1, "F2" => PhysicalKey::F2,
        "F3" => PhysicalKey::F3, "F4" => PhysicalKey::F4,
        "F5" => PhysicalKey::F5, "F6" => PhysicalKey::F6,
        "F7" => PhysicalKey::F7, "F8" => PhysicalKey::F8,
        "F9" => PhysicalKey::F9, "F10" => PhysicalKey::F10,
        "F11" => PhysicalKey::F11, "F12" => PhysicalKey::F12,
        "PrintScreen" => PhysicalKey::PrintScreen,
        "ScrollLock" => PhysicalKey::ScrollLock,
        "Pause" => PhysicalKey::Pause,
        _ => PhysicalKey::Unknown,
    }
}

/// DOM `KeyboardEvent.key` string to our LogicalKey. A single character maps
/// to Character(char); anything else goes through parse_named_key.
fn parse_logical_key(key: &str) -> LogicalKey {
    if key.is_empty() {
        return LogicalKey::Unknown;
    }
    let mut chars = key.chars();
    let first = chars.next().unwrap();
    if chars.next().is_none() {
        return LogicalKey::Character(first);
    }
    match parse_named_key(key) {
        Some(named) => LogicalKey::Named(named),
        None => LogicalKey::Unknown,
    }
}

fn parse_named_key(name: &str) -> Option<NamedKey> {
    Some(match name {
        "Alt" => NamedKey::Alt, "AltGraph" => NamedKey::AltGraph,
        "CapsLock" => NamedKey::CapsLock, "Control" => NamedKey::Control,
        "Fn" => NamedKey::Fn, "FnLock" => NamedKey::FnLock,
        "Meta" | "Super" | "OS" => NamedKey::Super,
        "NumLock" => NamedKey::NumLock, "ScrollLock" => NamedKey::ScrollLock,
        "Shift" => NamedKey::Shift,
        "Enter" => NamedKey::Enter, "Tab" => NamedKey::Tab,
        "ArrowDown" => NamedKey::ArrowDown, "ArrowLeft" => NamedKey::ArrowLeft,
        "ArrowRight" => NamedKey::ArrowRight, "ArrowUp" => NamedKey::ArrowUp,
        "End" => NamedKey::End, "Home" => NamedKey::Home,
        "PageDown" => NamedKey::PageDown, "PageUp" => NamedKey::PageUp,
        "Backspace" => NamedKey::Backspace, "Clear" => NamedKey::Clear,
        "Copy" => NamedKey::Copy, "Cut" => NamedKey::Cut,
        "Delete" => NamedKey::Delete, "Insert" => NamedKey::Insert,
        "Paste" => NamedKey::Paste, "Redo" => NamedKey::Redo, "Undo" => NamedKey::Undo,
        "ContextMenu" => NamedKey::ContextMenu, "Escape" => NamedKey::Escape,
        "Pause" => NamedKey::Pause, "Play" => NamedKey::Play, "Select" => NamedKey::Select,
        "PrintScreen" => NamedKey::PrintScreen,
        "F1" => NamedKey::F1, "F2" => NamedKey::F2,
        "F3" => NamedKey::F3, "F4" => NamedKey::F4,
        "F5" => NamedKey::F5, "F6" => NamedKey::F6,
        "F7" => NamedKey::F7, "F8" => NamedKey::F8,
        "F9" => NamedKey::F9, "F10" => NamedKey::F10,
        "F11" => NamedKey::F11, "F12" => NamedKey::F12,
        _ => return None,
    })
}

fn to_json(value: &ExternalValue) -> Json {
    match value {
        ExternalValue::Undefined | ExternalValue::Null => Json::Null,
        ExternalValue::Bool(b) => Json::Bool(*b),
        // Whole numbers as integers, as JSON.stringify writes them: Skua
        // parses IDs from these.
        ExternalValue::Number(n) if n.fract() == 0.0 && n.abs() < 9.0e15 => json!(*n as i64),
        ExternalValue::Number(n) => {
            serde_json::Number::from_f64(*n).map_or(Json::Null, Json::Number)
        }
        ExternalValue::String(s) => Json::String(s.clone()),
        ExternalValue::List(items) => Json::Array(items.iter().map(to_json).collect()),
        ExternalValue::Object(map) => Json::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), to_json(v)))
                .collect::<Map<_, _>>(),
        ),
    }
}

fn from_json(value: &Json) -> ExternalValue {
    match value {
        Json::Null => ExternalValue::Null,
        Json::Bool(b) => ExternalValue::Bool(*b),
        Json::Number(n) => ExternalValue::Number(n.as_f64().unwrap_or(0.0)),
        Json::String(s) => ExternalValue::String(s.clone()),
        Json::Array(items) => ExternalValue::List(items.iter().map(from_json).collect()),
        Json::Object(map) => ExternalValue::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), from_json(v)))
                .collect::<BTreeMap<_, _>>(),
        ),
    }
}

/// SKUA_MEMORY_STATS=1: once a minute, a line on what the player holds, for
/// hunting what grows over a session (map changes, other players' gear):
/// resident memory, live GC objects, the loaded SWFs still alive (count,
/// bytes, the kinds with the most copies), orphaned clips, and what kept the
/// libraries of loaded movies alive at the last collection.
pub fn memory_stats_due() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    if !*ON.get_or_init(|| std::env::var_os("SKUA_MEMORY_STATS").is_some_and(|v| v != "0")) {
        return false;
    }
    let Ok(mut last) = LAST.lock() else { return false };
    match *last {
        Some(t) if t.elapsed() < Duration::from_secs(60) => false,
        _ => {
            *last = Some(Instant::now());
            true
        }
    }
}

/// This process's resident memory in MB (0 if /proc can't say).
#[cfg(windows)]
fn rss_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: an all-zero PROCESS_MEMORY_COUNTERS is valid; the call fills it in.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: counters is as large as the size passed.
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
        return 0;
    }
    // The working set: what Windows calls resident.
    counters.WorkingSetSize as u64 / (1024 * 1024)
}

#[cfg(not(windows))]
fn rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<u64>().ok()))
        .map(|pages| pages * 4096 / (1024 * 1024))
        .unwrap_or(0)
}

/// Hands memory the allocator holds but nothing uses back to the system:
/// glibc keeps freed pages in its heaps, so a player's RSS stayed near its
/// peak. After a full collection (System.gc(), which AQW calls after every map
/// change), and, for scripts that stay in one map, once its RSS has grown
/// 150 MB since the last trim (checked once a minute). With SKUA_MEMORY_STATS
/// on, logs what it gave back. RUFFLE_MALLOC_TRIM=0 turns it off.
pub fn trim_after_full_gc() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static ON: OnceLock<bool> = OnceLock::new();
        static LAST_CHECK: Mutex<Option<Instant>> = Mutex::new(None);
        // RSS after the last trim (0: none yet).
        static BASE_MB: AtomicU64 = AtomicU64::new(0);
        if !*ON.get_or_init(|| std::env::var("RUFFLE_MALLOC_TRIM").map_or(true, |v| v.trim() != "0")) {
            return;
        }
        let n = ruffle_core::FULL_COLLECTIONS.load(Ordering::Relaxed);
        let reason = if SEEN.swap(n, Ordering::Relaxed) != n {
            "after System.gc"
        } else {
            let Ok(mut last) = LAST_CHECK.lock() else { return };
            if last.is_some_and(|t| t.elapsed() < Duration::from_secs(60)) {
                return;
            }
            *last = Some(Instant::now());
            let base = BASE_MB.load(Ordering::Relaxed);
            if base == 0 {
                BASE_MB.store(rss_mb(), Ordering::Relaxed);
                return;
            }
            if rss_mb() < base + 150 {
                return;
            }
            "grew 150 MB"
        };
        // On its own thread: trimming a big heap took 0.4 s in a test, and
        // the game thread only waits for glibc's locks meanwhile. One at a time.
        static TRIMMING: AtomicBool = AtomicBool::new(false);
        if TRIMMING.swap(true, Ordering::AcqRel) {
            return;
        }
        let spawned = std::thread::Builder::new().name("malloc-trim".into()).spawn(move || {
            let before = rss_mb();
            let started = Instant::now();
            // SAFETY: malloc_trim only releases free memory; it takes glibc's locks.
            unsafe { libc::malloc_trim(0) };
            let after = rss_mb();
            BASE_MB.store(after, Ordering::Relaxed);
            if std::env::var_os("SKUA_MEMORY_STATS").is_some_and(|v| v != "0") {
                tracing::warn!(
                    "[memory] {}: malloc_trim gave back {} MB ({} -> {} MB) in {} ms",
                    reason,
                    before.saturating_sub(after),
                    before,
                    after,
                    started.elapsed().as_millis()
                );
            }
            TRIMMING.store(false, Ordering::Release);
        });
        if spawned.is_err() {
            TRIMMING.store(false, Ordering::Release);
        }
    }
}

/// Where the resident memory is: anonymous (heap) vs file and shared mappings
/// (the GPU driver's buffers are shared), and of glibc's heap how much is in
/// use vs freed but kept by the allocator (what malloc_trim can give back).
fn rss_split() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map_or(0, |kb| kb / 1024)
    };
    let mut out = format!(
        "anon {} MB, file {} MB, shmem {} MB",
        field("RssAnon:"),
        field("RssFile:"),
        field("RssShmem:")
    );
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // mallinfo, not mallinfo2 (glibc 2.33+): int fields, fine below 2 GB.
        #[allow(deprecated)]
        // SAFETY: mallinfo only reads glibc's allocator statistics.
        let m = unsafe { libc::mallinfo() };
        let mb = |b: libc::c_int| (b as u32 as u64) / 1048576;
        out.push_str(&format!(
            "; malloc: in use {} MB, free kept {} MB, mmapped {} MB",
            mb(m.uordblks),
            mb(m.fordblks),
            mb(m.hblkhd)
        ));
    }
    out
}

pub fn log_memory_stats(stats_json: &str) {
    let rss_mb = rss_mb();
    let Ok(v) = serde_json::from_str::<serde_json::Value>(stats_json) else {
        return;
    };
    let movies = v["movies"].as_array().cloned().unwrap_or_default();
    let count: u64 = movies.iter().filter_map(|m| m["count"].as_u64()).sum();
    let bytes: u64 = movies.iter().filter_map(|m| m["bytes"].as_u64()).sum();
    // The kinds of asset with the most live copies: the folder part of the
    // URL ("maps", "items/swords", ...) groups a kind.
    let mut kinds: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
    for m in &movies {
        let url = m["url"].as_str().unwrap_or("");
        let path = url.split("/gamefiles/").nth(1).unwrap_or(url);
        let kind = path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("(top)").to_string();
        let e = kinds.entry(kind).or_default();
        e.0 += m["count"].as_u64().unwrap_or(0);
        e.1 += m["bytes"].as_u64().unwrap_or(0);
    }
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    let top = kinds
        .iter()
        .take(6)
        .map(|(k, (c, b))| format!("{k} {c} ({:.1} MB)", *b as f64 / 1048576.0))
        .collect::<Vec<_>>()
        .join(", ");
    let lib = &v["lastGcLibraries"];
    let kept = lib["kept"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(reason, k)| format!("{reason} {}", k["count"]))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    tracing::warn!(
        "skua bridge: memory: rss {} MB ({}), {} GC objects, {} orphans, GC cycles {} (overdue help {} ms, totals); {} SWFs alive ({:.1} MB): {}; last GC libraries: {} collectable, {} dropped, kept by {}",
        rss_mb,
        rss_split(),
        v["gcObjects"],
        v["orphans"],
        v["gcCycles"],
        v["gcHelpMs"],
        count,
        bytes as f64 / 1048576.0,
        top,
        lib["collectable"],
        lib["dropped"],
        if kept.is_empty() { "-".to_string() } else { kept },
    );
    let kinds: Vec<String> = v["displayKinds"]
        .as_array()
        .map(|a| a.iter().map(|k| format!("{} {}", k["kind"].as_str().unwrap_or("?"), k["count"])).collect())
        .unwrap_or_default();
    tracing::warn!("skua bridge: display tree {} objects: {}", v["displayObjects"], kinds.join(", "));
    // Its own line: the container log cuts long lines.
    tracing::warn!("skua bridge: replies to Skua in the last minute: {}", take_replies());
    tracing::warn!("skua bridge: time in the last minute: {}", take_times());
}
