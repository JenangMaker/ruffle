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
use ruffle_core::external::{ExternalInterfaceProvider, Value as ExternalValue};
use serde_json::{Map, Value as Json, json};
use std::collections::{BTreeMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
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
static RENDER_CAP_FPS: AtomicU32 = AtomicU32::new(0);
static LAST_RENDER: Mutex<Option<Instant>> = Mutex::new(None);
/// How long the last frame took to draw, in microseconds (see rendered()).
static RENDER_COST_US: AtomicU64 = AtomicU64::new(0);

/// The player window's X11 id, for Skua to embed it (page.windowId), as
/// main.js's /game-window gave the Electron window's.
static WINDOW_ID: OnceLock<u64> = OnceLock::new();

pub fn set_window(window: &winit::window::Window) {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let id = match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Xlib(h)) => Some(h.window as u64),
        Ok(RawWindowHandle::Xcb(h)) => Some(h.window.get() as u64),
        _ => None,
    };
    match id {
        Some(id) => {
            let _ = WINDOW_ID.set(id);
        }
        None => tracing::warn!("skua bridge: not an X11 window; Skua cannot embed it"),
    }
}

/// Whether drawing is paused: the window then asks for no redraws at all.
pub fn drawing_paused() -> bool {
    DRAWING_PAUSED.load(Ordering::Relaxed)
}

/// Whether the window may draw a frame now (asked before each draw).
///
/// Skua's calls arrive one at a time, each waiting for the last one's answer,
/// and the event loop draws between them whenever the movie has changed: with
/// nothing to space the frames out, every call waited for a whole frame to be
/// drawn (0.3-0.5 s on a software renderer), so a Skua status read of ~20 calls
/// took 9-15 s and Headless Mode, itself a few calls, timed out. A browser draws
/// at most once per display frame. Here a frame comes at most 60 times a second
/// (or the setRender cap), and drawing gets at most half of the main thread: after
/// a frame that took D, the next one starts no sooner than 2*D after it.
pub fn may_render() -> bool {
    if DRAWING_PAUSED.load(Ordering::Relaxed) {
        return false;
    }
    let cap = RENDER_CAP_FPS.load(Ordering::Relaxed);
    let Ok(mut last) = LAST_RENDER.lock() else {
        return true;
    };
    let now = Instant::now();
    let frame = Duration::from_secs_f64(1.0 / if cap > 0 { cap.min(60) } else { 60 } as f64);
    let busy = Duration::from_micros(RENDER_COST_US.load(Ordering::Relaxed) * 2);
    if let Some(previous) = *last
        && now.duration_since(previous) < frame.max(busy)
    {
        return false;
    }
    *last = Some(now);
    true
}

/// Called after each frame is drawn with how long it took (see may_render).
pub fn rendered(cost: Duration) {
    RENDER_COST_US.store(cost.as_micros() as u64, Ordering::Relaxed);
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
static OUTGOING: OnceLock<(mpsc::Sender<String>, UnixStream)> = OnceLock::new();

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
    let (tx, rx) = mpsc::channel();
    let (wake_tx, wake_rx) =
        UnixStream::pair().expect("could not create the Skua bridge wake-up pipe");
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
    mut wake: UnixStream,
    event_loop: EventLoopProxy<RuffleEvent>,
) {
    let mut socket: Option<WebSocket<TcpStream>> = None;
    let mut queued: VecDeque<String> = VecDeque::new();
    let mut backoff = Duration::from_secs(1);
    let mut next_try = Instant::now();
    let drop_socket = |socket: &mut Option<WebSocket<TcpStream>>, why: String| {
        if socket.take().is_some() {
            tracing::info!("skua bridge disconnected ({why}); retrying");
        }
    };
    // Skua started this player; if Skua goes away without ending it (killed outright),
    // the player is re-parented and nothing drives it any more: exit.
    // SAFETY: getppid has no preconditions.
    let parent = unsafe { libc::getppid() };
    let would_block = |e: &tungstenite::Error| matches!(e, tungstenite::Error::Io(e) if e.kind() == ErrorKind::WouldBlock);
    loop {
        // Clear the wake-ups; the channel holds the messages.
        let mut buf = [0u8; 64];
        while matches!(wake.read(&mut buf), Ok(n) if n > 0) {}

        if socket.is_none() && unsafe { libc::getppid() } != parent {
            tracing::info!("skua bridge: Skua is gone (the player was re-parented); exiting");
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
                    tracing::info!("skua bridge connected: {url}");
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
                    tracing::debug!("skua bridge: {url} not reachable ({e}); retrying");
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
        let mut fds = vec![libc::pollfd {
            fd: wake.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        }];
        if let Some(ws) = &socket {
            let events = libc::POLLIN | if wants_write { libc::POLLOUT } else { 0 };
            fds.push(libc::pollfd {
                fd: ws.get_ref().as_raw_fd(),
                events,
                revents: 0,
            });
        }
        // SAFETY: fds is a valid, initialised pollfd array for the whole call.
        unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout_ms) };
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
    let (ws, _) = tungstenite::client::client(url, stream).map_err(|e| e.to_string())?;
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
        Some(page) => page_call(page, &args),
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
    send_to_skua(reply.to_string());
}

/// The page's own functions (window.vibeskua in the browser build):
/// pauseDrawing(paused), setRender({fps}) and getRender(). TabThrottle
/// pauses drawing for hidden and headless tabs.
fn page_call(name: &str, args: &[ExternalValue]) -> Result<Json, String> {
    match name {
        "pauseDrawing" => {
            let paused = matches!(args.first(), Some(ExternalValue::Bool(true)));
            DRAWING_PAUSED.store(paused, Ordering::Relaxed);
            Ok(Json::Bool(true))
        }
        "setRender" => {
            if let Some(ExternalValue::Object(options)) = args.first() {
                match options.get("fps") {
                    Some(ExternalValue::Number(fps)) if fps.is_finite() && *fps >= 1.0 => {
                        RENDER_CAP_FPS.store(*fps as u32, Ordering::Relaxed)
                    }
                    Some(ExternalValue::Number(fps)) if *fps <= 0.0 => {
                        DRAWING_PAUSED.store(true, Ordering::Relaxed)
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
        _ => Err(format!("no page function named {name}")),
    }
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
