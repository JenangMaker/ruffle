use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ruffle_core::backend::audio::{
    AudioBackend, AudioMixer, DecodeError, RegisterError, SoundHandle, SoundInstanceHandle,
    SoundStreamInfo, SoundTransform, swf,
};
use ruffle_core::impl_audio_mixer_backend;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum CpalError {
    #[error("No audio devices available")]
    NoDevices,

    #[error("Failed to get default output config")]
    DefaultStream(#[from] cpal::DefaultStreamConfigError),

    #[error("Unsupported sample format {0:?}")]
    UnsupportedSampleFormat(SampleFormat),

    #[error("Couldn't play the audio stream")]
    Play(#[from] cpal::PlayStreamError),

    #[error("Failed to construct audio stream")]
    Build(#[from] cpal::BuildStreamError),

    #[error("The output device's format no longer matches the mixer")]
    FormatChanged,
}

pub struct CpalAudioBackend {
    preferred_device_name: Option<String>,
    /// The mixer's output format (channels, sample rate); a new stream must match it.
    format: (u16, u32),
    stream: cpal::Stream,
    mixer: AudioMixer,
    playing: bool,
    // VibeSkua: set by the stream's error handler when the stream has died; tick()
    // then builds a new one. Seen in a container: KasmVNC's PulseAudio went away
    // under the ALSA plugin, and cpal's worker reported EBADFD in a tight loop
    // (~100,000 errors a second) for the rest of the run, with no sound.
    broken: Arc<AtomicBool>,
    retry_at: Option<Instant>,
    retry_delay: Duration,
}

impl CpalAudioBackend {
    pub fn new(preferred_device_name: Option<&str>) -> Result<Self, CpalError> {
        let host = cpal::default_host();
        let device =
            get_suitable_output_device(preferred_device_name, &host).ok_or(CpalError::NoDevices)?;
        let config = device
            .default_output_config()
            .map_err(CpalError::DefaultStream)?;
        let config = cpal::StreamConfig::from(config);
        let format = (config.channels, config.sample_rate.0);
        let mixer = AudioMixer::new(format.0 as u8, format.1);
        let broken = Arc::new(AtomicBool::new(false));
        let stream = build_stream(&device, &mixer, format, broken.clone())?;
        stream.play().map_err(CpalError::Play)?;

        Ok(Self {
            preferred_device_name: preferred_device_name.map(str::to_owned),
            format,
            stream,
            mixer,
            playing: true,
            broken,
            retry_at: None,
            retry_delay: Duration::from_secs(1),
        })
    }

    /// Replace a dead stream. The mixer stays, so sounds already playing carry on;
    /// the new stream must match its channels and rate, else retry later.
    fn rebuild_stream(&mut self) -> Result<(), String> {
        let host = cpal::default_host();
        let device = get_suitable_output_device(self.preferred_device_name.as_deref(), &host)
            .ok_or_else(|| CpalError::NoDevices.to_string())?;
        let broken = Arc::new(AtomicBool::new(false));
        let stream = build_stream(&device, &self.mixer, self.format, broken.clone())
            .map_err(|e| e.to_string())?;
        if self.playing {
            stream.play().map_err(|e| e.to_string())?;
        }
        // Dropping the old stream stops its worker thread.
        self.stream = stream;
        self.broken = broken;
        Ok(())
    }
}

/// An output stream on `device` fed by `mixer`, in the mixer's format.
fn build_stream(
    device: &cpal::Device,
    mixer: &AudioMixer,
    format: (u16, u32),
    broken: Arc<AtomicBool>,
) -> Result<cpal::Stream, CpalError> {
    let supported = device
        .default_output_config()
        .map_err(CpalError::DefaultStream)?;
    let sample_format = supported.sample_format();
    let config = cpal::StreamConfig::from(supported);
    if (config.channels, config.sample_rate.0) != format {
        return Err(CpalError::FormatChanged);
    }

    let mixer = mixer.proxy();
    // A dead stream reports an error on every pass of its worker. A burst of
    // errors (20 in a second) marks it broken for tick() to replace; from then
    // on, each error waits 100 ms so the worker does not spin until it is dropped.
    // Errors are logged once, then as a count every 10 s.
    let mut errors: u64 = 0;
    let mut logged_at: Option<Instant> = None;
    let mut burst = (Instant::now(), 0u32);
    let error_handler = move |err| {
        errors += 1;
        if logged_at.is_none_or(|t| t.elapsed() >= Duration::from_secs(10)) {
            tracing::error!("Audio stream error ({errors} so far): {}", err);
            logged_at = Some(Instant::now());
        }
        if burst.0.elapsed() >= Duration::from_secs(1) {
            burst = (Instant::now(), 0);
        }
        burst.1 += 1;
        if burst.1 >= 20 || broken.load(Ordering::Relaxed) {
            broken.store(true, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(100));
        }
    };

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config,
            move |buffer, _| mixer.mix::<f32>(buffer),
            error_handler,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_output_stream(
            &config,
            move |buffer, _| mixer.mix::<i16>(buffer),
            error_handler,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_output_stream(
            &config,
            move |buffer: &mut [u16], _| {
                // Since I couldn't easily make `mixer` work with `u16` samples,
                // we fill the buffer as if it was `&[i16]`, and then rotate
                // the sample values to make 32768 the equilibrium.
                mixer.mix::<i16>(bytemuck::cast_slice_mut(buffer));
                for s in buffer.iter_mut() {
                    *s = (*s).wrapping_add(32768);
                }
            },
            error_handler,
            None,
        ),
        _ => return Err(CpalError::UnsupportedSampleFormat(sample_format)),
    }?;
    Ok(stream)
}

impl AudioBackend for CpalAudioBackend {
    impl_audio_mixer_backend!(mixer);

    fn play(&mut self) {
        self.playing = true;
        self.stream.play().expect("Error trying to resume CPAL audio stream. This feature may not be supported by your audio device.");
    }

    fn pause(&mut self) {
        self.playing = false;
        self.stream.pause().expect("Error trying to pause CPAL audio stream. This feature may not be supported by your audio device.");
    }

    fn tick(&mut self) {
        if !self.broken.load(Ordering::Relaxed) {
            return;
        }
        let now = Instant::now();
        let retry_at = *self.retry_at.get_or_insert(now);
        if now < retry_at {
            return;
        }
        match self.rebuild_stream() {
            Ok(()) => {
                tracing::warn!("Audio stream failed; replaced it with a new one");
                self.retry_at = None;
                self.retry_delay = Duration::from_secs(1);
            }
            Err(e) => {
                tracing::warn!(
                    "Audio stream failed and a new one could not be made ({e}); retrying in {}s",
                    self.retry_delay.as_secs()
                );
                self.retry_at = Some(now + self.retry_delay);
                self.retry_delay = (self.retry_delay * 2).min(Duration::from_secs(60));
            }
        }
    }
}

fn get_suitable_output_device(
    preferred_device_name: Option<&str>,
    host: &cpal::Host,
) -> Option<cpal::Device> {
    // First let's check for any user preference...
    if let Some(preferred_device_name) = preferred_device_name
        && let Ok(mut devices) = host.output_devices()
        && let Some(device) =
            devices.find(|device| device.name().ok().as_deref() == Some(preferred_device_name))
    {
        return Some(device);
    }

    // Then let's fall back to the device default
    host.default_output_device()
}
