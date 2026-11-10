# Airwave demo

Airwave is a local-first hackathon prototype for context-aware podcast ads. It uses your local diarization project to find timestamped words and speaker turns, runs a bounded two-round debate between two models, then uses Gnani TTS to create and insert a sponsored continuation.

**[Read how Airwave works](HOW_IT_WORKS.md)** for the complete flow, including how the bundled [`dsp-project/`](dsp-project/) provides local speech recognition and speaker diarization.

## Run locally

Requirements: Node.js 18 or newer. There are no npm dependencies.

1. If `.env` does not exist yet, copy `.env.example` to `.env`.
2. Set up and start the included local STT/DSP project. From `dsp-project`, create its Python runtime, install its requirements, download the local speech assets, then run `Start-Listening-Room.cmd`. See [`dsp-project/README.md`](dsp-project/README.md). It serves locally on `127.0.0.1:5095`.
3. Add `GNANI_API_KEY` to enable Gnani TTS and optional voice cloning. The demo does not use Gnani STT.
4. Add a Groq API key in **Groq API settings** in the app. The two model IDs are editable there; defaults are `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`. A key entered in the UI stays in local server memory until restart. You can also set `GROQ_API_KEY` in `.env`.
5. Run `node server.mjs` from this directory and open `http://localhost:4175`.
6. Install FFmpeg and set `FFMPEG_PATH` in `.env` to the full path of `ffmpeg.exe` (or make `ffmpeg` available on the server's `PATH`). Airwave uses FFmpeg to crop the selected search area without decoding an entire podcast into browser memory, and to stream the ad insertion.

The DSP project source, setup scripts, tests, and reusable documentation live in `dsp-project/`. Model weights, virtual environments, cached recordings, and local processing outputs stay out of Git and are downloaded or created on the machine when needed.

Without API keys, the app remains usable in demo mode: its two labeled perspectives use a local relevance heuristic and sample scripts. The sample episode audio is a locally generated audio bed and is only intended to preview placement. Upload a real episode for a meaningful listening check.

## What works

- Upload an audio file, view its waveform, and adjust a broad search area.
- The browser reads audio metadata without decoding the whole podcast into memory. Airwave uploads the compressed file to its local server, uses FFmpeg to crop only the selected search range, then sends that crop to the local STT project (up to the service's 128 MB / 30-minute limits). Returned turns are restored to episode timestamps. Choose the host speaker once per new episode; the selection is cached with its transcript.
- Timestamped turns for the same audio and exact search range are cached in browser IndexedDB and in `.airwave-cache/transcripts` for 30 days. The local server cache is shared across Airwave ports; it stores transcript text and the selected host speaker. Temporary episode audio used for local silence analysis and previews is kept in the server temp directory for up to one hour, then removed.
- LLM request payloads, each actual provider prompt and response, and model retries for ad matching are saved in the browser's IndexedDB. Open **LLM request history** in the left sidebar to inspect or clear them. A local ranker sends Groq at most five diverse candidate moments with their nearby turns, not the full transcript. Local silence evidence stays on the server and is summarized as candidate features.
- Edit listener interests and locale. Add advertiser campaigns with product facts, audience, offer, and prohibited claims; advertiser data and preferences persist in browser local storage.
- Ask two configured text models to independently select an eligible completed host turn and advertiser, then critique both proposals. In round two, each model gets the competing draft and critique and can revise its advertiser, script, or anchor. The server validates all four proposals, resolves each turn ID to its precise end timestamp, and selects the strongest safe candidate by combined relevance, safety, naturalness, and slot-fit scores. The debate is capped at two rounds.
- Each model receives an explicit list of eligible host-turn IDs. For uploaded episodes, failed live review is shown as an error rather than replaced with simulated demo candidates.
- Local candidate ranking combines terminal wording, likely thought completion, question/answer risk, same-speaker continuation, turn timing, and FFmpeg silence detection when the source is available. Pause length is a ranking signal rather than a fixed eligibility cutoff. Groq sees at most five time-separated moments and their adjacent turn context.
- Each model proposal has a **Preview this placement** control. Gnani creates that candidate's sponsor read, and FFmpeg plays a short audio excerpt before the exact insertion, the ad, then the original episode after it. The episode audio is uploaded once per scan and reused across previews; it stays in the local server temp directory for up to one hour.
- After listening, **Start full episode with this ad** plays the continuous MP3 stream: original audio up to the selected host turn's exact end, the clearly labeled ad, then the remaining original audio. No replacement episode download is needed. The full stream is cleaned up when playback ends or the session expires.
- The stream endpoint checks the chosen anchor against the uploaded episode's transcript, selected host speaker, and exact clean turn end before accepting it.
- Optionally provide 5–30 seconds of your own clean speech to Gnani voice cloning. The embedding is held in server memory and can be forgotten; use requires a consent checkbox. The generated segment explicitly identifies itself as a sponsor read.

## API notes

Gnani keys are only read by `server.mjs`. TTS is sent to `https://api.vachana.ai/api/v1/tts/inference`; optional voice embeddings use `https://api.vachana.ai/api/v1/tts/voice-clone/embeddings`. Episode audio is sent only to local services: the local STT service at `127.0.0.1:5095` for transcription and the Airwave server for local boundary analysis and preview/stream assembly. Groq receives at most five timestamped ranked host-turn moments, each with nearby transcript context, listener interests, and advertiser facts—not the full transcript or audio. Groq and Gnani calls are server-side.

Review the transcript and preview the transition: speaker IDs are estimated groups, not verified identities, so choose the correct host voice yourself. Word timing and diarization can be wrong, particularly in overlap. The cloned sponsor voice is an AI-generated read; it does not represent a real spontaneous endorsement. This prototype uses local browser storage and files held in the current browser session. It does not include accounts, a hosted database, cloud episode storage, or production ad delivery. Only clone a voice you have permission to use.
