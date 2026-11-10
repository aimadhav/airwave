# Airwave

Airwave is a local-first podcast advertising demo. It finds a completed host turn, chooses a relevant advertiser, generates a clearly identified sponsor read, and plays the episode with the ad inserted into one continuous stream.

## How it works

1. Load a podcast file and choose a broad search range.
2. Airwave uses FFmpeg to crop that range and sends the crop to the local speech service in [`dsp/`](dsp/).
3. The DSP service returns timestamped words and speaker turns. Airwave ranks likely turn endings locally and sends Groq no more than five candidate moments with nearby transcript context—not the full episode or full transcript.
4. Two Groq models propose and review advertiser, script, and eligible host-turn choices. The server validates each choice against the transcript and advertiser catalog.
5. Preview a candidate. Gnani TTS creates the sponsor read; then Airwave can stream the original episode prefix, the ad, and the original remainder in sequence.

The full request and playback flow is in [How Airwave works](HOW_IT_WORKS.md).

## Run locally on Windows

Requirements: Node.js 18+, Python 3.11, and FFmpeg.

1. Copy `.env.example` to `.env`. Add your `GNANI_API_KEY`; set `FFMPEG_PATH` if `ffmpeg.exe` is not on `PATH`. A Groq key can go in `.env` as `GROQ_API_KEY` or in **Groq API settings** in the Airwave UI.
2. Follow the setup steps in [`dsp/README.md`](dsp/README.md), then start `dsp/Start-Listening-Room.cmd` in a separate terminal. The local speech API listens on `127.0.0.1:5095`.
3. From this repository's root, run:

   ```powershell
   node server.mjs
   ```

4. Open [http://localhost:4175](http://localhost:4175), choose **Try it**, and load an episode.

## Repository contents

- The repository root contains the Airwave website, dashboard, and Node server.
- [`dsp/`](dsp/) contains the local DSP/transcription code and setup files Airwave uses. Benchmarks, unrelated apps, research docs, tests, and generated package metadata are excluded.
- Speech models, virtual environments, recordings, API keys, and local caches are downloaded or created on the machine and excluded from Git.

This demo does not connect to Spotify or modify a Spotify playback session. You load an episode in Airwave and start its continuous stream after reviewing the proposed placement. See [How Airwave works](HOW_IT_WORKS.md) for data handling and limitations.
