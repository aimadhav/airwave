YOUR PRIVATE LISTENING ROOM

This folder contains the complete local page for your recording.
Keep index.html, both viewer/upload CSS and JavaScript files, and the audio file together.

Open index.html in a desktop browser, or serve this folder locally.
From the project root, the preview command is:

python scripts/serve_recording_viewer.py --directory data/listening-room/sample-podcast --port 5095

Then visit http://127.0.0.1:5095/

NEW RECORDINGS
Click "New recording" in the local HTTP app to choose and process another file.
The saved HTML version offers a link to that app; it cannot run models itself.
Uploads are limited to 128 MB and 30 minutes by default and processed one at a time.
Each recording gets a separate folder; previous results are not replaced.

Features: original audio playback, timestamp seeking, search, speaker filters,
overlap-only view, simultaneous active-speaker highlighting, timeline-model
comparison, playback speed and text export.

IMPORTANT LIMITATIONS
The speaker labels are estimated acoustic groups, not verified people.
The two diarizers disagree about the number of groups in this recording.
No source separation was performed. When a second overlapping voice has no
recovered words, the page explicitly shows that its words remain unresolved.
Machine transcription may contain mistakes; check the original audio.
The page does not contact a transcription service or load external fonts.

The bundle contains a copy of your audio and transcript. Keep it private unless
you intentionally choose to share it. Processing diagnostics are not bundled.
