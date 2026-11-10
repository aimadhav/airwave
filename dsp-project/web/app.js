// Audio File Platform - Modern Interactive Dashboard Frontend

let currentSessionId = null;
let activeSourceTab = 'source-upload';
let activeQueryScope = 'current';
let noticeTimeout = null;
let modalTrigger = null;
let chatSessionId = null;

function showNotice(message, isError = false) {
  const notice = document.getElementById('workspace-notice');
  if (!notice) return;
  clearTimeout(noticeTimeout);
  notice.textContent = message;
  notice.classList.toggle('error', isError);
  notice.hidden = false;
  noticeTimeout = setTimeout(() => { notice.hidden = true; }, 8000);
}

function displayTime(seconds) {
  const value = Number.isFinite(Number(seconds)) ? Math.max(0, Number(seconds)) : 0;
  return `${Math.floor(value / 60).toString().padStart(2, '0')}:${(value % 60).toFixed(1).padStart(4, '0')}`;
}

function displaySpeaker(name) {
  const match = /^SPEAKER_(\d+)$/i.exec(name || '');
  return match ? `Speaker ${Number(match[1]) + 1}` : (name || 'Unknown speaker');
}

function renderDSP(data) {
  const dsp = data.dsp || {};
  const note = document.getElementById('session-provenance');
  const container = document.getElementById('audio-timeline');
  note.hidden = false;
  if (dsp.fixture) {
    note.innerHTML = '<strong>Offline integration fixture · not real speaker recognition</strong>Real normalization, energy VAD, timestamping and WAV clipping run on generated signals. Speaker labels are assigned by the mock provider, words are scripted, and insight extraction is deterministic.';
  } else if (dsp.speaker_provider === 'local_model') {
    note.innerHTML = `<strong>Real audio · neural speaker detection · ${dsp.stt_provider === 'local_whisper' ? 'offline Whisper' : escapeHtml(dsp.stt_provider || 'configured STT')}</strong>Speakers are grouped by learned voice embeddings (${dsp.speaker_count_mode === 'provided' ? 'you supplied the count' : 'automatic count'}), not assigned in a rotation. ${dsp.analysis_mode === 'mock' ? 'Words are model transcriptions; insights are offline heuristics, not an LLM.' : 'Words and insights use configured providers.'} Labels are estimates, not verified identities. Overlap clips still contain both voices.`;
  } else if (dsp.analysis_mode === 'mock') {
    note.innerHTML = '<strong>Offline processing · transcription is mocked</strong>DSP runs on your audio, but these words are placeholders, not a transcription. The single-speaker fallback does not distinguish different people. Configure an STT provider for real words.';
  } else if (dsp.speaker_provider === 'single_speaker') {
    note.innerHTML = '<strong>Anonymous single-speaker fallback</strong>Speech regions are detected, but speaker changes are not recognized by this provider. One label does not prove there was only one person.';
  } else {
    note.innerHTML = '<strong>Check your source</strong>Speaker labels and transcript text depend on the providers used for this session. Older offline sessions may contain canned fixture results; run a new demo to see the reviewed pipeline.';
  }
  const segments = dsp.segments || [];
  const duration = Number(dsp.duration_seconds);
  container.hidden = !segments.length || !Number.isFinite(duration) || duration <= 0;
  if (container.hidden) { container.innerHTML = ''; return; }
  const speakers = [...new Set(segments.map(s => s.speaker_id))];
  container.innerHTML = `
    <div class="timeline-header"><h3>DSP clip timeline</h3><span>${(dsp.sample_rate / 1000).toFixed(0)} kHz · ${dsp.channels === 1 ? 'mono' : `${dsp.channels} channels`} · ${segments.length} clips · ${displayTime(duration)}</span></div>
    ${speakers.map((speaker, lane) => `<div class="timeline-lane speaker-color-${lane % 5}"><span class="timeline-lane-label">${escapeHtml(displaySpeaker(speaker))}</span><div class="timeline-track" aria-label="${escapeHtml(displaySpeaker(speaker))} speech intervals">${segments.map((s, index) => s.speaker_id !== speaker ? '' : `<button type="button" class="timeline-segment${s.is_overlap ? ' overlap' : ''}" data-clip-index="${index}" style="left:${Math.max(0, s.start_time / duration * 100)}%;width:${Math.max(0, (s.end_time - s.start_time) / duration * 100)}%" aria-label="Play clip ${index + 1}, ${escapeHtml(displaySpeaker(s.speaker_id))}, ${displayTime(s.start_time)} to ${displayTime(s.end_time)}${s.is_overlap ? ', overlap' : ''}" title="${escapeHtml(displaySpeaker(s.speaker_id))} · ${displayTime(s.start_time)}–${displayTime(s.end_time)}">${index + 1}</button>`).join('')}</div></div>`).join('')}
    <div class="timeline-ticks"><span>00:00.0</span><span>${displayTime(duration / 2)}</span><span>${displayTime(duration)}</span></div>
    <div class="timeline-legend">${speakers.map((s, i) => `<span class="speaker-color-${i % 5}"><i class="legend-dot"></i>${escapeHtml(displaySpeaker(s))}${dsp.fixture ? ' (fixture)' : ''}</span>`).join('')}<span>Hatching = overlap · grey = no selected clip</span></div>
    <p class="timeline-help" id="selected-clip-description">Choose a segment to inspect its original timestamps and listen to the generated WAV clip.</p>
    <audio class="clip-audio" id="clip-player" controls preload="none" aria-label="Selected DSP clip"></audio>`;
  function selectClip(index, autoplay) {
    const s = segments[index];
    const player = document.getElementById('clip-player');
    player.src = `/api/audio/${encodeURIComponent(data.session_id)}/${encodeURIComponent(s.segment_id)}`;
    document.getElementById('selected-clip-description').textContent = `Clip ${index + 1} · ${displaySpeaker(s.speaker_id)} · ${displayTime(s.start_time)}–${displayTime(s.end_time)} · label source: ${dsp.fixture ? 'mock diarization' : s.speaker_source}${s.is_overlap ? ' · overlapping interval' : ''}. ${dsp.fixture ? 'This clip contains test tones, not the scripted words below.' : ''}`;
    if (autoplay) player.play().catch(() => showNotice('Use the audio player to start playback.'));
  }
  container.querySelectorAll('[data-clip-index]').forEach(btn => {
    btn.addEventListener('click', () => selectClip(Number(btn.dataset.clipIndex), true));
  });
  selectClip(0, false);
}

// Upload & Recording State
let uploadedFileData = null;
let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordedAudioBase64 = null;
let recordTimerInterval = null;
let recordSeconds = 0;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initModal();
  initSourceTabs();
  initDropzone();
  initMicRecorder();
  initChatScope();
  loadSystemStatus();
  loadSessions();

  const globalModelSelect = document.getElementById('global-model-select');
  if (globalModelSelect) {
    globalModelSelect.addEventListener('change', async (e) => {
      await setSystemMode(e.target.value);
    });
  }

  const modalModeSelect = document.getElementById('ingest-provider-mode');
  if (modalModeSelect) {
    modalModeSelect.addEventListener('change', async (e) => {
      await setSystemMode(e.target.value);
    });
  }

  const generateBtn = document.getElementById('btn-generate-sample');
  if (generateBtn) generateBtn.addEventListener('click', handleGenerateSample);

  const multiSpeakerDemoBtn = document.getElementById('btn-multispeaker-demo');
  if (multiSpeakerDemoBtn) multiSpeakerDemoBtn.addEventListener('click', handleMultiSpeakerDemo);
  document.getElementById('btn-real-audio-demo')?.addEventListener('click', handleRealAudioDemo);

  const ingestForm = document.getElementById('ingest-form');
  if (ingestForm) ingestForm.addEventListener('submit', handleIngestForm);

  const queryForm = document.getElementById('chat-query-form');
  if (queryForm) queryForm.addEventListener('submit', handleChatQuery);

  const searchInput = document.getElementById('session-search-input');
  if (searchInput) searchInput.addEventListener('input', handleSessionSearch);

  const refreshMemBtn = document.getElementById('btn-refresh-memory');
  if (refreshMemBtn) refreshMemBtn.addEventListener('click', loadMemoryGraph);

  const chipMemory = document.getElementById('stage-chip-memory');
  if (chipMemory) {
    chipMemory.style.cursor = 'pointer';
    chipMemory.addEventListener('click', () => {
      document.getElementById('btn-tab-memory')?.click();
    });
  }
});

// Tab Navigation for Session View (Transcript / Insights / Summary / Ask AI / Memory)
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      tabButtons.forEach(b => b.setAttribute('aria-current', b === btn ? 'page' : 'false'));
      const target = btn.getAttribute('data-tab');
      if (target !== 'tab-transcript') document.getElementById('clip-player')?.pause();
      const pane = document.getElementById(target);
      if (pane) pane.classList.add('active');

      if (target === 'tab-memory') {
        loadMemoryGraph();
      }
    });
  });
}

function initChatScope() {
  const scopeCurrentBtn = document.getElementById('btn-scope-current');
  const scopeWorkspaceBtn = document.getElementById('btn-scope-workspace');
  if (scopeCurrentBtn && scopeWorkspaceBtn) {
    scopeCurrentBtn.addEventListener('click', () => {
      activeQueryScope = 'current';
      scopeCurrentBtn.classList.add('active');
      scopeWorkspaceBtn.classList.remove('active');
    });
    scopeWorkspaceBtn.addEventListener('click', () => {
      activeQueryScope = 'workspace';
      scopeWorkspaceBtn.classList.add('active');
      scopeCurrentBtn.classList.remove('active');
    });
  }

  // Quick suggestion pills
  document.querySelectorAll('.suggestion-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const q = pill.getAttribute('data-query');
      const input = document.getElementById('chat-query-input');
      if (input) {
        input.value = q;
        document.getElementById('chat-query-form')?.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });
  });
}

// Modal handling & Top Nav Actions
function initModal() {
  const modal = document.getElementById('ingest-modal');
  const openBtn = document.getElementById('btn-open-ingest-modal');
  const recordNavBtn = document.getElementById('btn-record-mic');
  const pasteNavBtn = document.getElementById('btn-paste-transcript-modal');
  const closeBtn = document.getElementById('btn-close-modal');
  const cancelBtn = document.getElementById('btn-cancel-ingest');
  function openModal(source) {
    modalTrigger = document.activeElement;
    modal.style.display = 'flex';
    document.querySelector('.app-layout').inert = true;
    document.body.style.overflow = 'hidden';
    switchSourceTab(source);
    closeBtn.focus();
  }

  if (openBtn) {
    openBtn.addEventListener('click', () => {
      openModal('source-upload');
    });
  }

  if (recordNavBtn) {
    recordNavBtn.addEventListener('click', () => {
      openModal('source-mic');
    });
  }

  if (pasteNavBtn) {
    pasteNavBtn.addEventListener('click', () => {
      openModal('source-transcript');
    });
  }

  const sampleTranscriptBtn = document.getElementById('btn-paste-sample-transcript');
  if (sampleTranscriptBtn) {
    sampleTranscriptBtn.addEventListener('click', () => {
      const textarea = document.getElementById('ingest-transcript-text');
      const titleInput = document.getElementById('ingest-session-title');
      if (textarea) {
        textarea.value = `Sarah: Welcome everyone. Let's align on the Q4 release roadmap and critical deliverables.
Alex: I have verified that all core backend migrations are finished and unit tests are passing.
David: That's great news. I will finalize the API security checklist and review all endpoint permissions by Friday 5 PM.
Sarah: Perfect. The team decided to target the official product release for October 15.
Alex: Agreed. I will coordinate with DevOps on the deployment rollback strategy.`;
      }
      if (titleInput && (!titleInput.value || titleInput.value === 'Executive Strategy Sync')) {
        titleInput.value = 'Q4 Roadmap & Release Sync';
      }
    });
  }

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (cancelBtn) cancelBtn.addEventListener('click', closeModal);
  modal.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); closeModal(); return; }
    if (event.key !== 'Tab') return;
    const items = [...modal.querySelectorAll('button, input, select, textarea, [tabindex="0"]')]
      .filter(el => !el.disabled && el.getClientRects().length);
    if (!items.length) return;
    const first = items[0], last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });

  // Close on backdrop click
  window.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });
}

function closeModal() {
  const modal = document.getElementById('ingest-modal');
  if (modal) modal.style.display = 'none';
  document.querySelector('.app-layout').inert = false;
  document.body.style.overflow = '';
  modalTrigger?.focus();
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    stopRecording();
  }
}

// Modal Source Tab Switcher
function initSourceTabs() {
  const tabs = document.querySelectorAll('.source-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const sourceId = tab.getAttribute('data-source');
      switchSourceTab(sourceId);
    });
  });
}

function switchSourceTab(sourceId) {
  activeSourceTab = sourceId;
  document.querySelectorAll('.source-tab').forEach(t => {
    t.classList.toggle('active', t.getAttribute('data-source') === sourceId);
  });
  document.querySelectorAll('.source-pane').forEach(p => {
    p.classList.toggle('active', p.id === sourceId);
  });
}

// Drag-and-drop & File Upload Handling
function initDropzone() {
  const dropzone = document.getElementById('audio-dropzone');
  const fileInput = document.getElementById('file-input');
  const clearBtn = document.getElementById('btn-clear-file');

  if (!dropzone || !fileInput) return;

  // Click dropzone to open file picker
  dropzone.addEventListener('click', (e) => {
    if (e.target !== clearBtn) {
      fileInput.click();
    }
  });

  // Drag over / leave effects
  dropzone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault(); fileInput.click();
    }
  });

  // Drag over / leave effects
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  // Drop event
  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      processSelectedFile(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  });

  // Clear selected file
  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetUploadedFile();
    });
  }
}

function processSelectedFile(file) {
  const maxBytes = 25 * 1024 * 1024; // 25 MB
  if (file.size > maxBytes) {
    alert(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 25 MB.`);
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    uploadedFileData = {
      filename: file.name,
      base64: event.target.result,
      size: file.size,
    };

    // Update UI indicator
    const indicator = document.getElementById('selected-file-indicator');
    const label = document.getElementById('file-name-label');
    if (indicator && label) {
      const sizeStr = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
      label.textContent = `Selected: ${file.name} (${sizeStr})`;
      indicator.style.display = 'flex';
    }

    // Prepopulate session title if default
    const titleInput = document.getElementById('ingest-session-title');
    if (titleInput && (!titleInput.value || titleInput.value === 'Executive Strategy Sync')) {
      titleInput.value = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
  };
  reader.readAsDataURL(file);
}

function resetUploadedFile() {
  uploadedFileData = null;
  const fileInput = document.getElementById('file-input');
  if (fileInput) fileInput.value = '';
  const indicator = document.getElementById('selected-file-indicator');
  if (indicator) indicator.style.display = 'none';
}

// Live Microphone Recording using MediaRecorder API
function initMicRecorder() {
  const startBtn = document.getElementById('btn-start-record');
  const stopBtn = document.getElementById('btn-stop-record');

  if (startBtn) {
    startBtn.addEventListener('click', startRecording);
  }
  if (stopBtn) {
    stopBtn.addEventListener('click', stopRecording);
  }
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Live audio recording is not supported in this browser or requires an HTTPS/localhost context.');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];

    let options = {};
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
      options = { mimeType: 'audio/webm;codecs=opus' };
    } else if (MediaRecorder.isTypeSupported('audio/webm')) {
      options = { mimeType: 'audio/webm' };
    }

    mediaRecorder = new MediaRecorder(mediaStream, options);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = () => {
      // Release hardware tracks
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
      }

      const mimeType = mediaRecorder.mimeType || 'audio/webm';
      const blob = new Blob(recordedChunks, { type: mimeType });

      const reader = new FileReader();
      reader.onloadend = () => {
        recordedAudioBase64 = reader.result;
        const hint = document.getElementById('mic-status-hint');
        if (hint) {
          hint.textContent = `✓ Recording ready (${formatTime(recordSeconds)}). Click 'Process & Ingest Meeting' below.`;
          hint.style.color = '#34d399';
        }
      };
      reader.readAsDataURL(blob);

      // Reset visualizer & timer
      const visualizer = document.getElementById('mic-visualizer');
      if (visualizer) visualizer.classList.remove('recording');
      clearInterval(recordTimerInterval);
    };

    mediaRecorder.start(250);

    // Update UI
    const startBtn = document.getElementById('btn-start-record');
    const stopBtn = document.getElementById('btn-stop-record');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    const visualizer = document.getElementById('mic-visualizer');
    if (visualizer) visualizer.classList.add('recording');

    const timer = document.getElementById('record-timer');
    recordSeconds = 0;
    if (timer) timer.textContent = '00:00';

    const hint = document.getElementById('mic-status-hint');
    if (hint) {
      hint.textContent = 'Recording live... Speak clearly into your microphone.';
      hint.style.color = '#f43f5e';
    }

    recordTimerInterval = setInterval(() => {
      recordSeconds++;
      if (timer) timer.textContent = formatTime(recordSeconds);
    }, 1000);

    // Prepopulate session title
    const titleInput = document.getElementById('ingest-session-title');
    if (titleInput && (!titleInput.value || titleInput.value === 'Executive Strategy Sync')) {
      const now = new Date();
      titleInput.value = `Live Voice Meeting ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
  } catch (err) {
    console.error('Microphone access failed:', err);
    alert(`Could not access microphone: ${err.message || err}`);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }

  const startBtn = document.getElementById('btn-start-record');
  const stopBtn = document.getElementById('btn-stop-record');
  if (startBtn) {
    startBtn.disabled = false;
    startBtn.innerHTML = '<span class="record-dot"></span> Record Again';
  }
  if (stopBtn) {
    stopBtn.disabled = true;
  }
}

function formatTime(totalSeconds) {
  const mins = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
  const secs = (totalSeconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

// Fetch System Status
async function loadSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    updateStatusBadge(data);
    syncModelSelects(data.mode);
  } catch (err) {
    console.error('Failed to load status:', err);
  }
}

function updateStatusBadge(data) {
  const badge = document.getElementById('provider-badge');
  if (badge) {
    badge.textContent = data.mode === 'mock' ? 'Offline · no API calls' : `${(data.mode || 'unknown').toUpperCase()} · external provider`;
  }
  const chatEngineLabel = document.getElementById('chat-active-engine');
  if (chatEngineLabel && data.mode) {
    chatEngineLabel.textContent = data.mode.toUpperCase();
  }
}

function syncModelSelects(mode) {
  if (!mode) return;
  const globalSel = document.getElementById('global-model-select');
  if (globalSel) globalSel.value = mode;

  const modalSel = document.getElementById('ingest-provider-mode');
  if (modalSel) modalSel.value = mode;

  const hint = document.getElementById('provider-mode-hint');
  if (hint) {
    if (mode === 'gemini') {
      hint.textContent = 'Processes audio clips directly using Gemini native multimodal audio + text extraction.';
    } else if (mode === 'openai') {
      hint.textContent = 'Transcribes with Whisper-1 STT and extracts insights with GPT-4o.';
    } else {
      hint.textContent = 'Deterministic offline mock (zero cost, tests complete pipeline logic).';
    }
  }
}

async function setSystemMode(mode) {
  try {
    const res = await fetch('/api/set-mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: mode }),
    });
    const data = await res.json();
    updateStatusBadge(data);
    syncModelSelects(data.mode);
  } catch (err) {
    console.error('Failed to set system mode:', err);
  }
}

let activePollingInterval = null;
let activePollingSessionId = null;

// Fetch Sessions List
async function loadSessions() {
  const container = document.getElementById('sessions-list-container');
  try {
    const res = await fetch('/api/sessions');
    if (!res.ok) throw new Error('Unable to fetch sessions');
    const data = await res.json();
    const countBadge = document.getElementById('sessions-count-badge');
    if (countBadge) countBadge.textContent = data.sessions.length;

    if (data.sessions.length === 0) {
      container.innerHTML = '<div class="empty-state">No sessions found in database. Ingest an audio file to begin.</div>';
      return;
    }

    container.innerHTML = '';
    data.sessions.forEach((s, idx) => {
      const card = document.createElement('button');
      card.type = 'button';
      const isCardActive = currentSessionId ? s.session_id === currentSessionId : idx === 0;
      card.className = `session-card ${isCardActive ? 'active' : ''}`;
      card.setAttribute('data-id', s.session_id);
      card.setAttribute('aria-pressed', String(isCardActive));

      let statusBadgeHtml = '';
      if (s.status === 'processing') {
        statusBadgeHtml = '<span class="session-status-tag processing">Live</span>';
      } else if (s.status === 'failed') {
        statusBadgeHtml = '<span class="session-status-tag failed">Partial</span>';
      }

      card.innerHTML = `
        <div class="session-card-title">${escapeHtml(s.title || s.session_id)} ${statusBadgeHtml}</div>
        <div class="session-card-meta">
          <span>${s.turn_count || 0} turn${s.turn_count === 1 ? '' : 's'}</span> &bull;
          <span>${s.fact_count || 0} fact${s.fact_count === 1 ? '' : 's'}</span> &bull;
          <span>${s.task_count || 0} task${s.task_count === 1 ? '' : 's'}</span>
        </div>
      `;
      card.addEventListener('click', () => {
        document.querySelectorAll('.session-card').forEach(c => { c.classList.remove('active'); c.setAttribute('aria-pressed', 'false'); });
        card.classList.add('active');
        card.setAttribute('aria-pressed', 'true');
        loadSessionDetails(s.session_id);
      });
      container.appendChild(card);
    });

    if (!currentSessionId && data.sessions.length > 0) {
      await loadSessionDetails(data.sessions[0].session_id);
    }
  } catch (err) {
    container.innerHTML = '<div class="empty-state">Error loading sessions.</div>';
    console.error(err);
  }
}

// Render Helper Functions for Individual Sections
function renderTranscript(transcript, isProcessing = false) {
  const container = document.getElementById('transcript-stream-container');
  if (!container) return;
  if (!transcript || transcript.length === 0) {
    if (isProcessing) {
      container.innerHTML = '<div class="loading-shimmer-box">Transcribing dialogue &amp; resolving speaker identities...</div>';
    } else {
      container.innerHTML = '<div class="empty-state">No transcript turns recorded for this session.</div>';
    }
  } else {
    const speakers = [...new Set(transcript.map(t => t.speaker_id || t.speaker_name))];
    container.innerHTML = transcript.map(turn => `
      <div class="turn-card speaker-color-${speakers.indexOf(turn.speaker_id || turn.speaker_name) % 3}">
        <div class="turn-avatar" aria-hidden="true">${/^SPEAKER_/i.test(turn.speaker_name || '') ? speakers.indexOf(turn.speaker_id || turn.speaker_name) + 1 : escapeHtml((turn.speaker_name || '?')[0].toUpperCase())}</div>
        <div class="turn-content">
          <div class="turn-header">
            <span class="turn-speaker-name">${escapeHtml(displaySpeaker(turn.speaker_name))}</span>
            <span class="turn-timestamp">${displayTime(turn.start_ts)} – ${displayTime(turn.end_ts)}</span>
          </div>
          <p class="turn-text">${escapeHtml(turn.text)}</p>
        </div>
      </div>
    `).join('');
  }
}

function renderDecisions(decisions, isProcessing = false) {
  const container = document.getElementById('decisions-cards-container');
  const countBadge = document.getElementById('decisions-count-badge');
  if (countBadge) countBadge.textContent = decisions ? decisions.length : 0;
  if (!container) return;

  if (!decisions || decisions.length === 0) {
    if (isProcessing) {
      container.innerHTML = '<div class="loading-shimmer-box">Extracting consensus &amp; agreed decisions...</div>';
    } else {
      container.innerHTML = '<div class="empty-state">No agreed decisions detected in this conversation.</div>';
    }
  } else {
    container.innerHTML = decisions.map(d => `
      <div class="item-card">
        <div class="item-card-title">${escapeHtml(d.decision)}</div>
        <div class="item-card-meta">Recorded speaker: ${(d.agreed_by || []).map(s => escapeHtml(displaySpeaker(s))).join(', ') || 'Unknown'}</div>
      </div>
    `).join('');
  }
}

function renderTasks(actionItems, isProcessing = false) {
  const container = document.getElementById('tasks-cards-container');
  const countBadge = document.getElementById('tasks-count-badge');
  if (countBadge) countBadge.textContent = actionItems ? actionItems.length : 0;
  if (!container) return;

  if (!actionItems || actionItems.length === 0) {
    if (isProcessing) {
      container.innerHTML = '<div class="loading-shimmer-box">Extracting action items, assignees &amp; deadlines...</div>';
    } else {
      container.innerHTML = '<div class="empty-state">No action items detected in this conversation.</div>';
    }
  } else {
    container.innerHTML = actionItems.map(t => `
      <div class="item-card">
        <div class="item-card-title">${escapeHtml(t.task_description)}</div>
        <div class="item-card-meta">Assignee: <strong>${escapeHtml(displaySpeaker(t.assignee_name))}</strong> ${t.deadline ? `&bull; Due: ${escapeHtml(t.deadline)}` : ''}</div>
      </div>
    `).join('');
  }
}

function renderFacts(facts, isProcessing = false) {
  const container = document.getElementById('facts-cards-container');
  const countBadge = document.getElementById('facts-count-badge');
  if (countBadge) countBadge.textContent = facts ? facts.length : 0;
  if (!container) return;

  if (!facts || facts.length === 0) {
    if (isProcessing) {
      container.innerHTML = '<div class="loading-shimmer-box">Extracting verified factual statements...</div>';
    } else {
      container.innerHTML = '<div class="empty-state">No verified factual statements detected.</div>';
    }
  } else {
    container.innerHTML = facts.map(f => `
      <div class="item-card">
        <div class="item-card-title">${escapeHtml(f.statement)}</div>
        <div class="item-card-meta">Extracted statement${f.source_turn_id ? ` · ${escapeHtml(f.source_turn_id)}` : ''}</div>
      </div>
    `).join('');
  }
}

function renderSummary(summary) {
  const overviewEl = document.getElementById('summary-overview-text');
  const hlContainer = document.getElementById('summary-highlights-list');
  if (overviewEl && summary) {
    overviewEl.textContent = summary.overview || 'Synthesizing overview...';
  }
  if (hlContainer && summary && summary.discussion_highlights) {
    hlContainer.innerHTML = summary.discussion_highlights.map(h => `<li>${escapeHtml(h)}</li>`).join('');
  }
}

// Update the Progress Tracker UI
function updateTrackerUI(sessionData) {
  const tracker = document.getElementById('session-processing-tracker');
  if (!tracker) return;

  const turns = sessionData.transcript || [];
  const tasks = sessionData.action_items || [];
  const decisions = sessionData.decisions || [];
  const facts = sessionData.facts || [];
  const summary = sessionData.summary || {};
  const status = sessionData.status || 'processing';

  let percent = 20;
  if (turns.length > 0) percent += 20;
  if (tasks.length > 0) percent += 15;
  if (decisions.length > 0) percent += 15;
  if (facts.length > 0) percent += 15;
  if (summary.overview && !summary.overview.includes('Synthesizing') && !summary.overview.includes('Session intelligence')) percent += 15;

  if (status === 'completed') percent = 100;
  percent = Math.min(100, percent);

  const fill = document.getElementById('tracker-progress-fill');
  const percentEl = document.getElementById('tracker-percentage');
  const statusText = document.getElementById('tracker-status-text');
  const pulseDot = document.getElementById('tracker-pulse-dot');

  if (fill) fill.style.width = `${percent}%`;
  if (percentEl) percentEl.textContent = `${percent}%`;

  // Update Chips
  setChipState('stage-chip-transcript', 'badge-transcript', turns.length > 0 ? 'completed' : 'active', turns.length > 0 ? `${turns.length} Turns` : 'Processing');
  setChipState('stage-chip-tasks', 'badge-tasks', tasks.length > 0 ? 'completed' : 'active', tasks.length > 0 ? `${tasks.length} Saved` : 'Extracting');
  setChipState('stage-chip-decisions', 'badge-decisions', decisions.length > 0 ? 'completed' : 'active', decisions.length > 0 ? `${decisions.length} Saved` : 'Extracting');
  setChipState('stage-chip-facts', 'badge-facts', facts.length > 0 ? 'completed' : 'active', facts.length > 0 ? `${facts.length} Saved` : 'Extracting');

  const summaryReady = summary.overview && !summary.overview.includes('Synthesizing');
  setChipState('stage-chip-summary', 'badge-summary', summaryReady ? 'completed' : 'pending', summaryReady ? 'Ready' : 'Waiting');
  setChipState('stage-chip-memory', 'badge-memory', status === 'completed' ? 'completed' : 'pending', status === 'completed' ? 'Synced' : 'Pending');

  if (status === 'completed') {
    if (statusText) statusText.textContent = '✓ Analysis Complete — All Intelligence & Memory Saved';
    if (pulseDot) {
      pulseDot.className = 'live-pulse-dot completed';
    }
  } else if (status === 'failed') {
    if (statusText) statusText.textContent = `⚠ Analysis Warning: ${sessionData.progress?.error || 'Partial insights preserved'}`;
    if (pulseDot) {
      pulseDot.className = 'live-pulse-dot error';
    }
  } else {
    if (statusText) statusText.textContent = 'Live Analysis Running — Saving Insights Progressively...';
    if (pulseDot) {
      pulseDot.className = 'live-pulse-dot';
    }
  }
}

function setChipState(chipId, badgeId, state, text) {
  const chip = document.getElementById(chipId);
  const badge = document.getElementById(badgeId);
  if (chip) {
    chip.className = `stage-chip ${state}`;
  }
  if (badge) {
    badge.textContent = text;
  }
}

function startLivePolling(sessionId) {
  stopLivePolling();
  activePollingSessionId = sessionId;

  activePollingInterval = setInterval(async () => {
    if (currentSessionId !== activePollingSessionId) {
      stopLivePolling();
      return;
    }

    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
      if (!res.ok) return;
      const data = await res.json();
      if (currentSessionId !== sessionId || activePollingSessionId !== sessionId) return;

      // Live update sections as they arrive
      if (data.dsp?.segments?.length && !document.getElementById('clip-player')) renderDSP(data);
      const isProc = data.status === 'processing';
      renderTranscript(data.transcript, isProc);
      renderTasks(data.action_items, isProc);
      renderDecisions(data.decisions, isProc);
      renderFacts(data.facts, isProc);
      if (data.summary) renderSummary(data.summary);

      // Live update header turns
      const turnCountEl = document.getElementById('hero-turn-count');
      if (turnCountEl && data.transcript) {
        turnCountEl.textContent = `${data.transcript.length} Turns`;
      }

      updateTrackerUI(data);

      if (data.status === 'completed' || data.status === 'failed') {
        stopLivePolling();
        await loadSessions();
        if (currentSessionId === sessionId) await loadSessionDetails(sessionId);
      }
    } catch (err) {
      console.warn('Polling error:', err);
    }
  }, 1000);
}

function stopLivePolling() {
  if (activePollingInterval) {
    clearInterval(activePollingInterval);
    activePollingInterval = null;
  }
  activePollingSessionId = null;
}

// Load Full Session Details
async function loadSessionDetails(sessionId) {
  currentSessionId = sessionId;
  stopLivePolling();
  const previousPlayer = document.getElementById('clip-player');
  if (previousPlayer) previousPlayer.pause();
  try {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    const data = await res.json();
    if (currentSessionId !== sessionId) return;
    if (!res.ok) throw new Error(data.error || 'Unable to load session');
    if (chatSessionId !== data.session_id) {
      chatSessionId = data.session_id;
      document.getElementById('chat-history-container').innerHTML = '<div class="chat-message bot"><div class="chat-avatar">AI</div><div class="chat-bubble">Ask about this conversation’s stored evidence. Offline fixture answers are deterministic; real inference requires a configured provider.</div></div>';
    }

    // Update Hero Banner
    document.getElementById('hero-session-id').textContent = data.session_id;
    document.getElementById('hero-session-title').textContent = data.title || data.session_id;
    const recorded = new Date(data.recorded_at);
    document.getElementById('hero-recorded-date').textContent = Number.isNaN(recorded.getTime()) ? 'Date unavailable' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(recorded);
    document.getElementById('hero-turn-count').textContent = `${(data.transcript || []).length} Turns`;
    const speakers = [...new Set((data.transcript || []).map(t => t.speaker_id || t.speaker_name))];
    document.getElementById('hero-participants').textContent = `${speakers.length} speaker label${speakers.length === 1 ? '' : 's'}`;
    renderDSP(data);

    // Render individual sections
    const isProc = data.status === 'processing';
    renderTranscript(data.transcript, isProc);
    renderDecisions(data.decisions, isProc);
    renderTasks(data.action_items, isProc);
    renderFacts(data.facts, isProc);
    renderSummary(data.summary);

    // Tracker visibility based on session state
    const tracker = document.getElementById('session-processing-tracker');
    if (data.status === 'processing') {
      if (tracker) tracker.style.display = 'block';
      updateTrackerUI(data);
      if (!activePollingSessionId || activePollingSessionId !== sessionId) {
        startLivePolling(sessionId);
      }
    } else {
      if (tracker) tracker.style.display = data.status === 'failed' ? 'block' : 'none';
      if (data.status === 'failed') updateTrackerUI(data);
      stopLivePolling();
    }
  } catch (err) {
    console.error('Error loading session details:', err);
    if (currentSessionId === sessionId) showNotice(err.message || 'Unable to load session', true);
  }
}

// Generate Sample Audio
async function handleGenerateSample() {
  const btn = document.getElementById('btn-generate-sample');
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = 'Generating Audio...';

  try {
    const res = await fetch('/api/generate-sample', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Sample generation failed');
    showNotice(`Sample generated: ${data.file_path} (${(data.size_bytes / 1024).toFixed(1)} KB). Use Upload audio or the server-path input to process it.`);
  } catch (err) {
    showNotice('Failed to generate sample audio.', true);
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalText;
  }
}

async function handleMultiSpeakerDemo() {
  const btn = document.getElementById('btn-multispeaker-demo');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Running DSP demo…';
  }
  try {
    const res = await fetch('/api/demo-multispeaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'mock' }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Multi-speaker demo failed');
    currentSessionId = data.session_id;
    await loadSessions();
    await loadSessionDetails(data.session_id);
    document.getElementById('btn-tab-transcript').click();
    await loadSystemStatus();
    showNotice(`Offline fixture complete: ${data.dsp_segments || 0} WAV clips · ${data.speaker_ids?.length || 0} mock speaker labels. Select a timeline segment to listen.`);
  } catch (err) {
    console.error('Multi-speaker demo failed:', err);
    showNotice(`Demo failed: ${err.message || err}`, true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="8" cy="8" r="3"/><circle cx="17" cy="8" r="3"/><path d="M2.5 20c.6-3.2 2.5-5 5.5-5s4.9 1.8 5.5 5M12.5 20c.5-2.5 2-4 4.5-4 2.4 0 3.9 1.5 4.5 4"/></svg> Synthetic fixture';
    }
  }
}

async function handleRealAudioDemo() {
  const btn = document.getElementById('btn-real-audio-demo');
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = 'Starting real conversation…';
  try {
    const response = await fetch('/api/demo-real-audio', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Real demo failed');
    currentSessionId = result.session_id;
    await loadSessions();
    await loadSessionDetails(result.session_id);
    document.getElementById('btn-tab-transcript').click();
    showNotice('Processing real speech locally: automatic speaker detection → WAV clips → Whisper → offline analysis. This can take a minute.');
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}

// Handle Multi-Source Ingest Form Submission
async function handleIngestForm(e) {
  e.preventDefault();

  const title = document.getElementById('ingest-session-title').value.trim();
  const mode = document.getElementById('ingest-provider-mode').value;

  let endpoint = '';
  let payload = {};

  if (activeSourceTab === 'source-upload') {
    if (!uploadedFileData || !uploadedFileData.base64) {
      alert('Please select or drop an audio file first.');
      return;
    }
    endpoint = '/api/upload';
    payload = {
      audio_base64: uploadedFileData.base64,
      filename: uploadedFileData.filename,
      title: title,
      mode: mode,
    };
  } else if (activeSourceTab === 'source-mic') {
    if (!recordedAudioBase64) {
      alert('Please record a voice clip with your microphone before submitting.');
      return;
    }
    endpoint = '/api/upload';
    payload = {
      audio_base64: recordedAudioBase64,
      filename: `mic_recording_${Date.now()}.webm`,
      title: title,
      mode: mode,
    };
  } else if (activeSourceTab === 'source-url') {
    const url = document.getElementById('ingest-url-input').value.trim();
    if (!url) {
      alert('Please enter a valid audio URL.');
      return;
    }
    endpoint = '/api/fetch-audio';
    payload = {
      url: url,
      title: title,
      mode: mode,
    };
  } else if (activeSourceTab === 'source-path') {
    const filePath = document.getElementById('ingest-file-path').value.trim();
    if (!filePath) {
      alert('Please enter an audio file path on the server.');
      return;
    }
    endpoint = '/api/ingest';
    payload = {
      file_path: filePath,
      title: title,
      mode: mode,
    };
  } else if (activeSourceTab === 'source-transcript') {
    const text = document.getElementById('ingest-transcript-text').value.trim();
    if (!text) {
      alert('Please paste or enter your conversation transcript first.');
      return;
    }
    endpoint = '/api/analyze-transcript';
    payload = {
      transcript_text: text,
      title: title,
      mode: mode,
    };
  }

  if (endpoint !== '/api/analyze-transcript') {
    payload.dsp_provider = document.getElementById('ingest-dsp-provider').value;
    const speakerCount = document.getElementById('ingest-speaker-count').value;
    payload.num_speakers = speakerCount ? Number(speakerCount) : -1;
  }

  // =========================================================================
  // KEY REQUIREMENT 2: IMMEDIATELY CLOSE THE DIALOG BOX!
  // Do not keep the user on the modal dialog. Navigate them to the session page
  // and display live section-by-section progress.
  // =========================================================================
  closeModal();
  resetUploadedFile();
  recordedAudioBase64 = null;

  // Immediately prepare the session dashboard view
  const provisionalTitle = title || 'Processing Meeting';
  document.getElementById('hero-session-id').textContent = 'INITIALIZING SESSION...';
  document.getElementById('hero-session-title').textContent = provisionalTitle;
  document.getElementById('hero-recorded-date').textContent = 'Recorded: Just now';
  document.getElementById('hero-turn-count').textContent = 'Analyzing...';
  document.getElementById('hero-participants').textContent = `AI Engine: ${mode.toUpperCase()}`;

  // Show shimmer placeholders in all sections
  renderTranscript([], true);
  renderTasks([], true);
  renderDecisions([], true);
  renderFacts([], true);
  renderSummary({
    overview: 'Initializing session intelligence pipeline and parsing transcript...',
    discussion_highlights: ['Analyzing speaker dialogue...'],
  });

  // Reveal the Progress Tracker Card
  const tracker = document.getElementById('session-processing-tracker');
  if (tracker) {
    tracker.style.display = 'block';
    const fill = document.getElementById('tracker-progress-fill');
    const percentEl = document.getElementById('tracker-percentage');
    const statusText = document.getElementById('tracker-status-text');
    const pulseDot = document.getElementById('tracker-pulse-dot');
    if (fill) fill.style.width = '15%';
    if (percentEl) percentEl.textContent = '15%';
    if (statusText) statusText.textContent = 'Session Initialized — Starting Parallel Extractors...';
    if (pulseDot) pulseDot.className = 'live-pulse-dot';

    setChipState('stage-chip-transcript', 'badge-transcript', 'active', 'Running');
    setChipState('stage-chip-tasks', 'badge-tasks', 'active', 'Running');
    setChipState('stage-chip-decisions', 'badge-decisions', 'active', 'Running');
    setChipState('stage-chip-facts', 'badge-facts', 'active', 'Running');
    setChipState('stage-chip-summary', 'badge-summary', 'pending', 'Waiting');
    setChipState('stage-chip-memory', 'badge-memory', 'pending', 'Waiting');
  }

  // Ensure user is on the first tab
  const transcriptTabBtn = document.getElementById('btn-tab-transcript');
  if (transcriptTabBtn) transcriptTabBtn.click();

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      if (tracker) {
        document.getElementById('tracker-status-text').textContent = `⚠ Ingestion failed: ${data.error}`;
        document.getElementById('tracker-pulse-dot').className = 'live-pulse-dot error';
      }
      alert(`Ingestion failed: ${data.error}`);
    } else if (data.session_id) {
      currentSessionId = data.session_id;
      document.getElementById('hero-session-id').textContent = data.session_id;
      document.getElementById('hero-session-title').textContent = data.title || provisionalTitle;
      if (data.turn_count !== undefined) {
        document.getElementById('hero-turn-count').textContent = `${data.turn_count} Turns`;
      }

      // Start live section polling
      startLivePolling(data.session_id);
      await loadSessions();
      await loadSystemStatus();
    }
  } catch (err) {
    console.error('Request dispatch error:', err);
    if (tracker) {
      document.getElementById('tracker-status-text').textContent = `⚠ Connection error: ${err.message}`;
      document.getElementById('tracker-pulse-dot').className = 'live-pulse-dot error';
    }
  }
}

// Chat / Ask AI Query
async function handleChatQuery(e) {
  e.preventDefault();
  const input = document.getElementById('chat-query-input');
  const query = input.value.trim();
  if (!query) return;

  const history = document.getElementById('chat-history-container');

  // Render user question
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-message user';
  userMsg.innerHTML = `
    <div class="chat-avatar">ME</div>
    <div class="chat-bubble"><p>${escapeHtml(query)}</p></div>
  `;
  history.appendChild(userMsg);
  input.value = '';
  history.scrollTop = history.scrollHeight;

  // Bot loading placeholder
  const botMsg = document.createElement('div');
  botMsg.className = 'chat-message bot';
  botMsg.innerHTML = `
    <div class="chat-avatar">AI</div>
    <div class="chat-bubble"><p>Searching conversation records...</p></div>
  `;
  history.appendChild(botMsg);
  history.scrollTop = history.scrollHeight;

  try {
    const activeMode = document.getElementById('global-model-select')?.value || 'gemini';
    const payload = {
      query: query,
      session_id: activeQueryScope === 'workspace' ? null : currentSessionId,
      scope: activeQueryScope,
      mode: activeMode,
    };
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    let citationsHtml = '';
    if (data.citations && data.citations.length > 0) {
      citationsHtml = data.citations.map(c => {
        const sessionBadge = c.session_title ? `<span class="citation-session-badge">${escapeHtml(c.session_title)}</span>` : '';
        return `
          <div class="citation-chip">
            <div class="citation-header">
              <span><strong>${escapeHtml(c.speaker_name)}</strong> [${escapeHtml(c.timestamp_str)}]</span>
              ${sessionBadge}
            </div>
            <div class="citation-quote">"${escapeHtml(c.verbatim_quote)}"</div>
          </div>
        `;
      }).join('');
    }

    botMsg.querySelector('.chat-bubble').innerHTML = `
      <p>${escapeHtml(data.answer_text)}</p>
      ${citationsHtml}
    `;
    history.scrollTop = history.scrollHeight;
  } catch (err) {
    botMsg.querySelector('.chat-bubble').innerHTML = '<p>Error retrieving answer from memory.</p>';
    console.error(err);
  }
}

// Search sessions in sidebar
function handleSessionSearch(e) {
  const filter = e.target.value.toLowerCase();
  document.querySelectorAll('.session-card').forEach(card => {
    const title = card.querySelector('.session-card-title').textContent.toLowerCase();
    card.style.display = title.includes(filter) ? 'block' : 'none';
  });
}

// Load and Render Knowledge Memory Graph & Evolution
async function loadMemoryGraph() {
  const recordsContainer = document.getElementById('memory-records-container');
  const logContainer = document.getElementById('memory-log-container');
  if (recordsContainer) recordsContainer.innerHTML = '<div class="loading-shimmer-box">Loading knowledge graph entities...</div>';
  if (logContainer) logContainer.innerHTML = '<div class="loading-shimmer-box">Loading evolution audit log...</div>';

  try {
    const res = await fetch('/api/memory-graph');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const records = data.records || [];
    const logs = data.change_log || [];
    const rels = data.relationships || [];

    const activeCount = records.filter(r => r.status === 'active').length;
    const supersededCount = records.filter(r => r.status === 'superseded' || r.status === 'flagged').length;

    const elTotal = document.getElementById('mem-stat-total-records');
    const elActive = document.getElementById('mem-stat-active-records');
    const elSuper = document.getElementById('mem-stat-superseded');
    const elRels = document.getElementById('mem-stat-relationships');

    if (elTotal) elTotal.textContent = records.length;
    if (elActive) elActive.textContent = activeCount;
    if (elSuper) elSuper.textContent = supersededCount;
    if (elRels) elRels.textContent = rels.length;

    // Render Records
    if (recordsContainer) {
      if (records.length === 0) {
        recordsContainer.innerHTML = '<div class="empty-state">No knowledge records stored in memory yet. Ingest transcripts to build graph memory.</div>';
      } else {
        recordsContainer.innerHTML = records.map(r => {
          const typeClass = r.record_type || 'fact';
          const statusClass = r.status || 'active';
          const edgeInfo = (r.related_record_ids && r.related_record_ids.length > 0)
            ? `<div class="memory-record-edge-chip">🔗 Connected to: ${escapeHtml(r.related_record_ids.join(', '))}</div>`
            : '';
          return `
            <div class="memory-record-card">
              <div class="memory-record-header">
                <span class="memory-record-type ${typeClass}">${escapeHtml(r.record_type)}</span>
                <span class="memory-record-status ${statusClass}">${escapeHtml(r.status)}</span>
              </div>
              <div class="memory-record-statement">${escapeHtml(r.statement)}</div>
              <div class="item-card-meta">Origin: <code>${escapeHtml(r.session_id)}</code></div>
              ${edgeInfo}
            </div>
          `;
        }).join('');
      }
    }

    // Render Change Log
    if (logContainer) {
      if (logs.length === 0) {
        logContainer.innerHTML = '<div class="empty-state">No evolution changes recorded yet.</div>';
      } else {
        logContainer.innerHTML = logs.slice().reverse().map(l => {
          const actionClass = l.action || 'novel';
          return `
            <div class="memory-log-card">
              <div style="margin-bottom: 4px;">
                <span class="memory-log-action ${actionClass}">${escapeHtml(l.action)}</span>
                <strong>Record:</strong> <code>${escapeHtml(l.record_id)}</code> &bull; 
                <span>${escapeHtml(l.session_id)}</span>
              </div>
              <div class="memory-log-detail">${escapeHtml(l.details || 'Knowledge updated')}</div>
            </div>
          `;
        }).join('');
      }
    }
  } catch (err) {
    console.error('Failed to load memory graph:', err);
    if (recordsContainer) recordsContainer.innerHTML = '<div class="empty-state">Error loading knowledge records.</div>';
    if (logContainer) logContainer.innerHTML = '<div class="empty-state">Error loading change log.</div>';
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
