# Verification — 2026-09-26

## Gemini migration (current configuration)

The default provider is now Gemini. Image understanding, story planning, critic review, and phrase transcription all use the Gemini adapter. There is no OpenAI fallback in Gemini mode. Analysis/transcript caches are separated by provider and model. Google free-tier data terms are disclosed in the creator flow.

The app includes a loopback-only, authenticated Connect Gemini form. A small real generation must succeed before a key is saved. The key is not returned to the browser or placed in an upstream URL. Workers reload AI settings at job startup.

The production frontend build passed. Fourteen of fifteen tests passed in the initial migration suite run; the upload test correctly encountered the host's real low-disk guard. That test was then updated to explicitly verify both the low-space rejection and sufficient-space upload paths, and passed on rerun. New Gemini tests exercise image/audio payloads, schema repair, blocked outputs, sanitized quota errors, timestamp bounds, and private key setup with mocked provider responses. These tests do not establish live Gemini access.

No valid Gemini key is configured yet, so live Gemini generation and the original real-gym acceptance test remain pending. At the last check, only about 52 MB remained free on C:, and real uploads are correctly blocked below the 64 MB guard. Free several GB before uploading a workout.

## Latest credential check

The user supplied a replacement key, saved privately in `.env`. OpenAI authenticated it successfully and returned access to `gpt-4.1`. The API and worker were restarted with the new configuration.

A new isolated live smoke test reached the actual Responses API, which returned HTTP 429 with `credit_balance_exhausted` / `insufficient_quota`: the API account has no credits remaining. Analysis therefore did not complete, and the job was correctly marked FAILED and retryable. A valid key is now configured; adding API credits is the remaining provider blocker. The earlier invalid-key notes below describe the first attempt, not the current configuration.

At this retry, about 82 MB remained free on C:. Real footage still needs substantially more space.

## Executed successfully

- Production React/TypeScript build (`npm run build`).
- Nine integration tests passed in 74.58 seconds before the no-login change.
- Two affected access/upload tests passed in 18.38 seconds after adding local-session bootstrap; the suite now contains ten tests total.
- Real FFmpeg output: 1080×1920 H.264, AAC audio, requested duration, successful full decode.
- Generated-video checks: source bounds, silent-source handling, music mix, burned captions, tracking crop expressions, segmentation coverage, and exact frame/beat timeline timing.
- API checks: corrupt-file rejection, duplicate-job prevention, authenticated byte-range media, ownership denial, deletion, and expired-worker lease recovery.
- Local-session bootstrap succeeds for loopback clients and rejects remote clients. No login screen remains in the frontend.
- Browser inspected: sign-in before its removal, project creation/upload surface, and direct workspace entry after removal. Responsive single-column layout displayed correctly.
- API and worker started with `scripts/start.ps1`; local app available at http://127.0.0.1:8000.

## Live AI attempt — blocked, not passed

The synthetic live pipeline genuinely preprocessed two generated clips and reached the real OpenAI vision endpoint. The inherited key was rejected with HTTP 401. Google also rejected that key. The job was recorded as FAILED with a recoverable credentials message; no semantic analysis or completed video was invented.

A project-local `.env` has been prepared with an empty `OPENAI_API_KEY`. This overrides the invalid inherited key. Enter a valid key and restart the API/worker. Then rerun `python -m scripts.live_smoke` and the 15-workout-clip acceptance test in README.md.

## Remaining acceptance work

- Actual model-generated analysis, global story planning, and critic revision need a successful live-provider run.
- Actual speech transcription and speech/noise behavior need real gym audio.
- Athlete/equipment tracking, duplicate grouping, caption positioning, music alignment, and edit taste need ordinary phone footage.
- The 15-clip → 45-second High Energy acceptance test has not been run; no gym footage was supplied.
- PostgreSQL and S3 adapters are implemented but not provisioned or integration-tested.
- About 160 MB remained on C: at the last disk check. Free several GB before uploading real footage.

## Local environment notes

The disk filled during dependency installation. Redundant downloads created by this task were removed, and the venv reuses existing NumPy/SciPy/OpenCV installations via `include-system-site-packages=true`. These imports and actual media tests passed. The host's pre-existing Python environment has package-version conflicts reported by `pip check`; a fresh isolated install with adequate disk space is recommended for reproducibility. No unrelated user documents were deleted.

FFmpeg 7.1 and FFprobe 9.0.2 are installed in `.tools`; both ran successfully. The failed live-provider fixture and logs remain private under `data/` for debugging. They are not demo content shown in the creator's local workspace.


## Creator reels and reliability update — 2026-09-26

- Scope is general creator reels. UI copy, analysis instructions, story planning and ranking no longer assume gym footage; static scenes are not penalized simply for lacking motion or physical effort. Existing database names remain compatible.
- Cleared disposable pip/npm download caches and compressed local FFmpeg binaries; measured approximately 15 GB available after recovery (free space fluctuates). No original footage was deleted.
- Browser uploads stream once to private storage. Tests cover space preflight, exact uploaded bytes, incomplete-file removal and real MP4 streaming upload with private byte-range playback.
- Render intermediates, transcription WAV files and test media are cleaned up. Original footage, final renders and provenance remain saved.
- Real Gemini 3.8 Flash calls returned repeated HTTP 503 responses. Switched the local/default model to Gemini 2.5 Flash after successful live text, structured connection and structured plan checks with the configured key.
- Gemini requests retry transient failures at bounded 5/10/20/40-second intervals. Connection and overload failures are distinguished; no cross-provider fallback is used.
- Removed array bounds only from the Gemini wire schema to avoid its constrained-decoder state limit. Full Pydantic array limits and source-bound plan checks still apply locally.
- Full suite passed 19 tests before the final schema regression test; the real raw-upload integration test passed separately. Frontend production build passed. Live user-edit status is recorded below when verification completes.

- Final Gemini adapter regression suite: 8 passed, including schema limits retained locally and bounded overload retries. Together with the preceding full suite and raw-upload integration rerun, all changed paths passed.
- The saved Walking.mp4 project completed a real Gemini 2.5 Flash run: three analyzed moments, five-shot planning, music mix, rendering, critic review and one revision. The UI shows “Your video is ready” and “Travel: Exploring Historic Architecture,” correctly treating the content as travel rather than forcing a gym narrative.
- The selected export loaded in the browser with duration 14.066341 seconds, dimensions 1080×1920, readyState 4 and no media error. Download is available in the project. This validates this particular live clip, not general edit taste across all creator topics.

## Interface simplification — 2026-09-27

- Reworked the dashboard, project setup, processing, and finished-video screens around one clear action per state.
- Moved AI key management into the header, shortened labels and consent copy, removed repeated output explanations, and hid re-edit controls, technical details, and feedback under **More options**.
- The finished-video workspace now keeps the playable reel beside the download and subtitle editor on desktop, with a video-first single-column layout on smaller screens.
- The new-project settings are grouped as Length, Pacing, Subtitles, Music, and Create reel. The subtitle controls use a compact style picker during initial setup.
- Keyboard focus remains visible, controls retain accessible labels, reduced-motion preferences disable decorative subtitle animation, and responsive rules cover 800 px and 520 px layouts.
- Browser review confirmed the saved project and custom subtitle state remain intact. Production build passed and the full backend/media suite passed: 24 tests.
