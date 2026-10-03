# Autonomous Editor — creator reels MVP

Private raw footage → sampled multimodal analysis → validated story → persisted EDL → real FFmpeg MP4 → visual critic. There is no demo-video fallback and no simulated analysis mode in the application.

## Run locally on Windows

Requirements: Python 3.11+, Node 20+, a Gemini API key from Google AI Studio, and **at least 5 GB free disk space for setup; substantially more for original footage and renders**.

```powershell
.\scripts\setup.ps1
# Copy .env.example to .env if needed. Connect Gemini in the app after startup.
.\scripts\start.ps1
```

Open http://127.0.0.1:8000. The app opens directly into a single local workspace with **no login page**, as requested. Upload up to 30 MP4/MOV/WebM clips (2 GB per file, 60 minutes per project by default). Choose length/style, optionally upload music, consent to AI processing, and create. Keep the API and worker running. Processing survives closing the browser. Retry failed jobs after correcting the cause; completed segment analysis and transcripts are reused.

Use the app's **Connect Gemini** form to paste your key. It verifies a real small generation before saving the key privately in `.env`. Keys in that file take precedence over inherited credentials. The default is `AI_PROVIDER=gemini`, `GEMINI_MODEL=gemini-2.5-flash`. The worker reads saved AI settings at the beginning of each job. OpenAI remains optional via `AI_PROVIDER=openai`; Gemini mode never falls back to OpenAI or requires OpenAI credits.

Gemini's free tier is quota-limited, and Google may use free-tier inputs and outputs to improve its products. Use non-sensitive test footage. The app discloses these terms in both setup and edit consent. A failed key or quota check does not overwrite the previous configuration.

Run `scripts/start.ps1` again to restart the API and worker, or `scripts/stop.ps1` to stop them. Restart only when no video is processing.

Development: run `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`, `python -m backend.worker` in a second shell, and `npm run dev` in `frontend`. Use the virtual environment's Python. Frontend Vite proxies `/api` to FastAPI.

## Architecture

- **React / TypeScript / Vite** creator interface, served by FastAPI in the built application. No SSR is required for this private editor.
- **FastAPI + SQLAlchemy** with SQLite/WAL locally; set `DATABASE_URL=postgresql+psycopg://...` for PostgreSQL. Dedicated tables cover users, assets, segments, analyses, events, transcripts, plans, EDLs, jobs, versions, critics, workouts, and feedback.
- **Separate worker and durable SQL queue**, using an atomic claim and heartbeat lease. A Redis service is unnecessary for this single-machine MVP. Expired jobs become retryable failures; parallel workers claim jobs atomically.
- **FFprobe** metadata; FFmpeg low-resolution proxies; OpenCV scans every proxy frame for visual changes, motion transitions, quality measurements, and six-second maximum segment windows. Three frames per segment reach the vision model.
- **Gemini structured outputs** through Pydantic schemas and the Generate Content REST API. Gemini handles sampled images, planning, rendered-frame review, and audio. Keys travel in headers, never URLs. Bounded retries repair invalid JSON; blocked/truncated responses are rejected. Optional OpenAI mode retains its structured-output adapter with `store=False`.
- **Interpretable highlight scoring**, HSV histogram descriptors plus semantic activity labels for approximate similarity, and global content/story reasoning. Similar angles are not assumed synchronized; MVP chooses a single angle per event.
- **Source-bound plan validation** rejects nonexistent IDs, out-of-range times, repeated source intervals, wrong duration, and unchanged alternative sequences. At most three planning attempts.
- **Reframing** uses model-proposed subject/object bounds and KLT feature tracking. Uncertain tracking or subjects too wide for a safe crop use a complete image over a blurred background. This preserves the subject but is not a production pose tracker.
- **Audio** uses Gemini phrase transcription with estimated timestamps and confidence filtering, short captions, RMS onset music analysis, bounded cut snapping, speech-aware music ducking, and loudness normalization. Source-bound timestamp validation rejects invalid or overlapping speech ranges. Optional OpenAI mode retains Whisper. Audio classification is approximate.
- **Rendering** uses source files, H.264/AAC MP4, 1080×1920, 30 fps, short readable top-safe-area captions, restrained exposure/contrast adjustment, and hard cuts. Final files are decoded and checked with FFprobe before review.
- **Critic** inspects actual rendered frames, one per shot (up to 30), together with timings, story and EDL. It does not listen to the rendered audio or inspect every video frame. One revision maximum; both versions are retained and the highest critic score is selected.
- **Privacy**: salted scrypt passwords, opaque hashed HttpOnly session cookies, ownership checks on every project/asset/result route, range-supported authenticated media streaming, private disk storage, optional private encrypted S3 copies, and project deletion. Feedback is saved with research consent off; no training pipeline exists.

## Storage and hosting

`data/` is private and must never be exposed as a static directory. The browser uses authenticated API routes for footage and downloads. Set `S3_BUCKET`, optional `S3_ENDPOINT_URL`, and standard AWS credentials to mirror originals and outputs privately. Workers currently need shared filesystem access: S3 is an archival copy, not a replacement for the local worker cache. S3/PostgreSQL deployment requires environment-specific testing.

This is a local MVP, not an internet-hardened hosted service. `LOCAL_MODE=true` automatically creates a private cookie session for one shared local workspace, only for loopback clients and localhost hostnames. Anyone using this computer can access that workspace. The login screen has been removed; backend ownership/session checks remain. Before public hosting, disable local mode and add a sign-in UI, use HTTPS (`COOKIE_SECURE=true`), configure registration policy, request/body limits at a reverse proxy, login rate limiting, disk quotas, backups, database migrations, and a process supervisor. FFmpeg must be isolated and kept updated when handling untrusted public uploads. No application is deployed by this repository setup.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
cd frontend
npm.cmd run build
```

Tests render genuine generated media and cover portrait H.264 output, missing audio, music/caption mixing, tracking expressions, full segmentation coverage, rejected invalid plans, caption source-time mapping, private uploads, cross-user denial, byte ranges, duplicate job prevention, corrupt media, and deletion. They do not claim synthetic color bars validate editing taste across creator topics.

Optional live API check (consumes API credits): `python -m scripts.live_smoke`. This sends clearly synthetic test clips through actual analysis, planning, rendering, and review. Use the 15-phone-video acceptance workflow below for product validation.

## Real footage acceptance

1. Upload 15 ordinary, untrimmed phone clips from one creator story (travel, lifestyle, food, products, education, fitness, or another topic).
2. Choose High Energy, 45 seconds, and upload a song.
3. Click Create Video; observe actual stage/count messages.
4. Verify a playable 45-second 1080×1920 H.264/AAC MP4 is returned.
5. Inspect `/api/projects/{id}/provenance` while signed in: analyses, selection/rejection reasons, source IDs/ranges, music beats, crop tracks, critic and any revision.
6. Create Another Version and verify the planner changes selections or ordering.
7. Save feedback and delete the project; confirm its media is no longer accessible.

## Operational notes and limits

- An API key being configured does not guarantee model access or sufficient quota. Errors remain visible, and technical traces go to `data/worker.log`.
- Short inputs cannot become a truthful 45-second edit without repetition. Fixed lengths require enough source footage; Auto may produce less than 30 seconds if sources are shorter.
- Approximate histogram matching is not a learned video embedding model. It assists duplicate penalties but does not establish event identity or camera sync.
- KLT tracks appearance features, not skeletons; unreliable cases preserve the frame. Optical stabilization is not yet implemented.
- Caption placement is in a fixed upper safe area; it does not perform full moving-object avoidance. Speech models can hallucinate in noisy footage; confidence filtering reduces but cannot eliminate this. Gemini phrase timestamps are model estimates and need evaluation against real creator audio.
- Analysis is sequential and sampled, so fine-grained action boundaries, edit taste, audio classification and beat grids need evaluation on real creator footage. Large projects may take considerably longer than a few minutes.
- API token usage and stage latencies are recorded. Dollar cost remains `null` because account-specific current prices are not inferred. Gemini speech calls are included; optional Whisper calls are not included in token totals.
- Source upload SHA-256 rejects exact duplicate files within a project. Per-project analysis/transcript caching never exposes one creator's cache to another.

Gemini references: [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output), [Audio understanding](https://ai.google.dev/gemini-api/docs/audio), [Pricing and data terms](https://ai.google.dev/gemini-api/docs/pricing).

Optional OpenAI references: [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Timestamped transcription](https://developers.openai.com/api/docs/guides/speech-to-text).

Uploads stream directly to private storage without a multipart temporary copy. A 64 MB free-space reserve remains enforced, partial uploads are removed on failure, and completed renders discard intermediate chunk files. Temporary transcription WAV files and test media are cleaned up. Gemini transient failures use up to five attempts with bounded backoff; quota errors without a short provider retry delay stop immediately.


## Subtitle studio

Choose detected speech, write your own timed lines, or turn subtitles off. Pop uses a quick scale bounce, Karaoke sweeps an accent color over words, Comic uses tilted bold type with heavy outlines, and Behind person uses a local PP-HumanSeg mask to restore people over the lettering. Choose accent color and top/center/bottom placement for foreground styles. Long lines display in short bursts; karaoke timing is estimated from phrase duration, not forced word alignment.

Finished videos expose **Apply subtitles & export**. This queues a local render using the existing EDL and original footage, without new AI calls or changes to the cuts. It saves a separate version; use the saved-version selector to return to earlier exports. Enter times against the final video. Source speech timing is mapped through cuts. The source video is rebuilt so captions never accumulate over earlier burned-in text.

The behind-person model is a 6.2 MB OpenCV Zoo PP-HumanSeg ONNX model, processed locally. `python -m scripts.setup_caption_model` downloads it with SHA-256 verification and its license. Windows setup includes this step. Background masking is approximate, especially for hair, motion blur, small subjects and crowded scenes; it preserves detected people rather than selecting a single person. If nobody is detected, the text remains visible over the scene. Style cards are illustrative previews; the exported MP4 is authoritative. [Model source](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg).
