# MiniCPM-o 4.5 real-time voice demo on Modal

This deploys OpenBMB's official PyTorch/CUDA MiniCPM-o 4.5 demo, including its
browser UI, audio full-duplex mode, omni audio/video full-duplex mode, and WSS
gateway. It is the BF16 quality-ceiling experiment before any ternarization.

## Live endpoint

The previously deployed L40S version is live at:

`https://sanjuhs123--minicpm-omni-demo.modal.run`

Its important routes are:

- `/audio_duplex` — the first route to try; live microphone in/out.
- `/omni` — live microphone, camera, and model voice.
- `/` — turn-based multimodal chat and basic smoke testing.
- `/health` — gateway and worker status.

Use headphones for the duplex experiments. The upstream project currently
documents echo-cancellation problems that can interfere with interruption when
speakers feed back into the microphone.

Verified on 2026-08-12:

- Cold `/health` request reached HTTP 200 in roughly one minute.
- `/`, `/audio_duplex`, `/omni`, and `/health` all returned HTTP 200.
- A real WSS lifecycle reached `queue_done` -> `prepared` -> `stopped` in 2.72s.
- A ten-minute, 60-question cake conversation completed all 600 one-second audio
  units without a disconnect. BF16 answered 60/60 question slots, reached 49/60
  keyword-quality signals, and kept every server unit below the one-second
  real-time budget (510.2ms p95, 621.1ms maximum).
- The guarded ternary deployment passed the same ten-minute fixture: 600/600
  units, 60/60 answered slots, 50/60 keyword-quality signals, zero duplicate
  turns, and 99.5% of server units below one second (915.1ms p95).
- Context compaction held the BF16 and guarded-ternary KV caches to 3,419 and
  3,539 tokens respectively, below the configured 4,000-token high watermark.

## GPU choice

The live BF16 and guarded-ternary endpoints request Modal's `L40S`. This is not
RunPod and it is not CPU inference: the language model, audio encoder, vision
encoder, TTS model, and vocoder are placed on CUDA. L40S is already comfortably
real-time for BF16 and remains real-time for the current fake-quant guarded
ternary experiment. A faster H100/H200 is unnecessary for this one-user demo;
an RTX PRO 6000 is not presently a Modal GPU option used by this deployment.

## Performance configuration

- BF16 model weights (no quality-lowering quantization).
- PyTorch 2.8 and CUDA 12.8.
- SDPA attention, matching the robust path in the current official installer.
- Eager SDPA by default: it is real-time and avoids a source-keyed 10–16 minute
  compile warmup after reliability patches. Set `MINICPM_ENABLE_COMPILE=1` only
  for dedicated compilation experiments.
- Binary one-second Float32 PCM frames with a JSON/base64 compatibility fallback.
- Context-aware sliding at 4,000 KV tokens, compacting to 3,500 while retaining
  the latest 180 audio units plus up to 500 tokens of older generated context.
- A ten-unit maximum continuous speaking run, after which the server forces a
  return to listening/end-of-turn behavior.
- One user / one worker / one GPU, avoiding concurrency interference.
- 15-minute idle scale-down. A cold request reloads and warms the model.

Modal's public HTTPS endpoint supplies the secure context required by browser
microphone/camera APIs; the local gateway deliberately uses plain HTTP only
inside the container.

## Commands

Run from the repository root:

```bash
modal run modal_minicpm_omni/modal_minicpm_omni.py --action cache
modal run modal_minicpm_omni/modal_minicpm_omni.py --action gpu
modal deploy modal_minicpm_omni/modal_minicpm_omni.py
modal app logs minicpm-omni-45 -f
python modal_minicpm_omni/smoke_test.py
python modal_minicpm_omni/smoke_test.py --routes-only
python modal_minicpm_omni/smoke_test.py --routes-only \
  --record-jsonl modal_minicpm_omni/monitoring/keepalive.jsonl
python modal_minicpm_omni/build_cake_fixture.py
python -m modal_minicpm_omni.long_session_test \
  --base-url https://sanjuhs123--minicpm-omni-demo.modal.run \
  --label bf16-production
python modal_minicpm_omni/monitor_report.py \
  modal_minicpm_omni/monitoring/keepalive.jsonl --minimum-hours 5
```

`smoke_test.py` verifies the public routes and the complete WSS queue/prepare/
stop lifecycle, including binary-audio capability negotiation. `--routes-only`
is safe for frequent keepalives because it does not claim the app's single
duplex worker. Pass mono 16 kHz float32 PCM with `--audio-f32` to measure the
server's per-unit real-time budget. `build_cake_fixture.py` creates an exact
600-second/60-question fixture, and `long_session_test.py` measures delivery,
context growth, sliding events, end-of-turn behavior, transport gaps, latency,
duplicate turns, and simple question-response quality signals.
`--record-jsonl` preserves every timestamped success or failure. After the
monitoring window, `monitor_report.py` proves its duration, checks for gaps over
179 seconds (one second inside the live deployment's three-minute scale-down
window), and fails if any recorded health check failed.

The gateway and model worker run inside the same GPU-backed Modal `web_server`
container. Therefore even the non-reserving `/health` request refreshes that
container's scale-down window and keeps the loaded model resident; it does not
need to occupy the single duplex conversation slot.

The model checkpoint already exists in the `minicpm-omni-cache` Volume for the
current Modal workspace. In a fresh workspace, create the Hugging Face secret
and populate the cache once:

```bash
modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"
modal run modal_minicpm_omni/modal_minicpm_omni.py --action download
```

Do not put a Hugging Face token in this repository.

## Cost behavior

`min_containers=0` prevents a GPU from running continuously. The H100 stays warm
for 15 minutes after activity to support an uninterrupted evaluation session,
then scales to zero. Opening the URL after it has scaled down causes a cold
start; refresh when `/health` is ready.

## Source pin

The demo is pinned to OpenBMB/MiniCPM-o-Demo revision
`b45ce889c5086d506434f4acc2e2b59ff7191bff` (2026-05-20). This keeps the
experiment reproducible and prevents a later upstream push from silently
changing the deployed system. The image also works around that revision's
temporary `librosa` dependency-resolution conflict while retaining its newer,
maintained `librosa` requirement.
