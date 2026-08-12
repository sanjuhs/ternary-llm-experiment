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

Verified on 2026-08-10:

- Cold `/health` request reached HTTP 200 in 49.94 seconds.
- `/`, `/audio_duplex`, `/omni`, and `/health` all returned HTTP 200.
- A real WSS lifecycle reached `queue_done` -> `prepared` -> `stopped` in 2.72s.
- Eight one-second chunks of OpenBMB's sample audio produced a 489.2ms mean and
  904.4ms maximum server `cost_all_ms`; all eight stayed below the one-second
  real-time compute budget.
- Sequential test-client round trips averaged 1.15s because they include network
  overhead and intentionally wait for each reply. The browser streams its input
  every second without that artificial request/reply serialization.

## GPU choice

The model fits RTX PRO 6000 Blackwell, H100, and L40S. H100 SXM is the fastest
conservative choice because its 3.35 TB/s HBM3 bandwidth is materially above
RTX PRO 6000's 1.6 TB/s GDDR7 bandwidth, and its Hopper kernels have mature
PyTorch/Triton support. Modal can also upgrade an H100 request to H200 at the
H100 price. Therefore this source requests `H100`.

The currently live legacy endpoint remains on L40S with compilation disabled.
The optimized deployment in this folder enables compilation and persistent
kernel caching, but Modal rejected activation of H100, RTX PRO 6000, and L40S
functions because this account has no payment method. Add one in Modal billing,
then rerun `modal deploy`; failed deploy attempts did not replace the working
legacy endpoint.

## Performance configuration

- BF16 model weights (no quality-lowering quantization).
- PyTorch 2.8 and CUDA 12.8.
- SDPA attention, matching the robust path in the current official installer.
- `torch.compile` enabled with the official real-duplex warmup.
- Persistent Inductor/Triton cache in `minicpm-omni-compile-cache`.
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
python modal_minicpm_omni/monitor_report.py \
  modal_minicpm_omni/monitoring/keepalive.jsonl --minimum-hours 5
```

`smoke_test.py` verifies the public routes and the complete WSS queue/prepare/
stop lifecycle. `--routes-only` is safe for frequent keepalives because it does
not claim the app's single duplex worker. Pass mono 16 kHz float32 PCM with
`--audio-f32` to measure the server's per-unit real-time budget.
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
`d0a002093615b7f1d4d0f87a03fc01cb39bef3f6` (2026-08-05). This keeps the
experiment reproducible and prevents a later upstream push from silently
changing the deployed system. The image also works around that revision's
temporary `librosa` dependency-resolution conflict while retaining its newer,
maintained `librosa` requirement.
