# python-agent — Voice Agent driver for the self-contained container

Drives Deepgram's Voice Agent (`/v1/agent/converse`) against the self-contained
SageMaker image, where Flux ASR, a local LLM, and Flux TTS all run inside one
container with no egress.

Built for the image in `deepgram-aws-sagemaker` at
`docker/inference-billing-va/Dockerfile`.

```bash
uv sync
```

## Check the container first

```bash
uv run agent_converse.py health --host <ec2-host>
```

Probes each process on its own port and reports which are up:

| component | port | probe |
|---|---|---|
| impeller-asr | 8083 | `/v2/models` |
| impeller-tts | 8183 | `/v2/models` |
| vLLM | 3081 | `/v1/models` — **loopback-only, reported SKIP** |
| stem | 8092 | `/health`, `/v1/agent/settings/think/providers` |
| inference-shim | 8080 | `/ping` |

Exits non-zero if anything is down. Worth running before every conversation —
it turns "the agent didn't reply" into a named component in a couple of seconds.

**vLLM always reports SKIP, and that is correct.** The entrypoint starts it with
`--host 127.0.0.1`, so it listens on the *container's* loopback and `-p 3081:3081`
cannot reach it. That is deliberate — the LLM is an internal leg that stem dials
on loopback from inside the container, and it should not be exposed. Check it
directly instead:

```bash
docker exec <container> curl -sf http://127.0.0.1:3081/v1/models
```

An HTTP error on that port (as opposed to a connection failure) is still reported
as a real FAIL, since something answered.

## Run a conversation

```bash
uv run agent_converse.py talk --host <ec2-host> \
    --file ../test-audio-files/127389__acclivity__thetimehascome_mono16k.wav \
    --out reply.wav
```

Streams the WAV at real-time pace, holds the socket open (`--linger`, default
20 s) while end-of-turn fires and the LLM and TTS run, then scores the three legs
independently:

```
  PASS  ASR  (Flux -> user transcript)     1 turn(s)
  PASS  LLM  (agent text reply)            1 turn(s)
  PASS  TTS  (non-silent audio out)        412800 bytes, peak amplitude 8134
```

Input must be **mono 16-bit PCM at 16 kHz** — stem transcodes into Flux at that
rate regardless, so matching it avoids a resample and makes a mismatch fail here
rather than degrade quality silently.

## Conversational Mode (Chatty)

Add `--conversational` to `talk` or `agent_live.py` to send
`agent.think_conversational`. The container must carry the chatty package
(`health` POSTs its three routes and prints `WARN` if they return 503).

```bash
uv run agent_converse.py talk --host <host> --file audio16k.wav              # classic
uv run agent_converse.py talk --host <host> --file audio16k.wav --conversational
uv run agent_live.py --conversational --presence-check-interval-ms 5000
```

**On laptop speakers the agent hears itself** through the mic and barges in on
its own voice. Use headphones, or add `--push-to-talk`: the mic starts muted, and
Enter toggles it (a terminal sees no key-release, so this is a toggle, not
hold-to-talk). While muted the client streams silence instead of stopping, so
turns still end and presence checks still fire. What you give up is barge-in and
user backchannels during the agent's speech, unless you unmute first.

**Escalation demo (default in `agent_live.py`).** Stem only *reports* frustration:
`FrustrationDetected` carries `escalated=true` once the frustrated-turn count reaches
the server's `max_count` (2 in this container; override with
`--frustration-max-count`). It speaks nothing itself, by design, because the
customer's app owns the handoff. On the first escalated event the live client plays
that app:

1. It sends `UpdatePrompt` with a senior-specialist persona. Stem keeps earlier
   prompts in history, so this layers on top of the original prompt.
2. It sends `InjectAgentMessage` with `behavior: "queue"`, so the handoff line plays
   right after the agent's current reply. The default behavior would be refused
   mid-reply. If the user is speaking, stem refuses the injection anyway; the client
   retries up to 3 times.

`--no-escalate-demo` just prints the event. `--handoff-line` and
`--specialist-prompt` change the wording. Repeating the same unanswered question 2–3
times is the most reliable way to trigger it.

It is independent of `think`: Conversational Mode decides *when* the agent speaks,
`think` decides *what* it says. The tuning flags (`--agent-backchannel-sensitivity`,
`--presence-check-max-count`, `--presence-check-interval-ms`,
`--frustration-max-count`, `--restricted-terms`) are sent only when given. Otherwise
the server's `[voice_agent.chatty_integration]` defaults apply.

`talk` requests `standard` telemetry by default and scores an extra line, `CHATTY`,
which passes when at least one `eot_score` frame arrived. That frame proves stem
reached ConvoLLM (`/v2/eot_bc`). Backchannels and probe decisions are reported but
not scored, because whether they fire depends on the audio.

## Session recordings (on by default)

`agent_live.py` saves every session to `./recordings/` (gitignored) when you hang up:

- `<time>[_<scenario>]_stereo_24k.wav`: you on the left, the agent on the right,
  time-aligned. This is the one to listen to.
- `..._user_16k.wav`: exactly what was sent to stem (silence while muted with
  `--push-to-talk`).
- `..._agent_24k.wav`: exactly what your speakers played, after any barge-in flush,
  so it's what you actually heard.

`--no-record` turns it off; `--record-dir` moves it. Recordings contain your voice,
and this repo is public, so keep them out of git (the `.gitignore` entry does that).

## Role-play scenarios

`--scenario` (both clients) swaps the generic assistant for a fictional business.
It sets the prompt and greeting, and in `agent_live.py` also the escalation handoff
line and specialist persona, so a frustrated caller is escalated within the same
business. Explicit `--prompt` / `--greeting` / `--handoff-line` /
`--specialist-prompt` still win. The presets live in `scenarios.py`.

| scenario | agent → escalates to | what the agent does |
|---|---|---|
| `car-dealership` | Jordan (sales) → Casey (sales manager) | finds a vehicle, discusses trade-in and financing generally, books a test drive |
| `pharmacy` | Riley (support) → Dr. Patel (pharmacist) | refills, order and pickup status, transfers. Verifies name and DOB first, gives no medical advice, sends emergencies to 911 |
| `home-loan` | Morgan (intake) → Alex (senior loan officer) | collects pre-qualification details one at a time (name, location, purchase or refi, price, down payment, income, employment, credit range) and reads them back |

```bash
uv run agent_live.py --conversational --scenario home-loan
```

Every prompt forbids inventing facts the agent can't know: prices, stock, rates,
approvals, a patient's medication. The LLM behind this container has no tools or
data, so a confident invention would sound like a pipeline bug. Stay on-topic: the
Qwen3.6 stand-in holds the persona well, but these presets were written for the
weaker Qwen3-4B, so they still steer conservatively.

## Pizza ordering demo (function calling + live order board)

`pizza_demo.py` shows the agent calling functions. Tony takes a phone order off a
fixed menu, and a local web page (http://127.0.0.1:8765, opened automatically)
shows the menu, the order as it builds, every function call with its result, and
the transcript. You talk to Tony **through the page**: click *Start call*.

```bash
uv run pizza_demo.py --host <public-ip>                    # laptop speakers are fine
uv run pizza_demo.py --host <public-ip> --conversational
uv run pizza_demo.py --host <public-ip> --terminal-audio   # old terminal mic/speaker path
```

- **Browser audio and echo cancellation.**
  - The page captures the mic with `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}})`. The browser's WebRTC echo canceller removes Tony's voice from the mic, so laptop speakers work without push-to-talk. The header shows whether the browser actually applied echo cancellation.
  - Mic audio goes to the script as 16 kHz PCM over the page's WebSocket, and the script relays it to stem. Tony's 24 kHz audio comes back the same way.
- **Playback path.**
  - Tony's audio plays through a local WebRTC loopback (two `RTCPeerConnection`s in the page) into an `<audio>` element. Chrome's echo canceller has historically only subtracted audio played through WebRTC.
  - `?playback=direct` on the page URL plays through plain Web Audio instead, for an A/B.
  - Both paths re-buffer on a mid-reply underrun (`--playback-buffer-ms`).
- **Page address.** Open the page as `127.0.0.1` or `localhost`. Browsers only allow the mic from a secure context, and a plain-http LAN address isn't one.
- **Muting and calls.**
  - *Mute* sends silence rather than stopping the stream, because stem needs continuous audio.
  - Closing the tab hangs up.
  - Each call starts a fresh order.
- **`--terminal-audio`** keeps the `agent_live.py` loop, including push-to-talk and the escalation demo. In that mode the page is a display only.

- **Functions.** The five functions are client-side: `add_item`, `update_item`,
  `remove_item`, `set_customer` and `submit_order`.
  - Stem sends each call as a `FunctionCallRequest`.
  - The script applies it to the order it holds and answers with a `FunctionCallResponse` containing the updated lines and total. Prices the agent reads out come from the script, not the LLM.
  - Bad calls (off-menu item, missing size, delivery with no address) come back as `ok:false` with the reason, and the agent asks the caller.
- **`defer_until_eot`.** Every function sets it, so stem only dispatches a call
  after a confirmed end of turn. An eager end-of-turn draft can't add the same
  pizza twice.
- **Other flags.** `--conversational` and the other Conversational Mode flags,
  `--host`/`--port`, recording (`<time>_pizza_*.wav`) and `--playback-buffer-ms`
  work in both modes. `--web-port` and `--no-browser` control the board.
- **Checked 2026-09-25 without a GPU box.** Headless Chrome drove the page, with
  its fake microphone, against a stand-in stem:
  - mic PCM reached "stem" at real time (32 KB/s);
  - the browser reported echo cancellation on;
  - agent audio came out of the loopback `<audio>` element;
  - a function call updated the board, and barge-in, hang-up and the recording all worked.

  How well echo cancellation works on real speakers still needs a live call.

Verified 2026-09-25 against the self-contained container (Qwen3.6-35B-A3B-FP8
through vLLM's `qwen3_coder` tool parser):

- A scripted five-turn order ran end to end: add, modify, remove, set customer,
  submit. It made seven function calls, and the page's state matched.
- On the first turn, the model sometimes promises to add an item without calling
  `add_item`. That happened in 1 of 6 runs even with the prompt rules added
  against it. If the board doesn't update, say "add that".

## Reproducing the "answer filtered as backchannel" issue

`repro_backchannel_answer.py` makes a Chatty misclassification deterministic, for
reporting. It speaks a fixed yes/no question with `InjectAgentMessage`, streams an
answer WAV at a fixed offset (the client plays no audio), and uses a fresh session
per trial:

```bash
uv run repro_backchannel_answer.py --host 127.0.0.1 --anchor start \
    --phrase yeah_not.wav --control no_not.wav --offsets 1.0 --repeats 5
```

On 2026-09-25 (Qwen3.6-35B-A3B-FP8, prod chatty config), "Yeah. Not right now."
starting 1.0 s into the question was filtered 5/5, every time at confidence 0.64.
"No, not right now." at the same moment was 0/5. Starting ≥2 s in, both are normal
turns. Hosted reproduces it too (`--url wss://agent.deepgram.com/v1/agent/converse
--token "$DEEPGRAM_API_KEY" --think hosted`): 4/5 filtered, control 0/5. The script
is standalone (no imports from this folder), so it can be shared as a single file. Use `--anchor start`: the end-anchored default can't overlap by more than
~1 s, because the question's length is known only from `AgentAudioDone`, which
arrives about 1 s before the audio ends. The WAVs are macOS `say -v Samantha`, 16 kHz
mono.

## Notes

- **Port 8092 is stem, not the shim.** Talking to stem directly keeps the
  SageMaker bidi transport and SigV4 out of the picture while the speech pipeline
  is what's being debugged. The shim front door (8080) is what a SageMaker deploy
  uses.
- **`--llm-url` is resolved inside the container**, so it stays loopback
  (`http://127.0.0.1:3081/...`) even when `--host` is a remote EC2 box.
- **`--llm-model` must match the container's vLLM `--served-model-name`**
  (`dsva` by default).
- The `think` leg uses provider type `open_ai` with an explicit endpoint rather
  than stem's built-in `deepgram` provider. The latter also defaults to a
  localhost OpenAI-shaped URL, but it rejects function calling and requires a
  server-side model allowlist; an explicit endpoint skips both.
- `speak` deliberately sends **no** endpoint. Speak endpoints are restricted to
  `https`/`wss` unconditionally, so a loopback one is rejected — leaving it unset
  routes TTS through stem's driver pool to the TTS impeller.

## Audio came back but nothing was transcribed

That is usually the greeting alone: it is synthesized before any audio is sent,
so TTS can succeed while ASR produced nothing. Check the ASR engine and the
`--asr-model` value. The driver calls this case out explicitly when it sees it.

## Agent audio is choppy or breaks up

That is the network, not TTS. Stem paces agent audio to real time with only a 1 s
lead (`STREAMING_PLAYBACK_LEAD_TIME` in stem's `pipeline/tts.rs`), so any stretch
where the link delivers slower than real time drains the speaker queue. Measured
2026-09-25 with identical text:

| where the client ran | lead at the end of each reply |
|---|---|
| on the Sydney host | 1.0 s on all 6 replies |
| a US laptop over the SSM tunnel | 1.0 s on 4 replies, 0.07 s and −0.16 s on the other 2 |

`agent_live.py` and `pizza_demo.py` re-buffer when the queue runs dry mid-reply
(`--playback-buffer-ms`, default 250), which turns the crackle into a short pause.
They print how many times that happened at hang-up. The real fix is to connect
to the host's public IP (`--host <public-ip>`, security group open to your /32 on
8092) rather than over an SSM tunnel: over the public IP from the same laptop the
lead held at 0.77–1.01 s on 12/12 replies.
Recordings show which it was: the agent track has exact-zero holes in multiples
of 20 ms wherever playback ran dry.
