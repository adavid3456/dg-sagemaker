#!/usr/bin/env python3
"""Talk to the self-contained Voice Agent with your microphone, from a local terminal.

Speech in and out stay on your machine's audio devices; the container does ASR,
the LLM, and TTS with no egress of its own.

Connect straight to the EC2 host's public IP, with its security group opening
stem's port 8092 to your own /32 only (stem has no auth):

    uv run agent_live.py --host <public-ip>

Don't use an SSM port-forwarding tunnel for this. Stem paces agent audio at real
time with only a 1 s lead, and over SSM that lead collapsed on 2 of 6 replies,
so playback broke up. Over the public IP it held on 12/12 (2026-09-25).

Press Ctrl-C to hang up.

On laptop speakers the agent hears itself through the mic and barges in on its own
voice. Headphones fix that with every behavior intact; otherwise use
--push-to-talk, where Enter opens the mic and Enter again mutes it. (A terminal
sees no key-release events, so this is a toggle, not hold-to-talk.)

ESCALATION DEMO (on by default with --conversational; --no-escalate-demo turns it
off). Stem only REPORTS frustration: `FrustrationDetected` with `escalated=true`
once the frustrated-turn count reaches the configured max. It deliberately speaks
nothing itself, and the client owns the handoff. So on the first escalated event
this client plays the part of a real integration: it sends `UpdatePrompt` so the
agent carries on as a senior specialist, and `InjectAgentMessage` so the caller
hears the handoff line.

WHY THIS EXISTS ALONGSIDE `agent_converse.py talk`: a canned WAV keeps streaming
while the agent is still speaking its greeting, which triggers barge-in on every
run. A human waits for the greeting, so a live session exercises the ordinary
conversational path instead of the interrupt path.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import queue
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    sys.exit(
        "missing dependency: run `uv sync` in python-agent/ first.\n"
        "sounddevice needs PortAudio; on macOS `brew install portaudio` if the "
        "import still fails."
    )

try:
    import websockets
except ImportError:
    sys.exit("missing dependency: run `uv sync` in python-agent/ first")

from agent_converse import (
    CHUNK_MS,
    INPUT_RATE,
    OUTPUT_RATE,
    add_conversational_args,
    build_settings,
)
from scenarios import add_scenario_arg, apply_scenario

# Mic frames per send. Matches the WAV driver's pacing so both clients present
# the same cadence to Flux's end-of-turn detector.
FRAMES_PER_CHUNK = INPUT_RATE * CHUNK_MS // 1000

# Stem refuses an injection while the USER is speaking, with a Warning
# (InjectAgentMessageDuringUserSpeech); a short pause and a retry covers it.
INJECT_RETRIES = 3
INJECT_RETRY_DELAY_S = 1.5


class Player:
    """Agent-audio queue for a real-time output callback, with a re-buffering jitter
    buffer. `read()` runs on the PortAudio thread; everything else on the event loop.

    Stem paces agent audio to real time with only a 1 s lead
    (STREAMING_PLAYBACK_LEAD_TIME in stem's pipeline/tts.rs), so the network has no
    more than that to spare. When the link delivers slower than real time for a
    while (seen over an SSM tunnel to Sydney: the lead collapsed from 1.0 s to
    about 0 s on 2 of 6 replies, while the same test on the host held 1.0 s every
    time), playing whatever arrives gives a rapid string of 20-160 ms holes, which is
    the choppy sound. Instead, when the queue runs dry mid-reply, `read()` holds
    silence until `rebuffer_bytes` have queued up again, trading the crackle for
    fewer, longer pauses. After AgentAudioDone the reply is fully sent, so its tail
    plays out without waiting to refill. This cannot make a slow link fast; it only
    makes it sound better.
    """

    def __init__(self, rebuffer_bytes: int) -> None:
        self.q: deque[bytes] = deque()
        self.lock = threading.Lock()
        self.rebuffer_bytes = rebuffer_bytes
        self.buffering = False
        self.reply_done = False
        self.rebuffers = 0

    def push(self, chunk: bytes) -> None:
        with self.lock:
            self.reply_done = False  # audio after AgentAudioDone starts a new reply
            self.q.append(chunk)

    def mark_reply_done(self) -> None:
        with self.lock:
            self.reply_done = True

    def flush(self) -> int:
        """Drop everything queued (barge-in). Returns the bytes dropped."""
        with self.lock:
            dropped = sum(len(c) for c in self.q)
            self.q.clear()
            self.buffering = False
            return dropped

    def read(self, need: int) -> bytes:
        buf = bytearray()
        with self.lock:
            if self.buffering:
                queued = sum(len(c) for c in self.q)
                if queued >= self.rebuffer_bytes or (self.reply_done and queued):
                    self.buffering = False
            while not self.buffering and self.q and len(buf) < need:
                chunk = self.q.popleft()
                take = min(len(chunk), need - len(buf))
                buf += chunk[:take]
                if take < len(chunk):
                    self.q.appendleft(chunk[take:])
            if (len(buf) < need and self.rebuffer_bytes and not self.reply_done
                    and not self.buffering):
                self.buffering = True  # ran dry before the reply finished arriving
                self.rebuffers += 1
        if len(buf) < need:  # underrun: play silence rather than glitch
            buf += b"\x00" * (need - len(buf))
        return bytes(buf)


def _resample_16k_to_24k(pcm: bytes) -> bytes:
    """Linear-interpolation 16 kHz -> 24 kHz for int16 mono (ratio 3:2). Only used to
    put the caller track on the agent's sample rate for the stereo mix."""
    src = array.array("h", pcm)
    n_out = len(src) * 3 // 2
    out = array.array("h", bytes(n_out * 2))
    last = len(src) - 1
    for i in range(n_out):
        pos = i * 2 / 3
        j = int(pos)
        frac = pos - j
        a = src[j]
        b = src[j + 1] if j < last else a
        out[i] = int(a + (b - a) * frac)
    return out.tobytes()


def _write_wav(path: Path, pcm: bytes, rate: int, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def save_recording(rec_dir: Path, stem: str, user: bytes, agent: bytes) -> None:
    """Write the caller track, the agent track, and a time-aligned stereo mix
    (caller left, agent right, 24 kHz). Both tracks are appended from real-time
    audio callbacks that start together, so they share one clock."""
    rec_dir.mkdir(parents=True, exist_ok=True)
    user_path = rec_dir / f"{stem}_user_16k.wav"
    agent_path = rec_dir / f"{stem}_agent_24k.wav"
    mix_path = rec_dir / f"{stem}_stereo_24k.wav"
    _write_wav(user_path, user, INPUT_RATE)
    _write_wav(agent_path, agent, OUTPUT_RATE)
    left = array.array("h", _resample_16k_to_24k(user))
    right = array.array("h", agent)
    n = max(len(left), len(right))
    left.extend([0] * (n - len(left)))
    right.extend([0] * (n - len(right)))
    mix = array.array("h", bytes(n * 4))
    mix[0::2] = left
    mix[1::2] = right
    _write_wav(mix_path, mix.tobytes(), OUTPUT_RATE, channels=2)
    print(f"\nrecorded {n / OUTPUT_RATE:.1f}s:\n  {mix_path}\n  {user_path}\n  {agent_path}")


async def run(args, settings: dict | None = None, on_message=None) -> int:
    """Run one live session.

    `settings` replaces the default Settings frame (a demo that adds functions).
    `on_message(ws, m)` is awaited for every JSON server message after the default
    handling, so a demo can answer FunctionCallRequest or mirror the conversation
    somewhere else without re-implementing the audio loop.
    """
    url = f"ws://{args.host}:{args.port}/v1/agent/converse"
    if settings is None:
        settings = build_settings(args)

    # Mic callback runs on a PortAudio thread, so hand frames to asyncio through a
    # plain queue rather than touching the loop from that thread.
    mic_q: queue.Queue[bytes] = queue.Queue()

    # Push-to-talk mutes by sending SILENCE, never by pausing the stream. Stopping
    # would break three things: stem times the socket out ("We waited too long for
    # a websocket message"), Flux never sees the pause that ends your turn, and
    # Conversational Mode's presence checks are driven by exactly that silence.
    talking = threading.Event()
    if not args.push_to_talk:
        talking.set()

    # Local recording (on by default; --no-record). Each track is exactly what
    # crossed that edge: the caller track is what was SENT (silence while muted),
    # the agent track is what the speaker PLAYED (after any barge-in flush).
    rec_user = bytearray()
    rec_agent = bytearray()

    def on_mic(indata, frames, time_info, status):
        if status:
            print(f"  [audio] input status: {status}", file=sys.stderr)
        chunk = bytes(indata) if talking.is_set() else b"\x00" * len(indata)
        if args.record:
            rec_user.extend(chunk)
        mic_q.put(chunk)

    def toggle_on_enter() -> None:
        # Daemon thread: a blocked input() must not keep the process alive at hang-up.
        while True:
            try:
                input()
            except EOFError:
                return
            if talking.is_set():
                talking.clear()
                print("  [mic] MUTED — press Enter to talk", flush=True)
            else:
                talking.set()
                print("  [mic] LIVE — press Enter when you're done", flush=True)

    # Agent audio is played from a callback-fed deque rather than blocking writes,
    # and barge-in clears that deque. Both halves of that matter:
    #
    #   * `RawOutputStream.write()` BLOCKS when the device buffer is full, and each
    #     reply arrives with about a second of audio up front (stem's pacing lead) —
    #     so a blocking write stalls this coroutine and `UserStartedSpeaking` isn't
    #     processed until that backlog drains.
    #   * `stream.stop()` does NOT discard buffered audio; PortAudio drains it first.
    #     So stopping on barge-in still plays the rest of the agent's sentence, which
    #     looks exactly like barge-in being broken when the server side is fine.
    #
    # Dropping the queued bytes is what actually makes the agent go quiet mid-word.
    # Re-buffering on a mid-reply underrun is the Player's job; see its docstring.
    player = Player(OUTPUT_RATE * 2 * args.playback_buffer_ms // 1000)

    def on_speaker(outdata, frames, time_info, status):
        buf = player.read(frames * 2)  # int16 mono
        outdata[:] = buf
        if args.record:
            rec_agent.extend(buf)

    print(f"connecting: {url}")
    print(f"  asr={args.asr_model}  llm={args.llm_model}  tts={args.tts_model}")
    print(f"  scenario={args.scenario or 'generic'}")

    try:
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps(settings))

            speaker = sd.RawOutputStream(
                samplerate=OUTPUT_RATE,
                channels=1,
                dtype="int16",
                blocksize=OUTPUT_RATE * CHUNK_MS // 1000,
                callback=on_speaker,
            )
            speaker.start()
            mic = sd.RawInputStream(
                samplerate=INPUT_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAMES_PER_CHUNK,
                callback=on_mic,
            )
            mic.start()
            if args.push_to_talk:
                threading.Thread(target=toggle_on_enter, daemon=True).start()
                print("\nPush-to-talk: mic MUTED. Press Enter to talk, Enter again to "
                      "mute. Ctrl-C to hang up.\n")
            else:
                print("\nMicrophone is live — start talking. Ctrl-C to hang up.\n")

            async def sender() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    # Block on the queue in a worker thread so the event loop
                    # stays free to receive and play the agent's audio.
                    chunk = await loop.run_in_executor(None, mic_q.get)
                    if chunk is None:  # hang-up sentinel, see `finally` below
                        return
                    await ws.send(chunk)

            escalation = {"done": False, "inject_attempts": 0}

            async def inject_handoff() -> None:
                # `queue`, not the default: the event lands at the END of the
                # caller's turn, just as the agent starts replying, and the default
                # behavior refuses to speak while the agent is mid-turn. `queue`
                # speaks the line right after that reply instead.
                escalation["inject_attempts"] += 1
                await ws.send(json.dumps({
                    "type": "InjectAgentMessage",
                    "content": args.handoff_line,
                    "behavior": "queue",
                }))

            async def escalate() -> None:
                # UpdatePrompt first, so the agent's next reply is already the
                # specialist. Stem keeps earlier prompts as system messages, so
                # this layers on top of the original prompt rather than erasing it.
                print("  [escalation] handing off to the specialist persona", flush=True)
                await ws.send(json.dumps({"type": "UpdatePrompt",
                                          "prompt": args.specialist_prompt}))
                await inject_handoff()

            async def retry_inject() -> None:
                await asyncio.sleep(INJECT_RETRY_DELAY_S)
                await inject_handoff()

            async def receiver() -> None:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        player.push(msg)
                        continue
                    try:
                        m = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    t = m.get("type", "?")
                    if t == "AgentAudioDone":
                        player.mark_reply_done()
                    if t == "ConversationText":
                        role = m.get("role", "?")
                        print(f"  [{role}] {m.get('content', '')}")
                    elif t == "UserStartedSpeaking":
                        # Drop already-queued agent audio so a barge-in sounds like
                        # an interruption instead of both talking at once. The server
                        # has already stopped generating by this point; what remains
                        # is whatever we buffered locally, which is a lot because the
                        # audio arrives faster than real time.
                        if args.barge_in_stops_playback:
                            dropped = player.flush()
                            if dropped and args.verbose:
                                print(
                                    f"  [barge-in] dropped {dropped / 2 / OUTPUT_RATE:.1f}s"
                                    " of queued agent audio"
                                )
                    elif t in ("Error", "Warning"):
                        detail = (
                            m.get("description") or m.get("message") or json.dumps(m)
                        )
                        print(f"  [{t}] {detail}")
                        refused = t == "Warning" and "inject" in json.dumps(m).lower()
                        if (refused and escalation["done"]
                                and escalation["inject_attempts"] < INJECT_RETRIES):
                            asyncio.create_task(retry_inject())
                    elif t == "PromptUpdated" and escalation["done"]:
                        print("  [escalation] specialist prompt accepted", flush=True)
                    elif t == "ConversationalEvent":
                        # Conversational Mode telemetry (--conversational). Show the
                        # events a listener can hear; the rest only under --verbose.
                        ev = m.get("event", "?")
                        if ev in ("agent_backchannel", "user_backchannel"):
                            print(f"  [{ev}] {m.get('text', '')}")
                        elif ev == "probe_decision" and m.get("spoke"):
                            print(f"  [probe] trigger={m.get('trigger')} "
                                  f"intent={m.get('intent')}")
                        elif args.verbose:
                            print(f"  [{ev}] {json.dumps(m)}")
                    elif t == "FrustrationDetected":
                        print(f"  [FrustrationDetected] count={m.get('count')} "
                              f"escalated={m.get('escalated')}")
                        # `escalated` stays true on every later count too; hand off
                        # once, on the first.
                        if (m.get("escalated") and args.escalate_demo
                                and not escalation["done"]):
                            escalation["done"] = True
                            await escalate()
                    elif args.verbose:
                        print(f"  [{t}]")
                    if on_message is not None:
                        await on_message(ws, m)

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())
            try:
                done, pending = await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
            finally:
                # Wake the executor thread parked in mic_q.get(). Cancelling the
                # sender task does not interrupt it, and interpreter shutdown
                # then joins that thread forever, which is the traceback Ctrl-C
                # used to print on hang-up.
                mic_q.put(None)
                if player.rebuffers:
                    print(f"\n  [playback] audio ran dry mid-reply "
                          f"{player.rebuffers}x and was re-buffered "
                          f"({args.playback_buffer_ms} ms); the network is not keeping "
                          "up with stem's 1 s lead")
                mic.stop()
                mic.close()
                speaker.stop()
                speaker.close()
                if args.record and (rec_user or rec_agent):
                    try:
                        save_recording(Path(args.record_dir),
                                       time.strftime("%Y%m%d-%H%M%S")
                                       + (f"_{args.scenario}" if args.scenario else ""),
                                       bytes(rec_user), bytes(rec_agent))
                    except OSError as e:
                        print(f"\n  [record] could not save recording: {e}",
                              file=sys.stderr)
    # A security group that doesn't admit this machine DROPS the SYN, so the usual
    # symptom is a TimeoutError, not a refusal: most often the laptop's public IP
    # changed (VPN, another network) since the /32 rule was written.
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        timed_out = isinstance(e, TimeoutError)
        print(
            f"\nCould not reach {url} ({type(e).__name__}).\n"
            + (
                "The connection timed out. The security group is most likely not "
                "admitting your current IP; check it with `curl -s "
                "https://checkip.amazonaws.com` and re-authorize that /32 on port "
                "8092.\n"
                if timed_out
                else "Nothing is listening on that port: is the container up?\n"
            )
            + f"\nVerify before retrying:  curl -s -o /dev/null -w '%{{http_code}}\\n' "
            f"http://{args.host}:{args.port}/health   # expect 204\n",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="the EC2 host's public IP (default 127.0.0.1, for a client "
                        "running on the host itself)")
    p.add_argument("--port", type=int, default=8092, help="stem (default 8092)")
    p.add_argument("--asr-model", default="flux-general-multi")
    p.add_argument("--tts-model", default="flux-alexis-en")
    p.add_argument("--llm-model", default="dsva",
                   help="must match the container's vLLM --served-model-name")
    p.add_argument("--llm-url", default="http://127.0.0.1:3081/v1/chat/completions",
                   help="resolved INSIDE the container, so keep it loopback")
    p.add_argument("--prompt", help="agent prompt (default: the scenario's, else "
                                    "a generic voice assistant)")
    p.add_argument("--greeting", help="agent greeting (default: the scenario's)")
    add_scenario_arg(p)
    p.add_argument("--barge-in-stops-playback", action="store_true", default=True,
                   help="flush queued agent audio when you start speaking")
    p.add_argument("--verbose", action="store_true", help="print every event")
    p.add_argument("--playback-buffer-ms", type=int, default=250,
                   help="when agent audio runs dry mid-reply, pause until this much "
                        "is buffered again (default 250; 0 = play whatever arrives)")
    p.add_argument("--no-escalate-demo", dest="escalate_demo", action="store_false",
                   help="on escalated frustration, only print the event (default: "
                        "hand off to a specialist persona via UpdatePrompt + "
                        "InjectAgentMessage)")
    p.add_argument("--handoff-line",
                   help="what the agent says when it escalates (default: the "
                        "scenario's)")
    p.add_argument("--specialist-prompt",
                   help="prompt layered on via UpdatePrompt at escalation "
                        "(default: the scenario's)")
    p.add_argument("--no-record", dest="record", action="store_false",
                   help="don't save the session audio (default: save caller, agent, "
                        "and a stereo mix under --record-dir)")
    p.add_argument("--record-dir", default="recordings",
                   help="where recordings go (default: ./recordings, gitignored)")
    p.add_argument("--push-to-talk", action="store_true",
                   help="start muted; Enter toggles the mic (for laptop speakers, "
                        "where the agent otherwise hears itself)")
    add_conversational_args(p)
    return p


def main() -> int:
    args = build_parser().parse_args()
    apply_scenario(args)

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nhung up.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
