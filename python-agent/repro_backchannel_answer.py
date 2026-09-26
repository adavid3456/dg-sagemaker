#!/usr/bin/env python3
"""Reproduce Conversational Mode filtering a short ANSWER as a user backchannel.

Observed live on 2026-09-25: the agent asked "Do you have a trade-in vehicle you'd
like to bring in as well?", the caller answered "Yeah. Not right now.", and stem
logged `Pure backchannel turn - filtering from LLM`, so the agent never heard it.

This takes the LLM and local playback out of the picture:

  * The question is spoken with `InjectAgentMessage`, so its text is identical
    every trial (its TTS length still varies a little, ~3 s).
  * The client plays no audio. It times the agent audio's end itself: first agent
    audio byte + bytes / (24000 Hz * 2 bytes).
  * The answer WAV is streamed at a fixed offset, with silence everywhere else.
    `--anchor end` (default) measures from the question audio's end (negative =
    overlapping it). `--anchor start` measures from its first byte, which is the
    only way to overlap by more than ~1 s: the end is known only once stem sends
    `AgentAudioDone`, which lands just ~1 s before the audio actually ends.
  * Every trial is a fresh session, so no history or frustration carries over.

Standalone: needs only Python 3.9+ and `pip install websockets`.

Make the two answer WAVs (macOS):
    say -v Samantha -o yeah_not.aiff "Yeah. Not right now."
    say -v Samantha -o no_not.aiff "No, not right now."
    for f in yeah_not no_not; do afconvert -f WAVE -d LEI16@16000 -c 1 $f.aiff $f.wav; done

Self-hosted stem with a local vLLM `think` leg (the setup it was found on):
    python repro_backchannel_answer.py --url ws://127.0.0.1:8092/v1/agent/converse \\
        --anchor start --offsets 1.0 --repeats 5 --phrase yeah_not.wav --control no_not.wav

Hosted Voice Agent (needs a Conversational Mode EA-enabled key):
    python repro_backchannel_answer.py --url wss://agent.deepgram.com/v1/agent/converse \\
        --token "$DEEPGRAM_API_KEY" --think hosted \\
        --anchor start --offsets 1.0 --repeats 5 --phrase yeah_not.wav --control no_not.wav

Outcome per trial: FILTERED (a `user_backchannel` frame, and stem dropped the
utterance), TURN (a user `ConversationText`, so the LLM saw it), or NONE.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave

import websockets

CHUNK_MS = 20
INPUT_RATE = 16000       # caller audio in: linear16 mono
OUTPUT_RATE = 24000      # agent audio out: linear16 mono
QUESTION = "Do you have a trade-in vehicle you'd like to bring in as well?"
CHUNK = INPUT_RATE * 2 * CHUNK_MS // 1000   # bytes per 20 ms of 16 kHz int16
OUT_BYTES_PER_S = OUTPUT_RATE * 2           # agent audio is linear16 @ 24 kHz


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (INPUT_RATE, 1, 2):
            sys.exit(f"{path}: need 16 kHz mono int16")
        return w.readframes(w.getnframes())


LOCAL_THINK = {
    "provider": {"type": "open_ai", "model": "dsva"},
    "endpoint": {"url": "http://127.0.0.1:3081/v1/chat/completions", "headers": {}},
}
HOSTED_THINK = {"provider": {"type": "open_ai", "model": "gpt-4o-mini"}}


def settings(think: str) -> dict:
    think_block = dict(LOCAL_THINK if think == "local" else HOSTED_THINK)
    think_block["prompt"] = ("You are Jordan, a car sales specialist. Reply in one "
                             "short spoken sentence.")
    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": INPUT_RATE},
            "output": {"encoding": "linear16", "sample_rate": OUTPUT_RATE,
                       "container": "none"},
        },
        "agent": {
            "greeting": "Hi.",
            "listen": {"provider": {"type": "deepgram", "model": "flux-general-multi",
                                    "version": "v2"}},
            "think": think_block,
            "speak": {"provider": {"type": "deepgram", "version": "v2",
                                   "model": "flux-alexis-en"}},
            # Server defaults for every tuning knob, so the result reflects the
            # deployed config, not client overrides.
            "think_conversational": {"provider": {"type": "deepgram",
                                                  "telemetry": "standard"}},
        },
    }


async def trial(url: str, answer: bytes, offset_s: float, listen_s: float,
                anchor: str = "end", token: str = "", think: str = "local") -> dict:
    out: dict = {"offset": offset_s, "outcome": "NONE", "confidence": None,
                 "user_text": None, "agent_reply": None, "note": ""}
    pending = bytearray()          # answer audio waiting to be streamed
    state = {"phase": "greeting", "audio_t0": None, "audio_bytes": 0,
             "done_seen": False}

    headers = {"Authorization": f"Token {token}"} if token else None
    async with websockets.connect(url, max_size=None, additional_headers=headers) as ws:
        await ws.send(json.dumps(settings(think)))

        async def sender() -> None:
            start, n = time.monotonic(), 0
            while True:
                if pending:
                    chunk = bytes(pending[:CHUNK]).ljust(CHUNK, b"\x00")
                    del pending[:CHUNK]
                else:
                    chunk = b"\x00" * CHUNK
                await ws.send(chunk)
                n += 1
                await asyncio.sleep(max(0.0, start + n * CHUNK_MS / 1000 - time.monotonic()))

        async def receiver() -> None:
            async for msg in ws:
                if isinstance(msg, bytes):
                    # Keep counting after the answer starts, until the question's own
                    # AgentAudioDone, so question_s is its full (or cut-off) length.
                    if state["phase"] in ("question", "answered") and not state["done_seen"]:
                        if state["audio_t0"] is None:
                            state["audio_t0"] = time.monotonic()
                        state["audio_bytes"] += len(msg)
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "AgentAudioDone":
                    if state["phase"] == "greeting":
                        state["phase"] = "greeting_done"
                    elif state["phase"] in ("question", "answered"):
                        state["done_seen"] = True
                elif t == "ConversationText" and state["phase"] == "answered":
                    if m.get("role") == "user":
                        out["user_text"] = m.get("content")
                        out["outcome"] = "TURN"
                    elif m.get("role") == "assistant" and out["agent_reply"] is None:
                        out["agent_reply"] = m.get("content")
                elif (t == "ConversationalEvent" and m.get("event") == "user_backchannel"
                      and state["phase"] == "answered"):
                    out["outcome"] = "FILTERED"
                    out["user_text"] = m.get("text")
                    out["confidence"] = m.get("confidence")
                elif t in ("Error", "Warning"):
                    out["note"] += f"{t}:{m.get('description') or m.get('code')} "

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        try:
            async def wait_for(cond, timeout):
                end = time.monotonic() + timeout
                while not cond():
                    if time.monotonic() > end:
                        return False
                    await asyncio.sleep(0.01)
                return True

            if not await wait_for(lambda: state["phase"] == "greeting_done", 20):
                out["note"] += "no greeting "
                return out
            await asyncio.sleep(1.5)
            state["phase"] = "question"
            await ws.send(json.dumps({"type": "InjectAgentMessage", "content": QUESTION}))
            if anchor == "start":
                if not await wait_for(lambda: state["audio_t0"] is not None, 20):
                    out["note"] += "question audio never started "
                    return out
                start_at = state["audio_t0"] + offset_s
            else:
                if not await wait_for(lambda: state["done_seen"], 20):
                    out["note"] += "question audio never finished "
                    return out
                start_at = (state["audio_t0"]
                            + state["audio_bytes"] / OUT_BYTES_PER_S + offset_s)
            lateness = time.monotonic() - start_at
            if lateness > 0.05:
                out["note"] += f"started {lateness:.2f}s late "
            await asyncio.sleep(max(0.0, start_at - time.monotonic()))
            state["phase"] = "answered"
            pending.extend(answer)
            await asyncio.sleep(len(answer) / (INPUT_RATE * 2) + listen_s)
            out["question_s"] = round(state["audio_bytes"] / OUT_BYTES_PER_S, 2)
        finally:
            send_task.cancel()
            recv_task.cancel()
    return out


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8092/v1/agent/converse",
                   help="Voice Agent WebSocket URL")
    p.add_argument("--token", default="", help="Deepgram API key (hosted only)")
    p.add_argument("--think", choices=["local", "hosted"], default="local",
                   help="local = self-hosted vLLM on 127.0.0.1:3081 served as 'dsva'; "
                        "hosted = OpenAI gpt-4o-mini via Deepgram")
    p.add_argument("--phrase", required=True, help="the answer that gets misfiled")
    p.add_argument("--control", help="a phrase expected to always be a turn")
    p.add_argument("--offsets", type=float, nargs="+", default=[-1, -0.5, 0, 0.5, 1, 2, 3],
                   help="seconds from the question audio's end (negative = overlapping)")
    p.add_argument("--anchor", choices=["end", "start"], default="end",
                   help="measure --offsets from the question audio's end or start")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--listen", type=float, default=6.0,
                   help="seconds to wait after the answer for the verdict")
    args = p.parse_args()

    url = args.url
    phrases = [("phrase", args.phrase)] + ([("control", args.control)] if args.control else [])
    rows = []
    for label, path in phrases:
        audio = read_wav(path)
        for off in args.offsets:
            for rep in range(args.repeats):
                r = await trial(url, audio, off, args.listen, args.anchor,
                                args.token, args.think)
                r.update(label=label, rep=rep)
                rows.append(r)
                conf = f"{r['confidence']:.2f}" if isinstance(r["confidence"], float) else "-"
                print(f"{label:7} offset={off:+5.1f}s rep={rep} {r['outcome']:8} "
                      f"conf={conf:5} q={r.get('question_s')}s user={r['user_text']!r} "
                      f"agent={r['agent_reply']!r} "
                      f"{r['note']}", flush=True)
    print("\nsummary (FILTERED / trials):")
    for label, _ in phrases:
        for off in args.offsets:
            sel = [r for r in rows if r["label"] == label and r["offset"] == off]
            f = sum(r["outcome"] == "FILTERED" for r in sel)
            print(f"  {label:7} offset={off:+5.1f}s  {f}/{len(sel)} filtered")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
