#!/usr/bin/env python3
"""Reproduce Conversational Mode speaking an AskContinue nudge over a caller who
is still talking, with no barge-in.

Observed live on 2026-09-25: mid-answer, the caller paused ~1.5 s, the sentinel
spoke "Go ahead, Victor. What day works best for you?" (trigger=gap,
intent=AskContinue), and it played to the end while the caller kept talking.
Stem only barges in on an utterance START (listen.rs `UtteranceStarted`), and an
AskContinue lands inside an utterance that is already open, so continued speech
never interrupts it.

Per trial (fresh session, client plays no audio):
  1. InjectAgentMessage asks a question; wait for its audio to finish playing.
  2. Stream part A (ends mid-thought), then --pause seconds of silence, then
     part B (keeps talking ~6 s).
  3. Record whether an AskContinue was spoken, how long its audio overlapped
     part B, and whether stem sent UserStartedSpeaking (a barge-in) while the
     nudge was still audible.

Standalone: Python 3.9+ and `pip install websockets`.

Make the two WAVs (macOS):
    say -v Samantha -o partA.aiff "So I'm driving in from pretty far away, and I was thinking maybe we could do next Monday, or maybe"
    say -v Samantha -o partB.aiff "next Tuesday, whichever one works better for your schedule, because I also need to arrange time off from work."
    for f in partA partB; do afconvert -f WAVE -d LEI16@16000 -c 1 $f.aiff $f.wav; done

Hosted:
    python repro_askcontinue_overlap.py --url wss://agent.deepgram.com/v1/agent/converse \\
        --token "$DEEPGRAM_API_KEY" --think hosted --pauses 1.0 1.5 2.0 2.5 --repeats 3
Self-hosted (local vLLM think leg):
    python repro_askcontinue_overlap.py --url ws://127.0.0.1:8092/v1/agent/converse \\
        --pauses 1.0 1.5 2.0 2.5 --repeats 3
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
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = INPUT_RATE * 2 * CHUNK_MS // 1000
OUT_BYTES_PER_S = OUTPUT_RATE * 2
QUESTION = "What day works best for you to come in for a test drive?"

LOCAL_THINK = {
    "provider": {"type": "open_ai", "model": "dsva"},
    "endpoint": {"url": "http://127.0.0.1:3081/v1/chat/completions", "headers": {}},
}
HOSTED_THINK = {"provider": {"type": "open_ai", "model": "gpt-4o-mini"}}


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (INPUT_RATE, 1, 2):
            sys.exit(f"{path}: need 16 kHz mono int16")
        return w.readframes(w.getnframes())


def settings(think: str) -> dict:
    think_block = dict(LOCAL_THINK if think == "local" else HOSTED_THINK)
    think_block["prompt"] = ("You are Jordan, a car sales specialist booking a test "
                             "drive. Reply in one short spoken sentence.")
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
            # Server defaults for every tuning knob.
            "think_conversational": {"provider": {"type": "deepgram",
                                                  "telemetry": "standard"}},
        },
    }


async def trial(url, part_a, part_b, pause_s, token, think, listen_s=6.0,
                trace=False) -> dict:
    out = {"pause": pause_s, "nudged": False, "nudge_text": None, "nudge_audio_s": 0.0,
           "overlap_s": 0.0, "barge_in_during_nudge": False, "note": ""}
    pending = bytearray()
    st = {"phase": "greeting", "q_t0": None, "q_bytes": 0, "q_done": False,
          "nudge_t0": None, "nudge_bytes": 0, "nudge_done": False, "b_start": None,
          "a_start": None,
          "usp_times": []}

    headers = {"Authorization": f"Token {token}"} if token else None
    async with websockets.connect(url, max_size=None, additional_headers=headers) as ws:
        await ws.send(json.dumps(settings(think)))

        async def sender():
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

        async def receiver():
            async for msg in ws:
                now = time.monotonic()
                if isinstance(msg, bytes):
                    if st["phase"] == "question" and not st["q_done"]:
                        st["q_t0"] = st["q_t0"] or now
                        st["q_bytes"] += len(msg)
                    elif (st["phase"] == "answering" and st["nudge_t0"] is not None
                          and not st["nudge_done"]):
                        st["nudge_bytes"] += len(msg)
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if trace and st["phase"] == "answering" and t != "LatencyReport":
                    rel = now - st["a_start"]
                    brief = {k: v for k, v in m.items()
                             if k in ("event", "role", "content", "intent", "spoke",
                                      "trigger", "text", "eot_fires", "bc_fires")}
                    print(f"    [{rel:6.2f}s from A] {t} {json.dumps(brief)[:150]}",
                          flush=True)
                if t == "AgentAudioDone":
                    if st["phase"] == "greeting":
                        st["phase"] = "greeting_done"
                    elif st["phase"] == "question":
                        st["q_done"] = True
                    elif st["phase"] == "answering" and st["nudge_t0"] is not None:
                        st["nudge_done"] = True   # the nudge's own audio is complete
                elif t == "UserStartedSpeaking" and st["phase"] == "answering":
                    st["usp_times"].append(now)
                elif (t == "ConversationalEvent" and m.get("event") == "probe_decision"
                      and st["phase"] == "answering" and m.get("spoke")
                      and m.get("intent", "").lower().replace("_", "") == "askcontinue"
                      and st["nudge_t0"] is None):
                    out["nudged"] = True
                    st["nudge_t0"] = now
                elif (t == "ConversationText" and m.get("role") == "assistant"
                      and st["nudge_t0"] is not None and out["nudge_text"] is None):
                    out["nudge_text"] = m.get("content")
                elif t in ("Error", "Warning"):
                    out["note"] += f"{t}:{m.get('description') or m.get('code')} "

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            async def wait_for(cond, timeout):
                end = time.monotonic() + timeout
                while not cond():
                    if time.monotonic() > end:
                        return False
                    await asyncio.sleep(0.01)
                return True

            if not await wait_for(lambda: st["phase"] == "greeting_done", 20):
                out["note"] += "no greeting "
                return out
            await asyncio.sleep(1.0)
            st["phase"] = "question"
            await ws.send(json.dumps({"type": "InjectAgentMessage", "content": QUESTION}))
            if not await wait_for(lambda: st["q_done"], 20):
                out["note"] += "question never finished "
                return out
            q_end = st["q_t0"] + st["q_bytes"] / OUT_BYTES_PER_S
            await asyncio.sleep(max(0.0, q_end - time.monotonic()) + 0.5)

            st["phase"] = "answering"
            st["a_start"] = time.monotonic()
            pending.extend(part_a)
            await asyncio.sleep(len(part_a) / (INPUT_RATE * 2) + pause_s)
            st["b_start"] = time.monotonic()
            pending.extend(part_b)
            b_end = st["b_start"] + len(part_b) / (INPUT_RATE * 2)
            await asyncio.sleep(len(part_b) / (INPUT_RATE * 2) + listen_s)
        finally:
            for t in tasks:
                t.cancel()

    if st["nudge_t0"] is not None:
        # Audible from its first byte for bytes/48000 s, bytes counted up to the
        # nudge's own AgentAudioDone. A barge-in also ends the count, since stem
        # sends AgentAudioDone when it interrupts TTS.
        n_end = st["nudge_t0"] + st["nudge_bytes"] / OUT_BYTES_PER_S
        out["nudge_audio_s"] = round(st["nudge_bytes"] / OUT_BYTES_PER_S, 2)
        out["overlap_s"] = round(max(0.0, min(n_end, b_end) - max(st["nudge_t0"],
                                                                   st["b_start"])), 2)
        out["barge_in_during_nudge"] = any(st["nudge_t0"] < u < n_end
                                           for u in st["usp_times"])
        out["nudge_started_rel_b"] = round(st["nudge_t0"] - st["b_start"], 2)
    return out


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8092/v1/agent/converse")
    p.add_argument("--token", default="", help="Deepgram API key (hosted only)")
    p.add_argument("--think", choices=["local", "hosted"], default="local")
    p.add_argument("--part-a", default="partA.wav")
    p.add_argument("--part-b", default="partB.wav")
    p.add_argument("--pauses", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--trace", action="store_true",
                   help="print every server event during the answer")
    args = p.parse_args()

    a, b = read_wav(args.part_a), read_wav(args.part_b)
    rows = []
    for pause in args.pauses:
        for rep in range(args.repeats):
            r = await trial(args.url, a, b, pause, args.token, args.think,
                            trace=args.trace)
            r["rep"] = rep
            rows.append(r)
            print(f"pause={pause:.1f}s rep={rep} nudged={r['nudged']!s:5} "
                  f"start_vs_B={r.get('nudge_started_rel_b', '-')}s "
                  f"audio={r['nudge_audio_s']}s overlap={r['overlap_s']}s "
                  f"barge_in={r['barge_in_during_nudge']!s:5} text={r['nudge_text']!r} "
                  f"{r['note']}", flush=True)
    print("\nsummary: nudged / talked over (overlap > 0, no barge-in) per pause")
    for pause in args.pauses:
        sel = [r for r in rows if r["pause"] == pause]
        nudged = [r for r in sel if r["nudged"]]
        over = [r for r in nudged if r["overlap_s"] > 0 and not r["barge_in_during_nudge"]]
        print(f"  pause={pause:.1f}s  nudged {len(nudged)}/{len(sel)}  "
              f"talked-over {len(over)}/{len(nudged) or 1}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
