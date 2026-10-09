#!/usr/bin/env python3
"""Pizza ordering demo: talk to the Voice Agent, watch the order build in a browser.

The agent takes an order off a fixed menu using client-side FUNCTION CALLS
(add_item, update_item, remove_item, set_customer, submit_order). This script owns
the order: stem sends each call as a `FunctionCallRequest`, the script applies it,
answers with a `FunctionCallResponse` carrying the updated order and total, and
pushes the new state to a local web page. Prices and totals the agent reads out
therefore come from this script, never from the LLM's arithmetic.

Audio runs IN THE BROWSER by default. Click "Start call" and the page captures the
mic with getUserMedia({audio: {echoCancellation: true, ...}}), so the browser's
built-in WebRTC echo canceller removes the agent's voice from the mic and laptop
speakers work without push-to-talk. The page streams 16 kHz PCM to this script
over its WebSocket; the script relays it to stem and relays the agent's 24 kHz
audio back. Agent audio is played through a local WebRTC loopback by default
(Chrome's canceller has historically only subtracted audio played via WebRTC);
open the page with ?playback=direct to compare plain Web Audio playback.

    uv run pizza_demo.py --host <public-ip>            # opens http://127.0.0.1:8765
    uv run pizza_demo.py --host <public-ip> --conversational
    uv run pizza_demo.py --host <public-ip> --terminal-audio   # old sounddevice path

The page must be opened as http://127.0.0.1 or http://localhost: browsers only
allow microphone access from a secure context, and a plain-http LAN address isn't
one. --terminal-audio keeps the agent_live.py mic/speaker loop (with its
--push-to-talk and escalation demo) and uses the page as a display only.
Recording (on by default) is written by this script in both modes.

Every function that changes the order sets `defer_until_eot`, so stem waits for a
confirmed end of turn before dispatching it. Without that, an eager end-of-turn
draft (the caller pausing mid-sentence) could add the same pizza twice.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
import webbrowser
from http import HTTPStatus
from pathlib import Path

import websockets
from websockets.asyncio.server import serve

from agent_converse import OUTPUT_RATE, build_settings
from agent_live import build_parser, run, save_recording
from scenarios import apply_scenario

SIZES = ("small", "medium", "large")
CRUSTS = {"hand-tossed": 0.0, "thin": 0.0, "stuffed": 2.0}

MENU = {
    "pizzas": {
        "Margherita": (11.0, 14.0, 17.0),
        "Pepperoni": (12.0, 15.0, 18.0),
        "Hawaiian": (13.0, 16.0, 19.0),
        "Meat Lovers": (14.0, 17.0, 20.0),
        "Veggie Supreme": (13.0, 16.0, 19.0),
        "BBQ Chicken": (14.0, 17.0, 20.0),
    },
    # Price per extra topping, by size.
    "topping_price": (1.5, 2.0, 2.5),
    "toppings": ["mushrooms", "onions", "green peppers", "black olives", "jalapenos",
                 "pineapple", "sausage", "bacon", "extra cheese", "pepperoni"],
    "sides": {"Garlic Knots": 6.0, "Cheesy Bread": 7.0, "Caesar Salad": 8.0,
              "Buffalo Wings": 12.0},
    "drinks": {"Coke": 2.5, "Diet Coke": 2.5, "Sprite": 2.5, "Lemonade": 2.5},
    "desserts": {"Chocolate Chip Cookie": 3.0, "Cinnamon Sticks": 6.0},
}
DELIVERY_FEE = 3.99
TAX_RATE = 0.08

SIMPLE_ITEMS = {**MENU["sides"], **MENU["drinks"], **MENU["desserts"]}
ALL_ITEMS = list(MENU["pizzas"]) + list(SIMPLE_ITEMS)


def _menu_text() -> str:
    lines = ["Pizzas (small / medium / large):"]
    for name, (s, m, l) in MENU["pizzas"].items():
        lines.append(f"- {name}: ${s:.2f} / ${m:.2f} / ${l:.2f}")
    s, m, l = MENU["topping_price"]
    lines.append(f"Extra toppings (${s:.2f} / ${m:.2f} / ${l:.2f} each): "
                 + ", ".join(MENU["toppings"]))
    lines.append("Crusts: hand-tossed (default), thin, stuffed (+$2.00)")
    for section in ("sides", "drinks", "desserts"):
        lines.append(section.capitalize() + ": " + ", ".join(
            f"{n} ${p:.2f}" for n, p in MENU[section].items()))
    lines.append(f"Delivery fee ${DELIVERY_FEE:.2f}. Tax {TAX_RATE:.0%}.")
    return "\n".join(lines)


PROMPT = f"""You are Tony, the friendly phone order-taker at Slice of Heaven Pizza, \
a fictional pizzeria. You take orders off this menu and nothing else:

{_menu_text()}

How to work:
- You are on a phone call: one or two short sentences per reply, no lists, no \
markdown, prices said out loud in words. Ask one question at a time.
- Change the order ONLY through the functions. The moment the caller names an item \
with enough detail, call add_item in that same turn, before you say anything about \
it. Never say you added, changed or removed something unless you called the \
function in this turn and it returned ok.
- Every pizza needs a size; ask if they didn't say one. Crust is hand-tossed unless \
they ask otherwise, so don't ask about crust. Pizzas that differ in any way (size, \
crust, toppings) are separate add_item calls: "two large pepperonis, one with \
mushrooms" is one plain large pepperoni plus one large pepperoni with mushrooms.
- Use update_item or remove_item, with the line number from the latest order, when \
they change their mind.
- Never do arithmetic yourself. Every function result contains the current order \
and total; quote those numbers.
- If they ask for something not on the menu, say so and suggest the closest item.
- Before finishing, get their name and whether it's pickup or delivery (and the \
address for delivery) with set_customer, read the order back once, and only call \
submit_order after they confirm. Then tell them the order number and ready time \
from the result."""

GREETING = "Thanks for calling Slice of Heaven Pizza, this is Tony. What can I get started for you?"

_LINE = {"type": "integer", "description": "Line number from the current order (1-based)"}
_SIZE = {"type": "string", "enum": list(SIZES)}
_CRUST = {"type": "string", "enum": list(CRUSTS)}
_TOPPINGS = {"type": "array", "items": {"type": "string", "enum": MENU["toppings"]},
             "description": "Extra toppings added on top of the pizza's recipe"}

FUNCTIONS = [
    {"name": "add_item",
     "description": "Add a menu item to the order. Pizzas need a size.",
     "parameters": {"type": "object", "properties": {
         "item": {"type": "string", "enum": ALL_ITEMS},
         "quantity": {"type": "integer", "minimum": 1, "default": 1},
         "size": _SIZE, "crust": _CRUST, "extra_toppings": _TOPPINGS,
         "notes": {"type": "string", "description": "e.g. well done, light sauce"},
     }, "required": ["item"]}},
    {"name": "update_item",
     "description": "Change quantity, size, crust, toppings or notes on an order line. "
                    "extra_toppings replaces the line's topping list.",
     "parameters": {"type": "object", "properties": {
         "line": _LINE, "quantity": {"type": "integer", "minimum": 1},
         "size": _SIZE, "crust": _CRUST, "extra_toppings": _TOPPINGS,
         "notes": {"type": "string"},
     }, "required": ["line"]}},
    {"name": "remove_item", "description": "Remove a line from the order.",
     "parameters": {"type": "object", "properties": {"line": _LINE},
                    "required": ["line"]}},
    {"name": "set_customer",
     "description": "Record the caller's name and pickup or delivery.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"},
         "fulfillment": {"type": "string", "enum": ["pickup", "delivery"]},
         "address": {"type": "string", "description": "Required for delivery"},
         "phone": {"type": "string"},
     }, "required": ["name", "fulfillment"]}},
    {"name": "submit_order",
     "description": "Place the order. Only after the caller confirms the read-back.",
     "parameters": {"type": "object", "properties": {}}},
]
for _f in FUNCTIONS:
    _f["defer_until_eot"] = True


def _match(name: str, options) -> str | None:
    """Case- and punctuation-insensitive lookup; tolerates 'pepperoni pizza'."""
    def norm(x: str) -> str:
        return "".join(c for c in x.lower().replace("pizza", "") if c.isalnum())
    want = norm(name or "")
    for opt in options:
        if norm(opt) == want:
            return opt
    return None


class Order:
    def __init__(self) -> None:
        self.lines: list[dict] = []
        self.customer: dict = {}
        self.submitted: dict | None = None
        self.changed_line: int | None = None

    # --- pricing ---------------------------------------------------------------
    @staticmethod
    def unit_price(line: dict) -> float:
        if line["item"] in MENU["pizzas"]:
            i = SIZES.index(line["size"])
            return (MENU["pizzas"][line["item"]][i] + CRUSTS[line["crust"]]
                    + MENU["topping_price"][i] * len(line["extra_toppings"]))
        return SIMPLE_ITEMS[line["item"]]

    def totals(self) -> dict:
        subtotal = sum(self.unit_price(l) * l["quantity"] for l in self.lines)
        fee = DELIVERY_FEE if self.customer.get("fulfillment") == "delivery" else 0.0
        tax = round(subtotal * TAX_RATE, 2)
        return {"subtotal": round(subtotal, 2), "delivery_fee": fee, "tax": tax,
                "total": round(subtotal + fee + tax, 2)}

    @staticmethod
    def describe(line: dict) -> str:
        if line["item"] in MENU["pizzas"]:
            d = f"{line['quantity']} x {line['size']} {line['item']}"
            if line["crust"] != "hand-tossed":
                d += f", {line['crust']} crust"
            if line["extra_toppings"]:
                d += ", add " + ", ".join(line["extra_toppings"])
        else:
            d = f"{line['quantity']} x {line['item']}"
        if line.get("notes"):
            d += f" ({line['notes']})"
        return d

    def summary(self) -> dict:
        return {
            "lines": [{"line": i + 1, "item": self.describe(l),
                       "price": round(self.unit_price(l) * l["quantity"], 2)}
                      for i, l in enumerate(self.lines)],
            **self.totals(),
            "customer": self.customer or None,
        }

    def snapshot(self) -> dict:
        s = self.summary()
        s["changed_line"] = self.changed_line
        s["submitted"] = self.submitted
        return s

    # --- functions -------------------------------------------------------------
    def _line(self, args: dict) -> tuple[int, dict]:
        try:
            n = int(args.get("line"))
        except (TypeError, ValueError):
            raise ValueError("line must be a line number from the current order")
        if not 1 <= n <= len(self.lines):
            raise ValueError(f"there is no line {n}; the order has {len(self.lines)} lines")
        return n - 1, self.lines[n - 1]

    def _apply_pizza_fields(self, line: dict, args: dict) -> None:
        if args.get("size") is not None:
            if args["size"] not in SIZES:
                raise ValueError(f"size must be one of {', '.join(SIZES)}")
            line["size"] = args["size"]
        if args.get("crust") is not None:
            crust = _match(args["crust"], CRUSTS)
            if not crust:
                raise ValueError(f"crust must be one of {', '.join(CRUSTS)}")
            line["crust"] = crust
        if args.get("extra_toppings") is not None:
            tops = []
            for t in args["extra_toppings"]:
                m = _match(t, MENU["toppings"])
                if not m:
                    raise ValueError(f"{t!r} is not a topping we offer")
                tops.append(m)
            line["extra_toppings"] = tops

    def add_item(self, args: dict) -> str:
        item = _match(args.get("item", ""), ALL_ITEMS)
        if not item:
            raise ValueError(f"{args.get('item')!r} is not on the menu")
        line = {"item": item, "quantity": max(1, int(args.get("quantity") or 1)),
                "notes": args.get("notes") or ""}
        if item in MENU["pizzas"]:
            if not args.get("size"):
                raise ValueError("ask the caller what size they want, then call again")
            line.update(size="", crust="hand-tossed", extra_toppings=[])
            self._apply_pizza_fields(line, args)
        self.lines.append(line)
        self.changed_line = len(self.lines)
        return f"added line {len(self.lines)}: {self.describe(line)}"

    def update_item(self, args: dict) -> str:
        i, line = self._line(args)
        if args.get("quantity") is not None:
            line["quantity"] = max(1, int(args["quantity"]))
        if args.get("notes") is not None:
            line["notes"] = args["notes"]
        if line["item"] in MENU["pizzas"]:
            self._apply_pizza_fields(line, args)
        self.changed_line = i + 1
        return f"updated line {i + 1}: {self.describe(line)}"

    def remove_item(self, args: dict) -> str:
        i, line = self._line(args)
        self.lines.pop(i)
        self.changed_line = None
        return f"removed {self.describe(line)}; remaining lines are renumbered"

    def set_customer(self, args: dict) -> str:
        if args.get("fulfillment") not in ("pickup", "delivery"):
            raise ValueError("fulfillment must be pickup or delivery")
        if args["fulfillment"] == "delivery" and not args.get("address"):
            raise ValueError("ask for the delivery address, then call again")
        self.customer = {k: args[k] for k in ("name", "fulfillment", "address", "phone")
                         if args.get(k)}
        return f"customer set: {self.customer.get('name')}, {args['fulfillment']}"

    def submit_order(self, args: dict) -> str:
        if not self.lines:
            raise ValueError("the order is empty")
        if not self.customer:
            raise ValueError("call set_customer first")
        if self.submitted:
            return f"already submitted as order {self.submitted['number']}"
        minutes = 35 if self.customer["fulfillment"] == "delivery" else 20
        self.submitted = {"number": 1000 + int(time.time()) % 9000,
                          "ready_in_minutes": minutes}
        return (f"order {self.submitted['number']} placed; "
                f"{self.customer['fulfillment']} in about {minutes} minutes")

    def call(self, name: str, raw_args: str) -> dict:
        fn = getattr(self, name, None) if name in {f["name"] for f in FUNCTIONS} else None
        try:
            args = json.loads(raw_args or "{}") or {}
            if fn is None:
                raise ValueError(f"unknown function {name}")
            if self.submitted and name != "submit_order":
                raise ValueError("the order was already submitted and can't be changed")
            result = {"ok": True, "result": fn(args)}
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            result = {"ok": False, "error": str(e)}
        result["order"] = self.summary()
        if self.submitted:
            result["submitted"] = self.submitted
        return result


# ------------------------------------------------------------------------------
# Browser side: pizza_board.html plus one /events WebSocket per tab. Text frames
# carry board state and call control; binary frames carry audio (browser -> here:
# 16 kHz int16 mic; here -> browser: 24 kHz int16 agent audio).
# ------------------------------------------------------------------------------

PAGE_TEMPLATE = (Path(__file__).with_name("pizza_board.html")).read_text()


class Call:
    """One browser-audio call: a stem connection bridged to one browser tab."""

    def __init__(self, hub: "Hub", browser) -> None:
        self.hub = hub
        self.args = hub.args
        self.browser = browser
        self.stem = None
        self.task: asyncio.Task | None = None
        self.t0 = 0.0
        self.rec_user = bytearray()
        self.rec_agent = bytearray()

    async def _to_browser(self, data) -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            await self.browser.send(data)

    # Recording. The caller track is the mic stream as it arrived (continuous, so
    # it keeps its own clock). Agent audio is placed on the wall clock at arrival,
    # which is when the browser starts playing it, and cut back on barge-in, when
    # the browser drops whatever it had queued.
    def _place_agent(self, pcm: bytes) -> None:
        now = int((time.monotonic() - self.t0) * OUTPUT_RATE) * 2
        if len(self.rec_agent) < now:
            self.rec_agent.extend(bytes(now - len(self.rec_agent)))
        self.rec_agent.extend(pcm)

    def _cut_agent(self) -> None:
        now = int((time.monotonic() - self.t0) * OUTPUT_RATE) * 2
        del self.rec_agent[now:]

    async def feed(self, pcm: bytes) -> None:
        if self.stem is None:
            return  # mic frames before stem is connected are dropped
        if self.args.record:
            self.rec_user.extend(pcm)
        with contextlib.suppress(websockets.ConnectionClosed):
            await self.stem.send(pcm)

    async def run(self) -> None:
        url = f"ws://{self.args.host}:{self.args.port}/v1/agent/converse"
        settings = build_settings(self.args)
        settings["agent"]["think"]["functions"] = FUNCTIONS
        print(f"call: connecting {url}", flush=True)
        try:
            async with websockets.connect(url, max_size=None, open_timeout=10) as stem:
                await stem.send(json.dumps(settings))
                self.t0 = time.monotonic()
                self.stem = stem
                await self._to_browser(json.dumps({"type": "call", "state": "connected"}))
                async for msg in stem:
                    if isinstance(msg, bytes):
                        if self.args.record:
                            self._place_agent(msg)
                        await self._to_browser(msg)
                        continue
                    m = json.loads(msg)
                    t = m.get("type")
                    if t == "UserStartedSpeaking":
                        # Barge-in: the browser drops its queued agent audio.
                        self._cut_agent()
                        await self._to_browser('{"type":"flush"}')
                    elif t == "AgentAudioDone":
                        await self._to_browser('{"type":"reply_done"}')
                    elif t == "ConversationText":
                        print(f"  [{m.get('role')}] {m.get('content', '')}", flush=True)
                    elif t in ("Error", "Warning"):
                        detail = m.get("description") or m.get("message") or json.dumps(m)
                        print(f"  [{t}] {detail}", flush=True)
                        await self._to_browser(json.dumps(
                            {"type": "notice", "text": f"{t}: {detail}"}))
                    elif t == "ConversationalEvent" and m.get("event") in (
                            "agent_backchannel", "user_backchannel"):
                        print(f"  [{m['event']}] {m.get('text', '')}", flush=True)
                    await self.hub.on_message(stem, m)
        except (OSError, TimeoutError, websockets.InvalidHandshake) as e:
            text = (f"Could not reach stem at {url} ({type(e).__name__}). Is the "
                    "container up, and does its security group admit your IP on "
                    f"{self.args.port}?")
            print(f"call: {text}", file=sys.stderr, flush=True)
            await self._to_browser(json.dumps({"type": "notice", "text": text}))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.stem = None
            print("call: ended", flush=True)
            if self.args.record and (self.rec_user or self.rec_agent):
                try:
                    save_recording(Path(self.args.record_dir),
                                   time.strftime("%Y%m%d-%H%M%S") + "_pizza",
                                   bytes(self.rec_user), bytes(self.rec_agent))
                except OSError as e:
                    print(f"  [record] could not save recording: {e}", file=sys.stderr)
            if self.hub.call is self:
                self.hub.call = None
            self.hub.status = "idle"
            await self._to_browser(json.dumps({"type": "call", "state": "ended"}))
            await self.hub.push()


class Hub:
    def __init__(self, args, audio: str) -> None:
        self.args = args
        self.audio = audio               # "browser" or "terminal"
        self.order = Order()
        self.clients: set = set()
        self.log: list[dict] = []        # transcript + function calls, oldest first
        self.status = "idle" if audio == "browser" else "connecting"
        self.call: Call | None = None

    def state(self) -> str:
        return json.dumps({"type": "state", "menu": MENU, "order": self.order.snapshot(),
                           "log": self.log[-200:], "status": self.status,
                           "audio": self.audio, "call": self.call is not None})

    async def push(self) -> None:
        msg = self.state()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except websockets.ConnectionClosed:
                self.clients.discard(ws)

    async def start_call(self, browser) -> None:
        if self.audio != "browser":
            await browser.send(json.dumps({"type": "notice", "text":
                "Started with --terminal-audio: talk in the terminal, not here."}))
            return
        if self.call is not None:
            await browser.send(json.dumps({"type": "notice", "text":
                "A call is already running in another tab."}))
            await browser.send(json.dumps({"type": "call", "state": "ended"}))
            return
        self.order = Order()             # every call is a fresh order
        self.log = []
        self.status = "connecting"
        self.call = Call(self, browser)
        self.call.task = asyncio.create_task(self.call.run())
        await self.push()

    async def stop_call(self, browser) -> None:
        call = self.call
        if call is not None and call.browser is browser and call.task:
            call.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call.task

    async def handler(self, ws) -> None:
        self.clients.add(ws)
        try:
            await ws.send(self.state())
            async for msg in ws:
                if isinstance(msg, bytes):
                    if self.call is not None and self.call.browser is ws:
                        await self.call.feed(msg)
                    continue
                try:
                    m = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if m.get("type") == "start":
                    await self.start_call(ws)
                elif m.get("type") == "stop":
                    await self.stop_call(ws)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            await self.stop_call(ws)     # closing the tab hangs up its call

    def process_request(self, connection, request):
        if request.path == "/events":
            return None  # continue the WebSocket handshake
        if request.path.split("?")[0] in ("/", "/index.html"):
            page = PAGE_TEMPLATE.replace("__CONFIG__", json.dumps(
                {"playbackBufferMs": self.args.playback_buffer_ms}))
            resp = connection.respond(HTTPStatus.OK, page)
            del resp.headers["Content-Type"]
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp
        if request.path == "/favicon.ico":
            return connection.respond(HTTPStatus.NO_CONTENT, "")
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def on_message(self, ws, m: dict) -> None:
        """Server message hook: runs function calls and tracks board state. `ws` is
        the stem connection (FunctionCallResponse goes back on it)."""
        t = m.get("type")
        if t == "FunctionCallRequest":
            for fc in m.get("functions", []):
                if not fc.get("client_side", True):
                    continue
                result = self.order.call(fc["name"], fc.get("arguments", ""))
                print(f"  [fn] {fc['name']}({fc.get('arguments', '')}) -> "
                      f"{result.get('result') or result.get('error')}  "
                      f"total=${result['order']['total']:.2f}", flush=True)
                self.log.append({"kind": "fn", "name": fc["name"],
                                 "args": fc.get("arguments", ""),
                                 "ok": result["ok"],
                                 "result": result.get("result") or result.get("error")})
                await ws.send(json.dumps({"type": "FunctionCallResponse", "id": fc["id"],
                                          "name": fc["name"],
                                          "content": json.dumps(result)}))
        elif t == "ConversationText":
            self.log.append({"kind": "text", "role": m.get("role"),
                             "content": m.get("content", "")})
        elif t == "UserStartedSpeaking":
            self.status = "caller speaking"
        elif t == "AgentThinking":
            self.status = "thinking"
        elif t == "AgentStartedSpeaking":
            self.status = "agent speaking"
        elif t in ("AgentAudioDone", "SettingsApplied", "Welcome"):
            self.status = "listening"
        else:
            return
        await self.push()


async def amain(args) -> int:
    hub = Hub(args, "terminal" if args.terminal_audio else "browser")
    async with serve(hub.handler, "127.0.0.1", args.web_port,
                     process_request=hub.process_request, max_size=None):
        page = f"http://127.0.0.1:{args.web_port}/"
        print(f"order board: {page}")
        if args.open_browser:
            webbrowser.open(page)
        if args.terminal_audio:
            rc = await run(args, settings=_terminal_settings(args),
                           on_message=hub.on_message)
            if hub.order.lines:
                print("\nfinal order:\n" + json.dumps(hub.order.snapshot(), indent=2))
            return rc
        print("click 'Start call' on the page (Ctrl-C here to quit)")
        await asyncio.Future()           # serve until Ctrl-C
    return 0


def _terminal_settings(args) -> dict:
    settings = build_settings(args)
    settings["agent"]["think"]["functions"] = FUNCTIONS
    return settings


def main() -> int:
    p = build_parser(__doc__)
    p.add_argument("--web-port", type=int, default=8765,
                   help="local port for the order board (default 8765)")
    p.add_argument("--no-browser", dest="open_browser", action="store_false",
                   help="don't open the order board automatically")
    p.add_argument("--terminal-audio", action="store_true",
                   help="use the terminal mic/speaker (agent_live.py's loop) instead "
                        "of the browser; the page becomes a display only")
    args = p.parse_args()
    args.prompt = args.prompt or PROMPT
    args.greeting = args.greeting or GREETING
    args.demo_function = False   # build_settings reads it; the pizza FUNCTIONS replace the demo one
    apply_scenario(args)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nbye.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
