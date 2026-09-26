#!/usr/bin/env python3
"""Pizza ordering demo: talk to the Voice Agent, watch the order build in a browser.

The agent takes an order off a fixed menu using client-side FUNCTION CALLS
(add_item, update_item, remove_item, set_customer, submit_order). This script owns
the order: stem sends each call as a `FunctionCallRequest`, the script applies it,
answers with a `FunctionCallResponse` carrying the updated order and total, and
pushes the new state to a local web page. Prices and totals the agent reads out
therefore come from this script, never from the LLM's arithmetic.

Audio runs in the terminal exactly as in agent_live.py (same mic/speaker loop,
same flags: --push-to-talk, --conversational, recording). The browser only shows
the menu, the current order, the function-call log, and the transcript.

    uv run pizza_demo.py                      # opens http://127.0.0.1:8765
    uv run pizza_demo.py --conversational --push-to-talk

Every function that changes the order sets `defer_until_eot`, so stem waits for a
confirmed end of turn before dispatching it. Without that, an eager end-of-turn
draft (the caller pausing mid-sentence) could add the same pizza twice.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import webbrowser
from http import HTTPStatus

from websockets.asyncio.server import serve

from agent_converse import build_settings
from agent_live import build_parser, run
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
# Browser side: one static page plus a /events WebSocket that receives the state.
# ------------------------------------------------------------------------------

class Hub:
    def __init__(self, order: Order) -> None:
        self.order = order
        self.clients: set = set()
        self.log: list[dict] = []        # transcript + function calls, oldest first
        self.status = "connecting"

    def state(self) -> str:
        return json.dumps({"type": "state", "menu": MENU, "order": self.order.snapshot(),
                           "log": self.log[-200:], "status": self.status})

    async def push(self) -> None:
        msg = self.state()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                self.clients.discard(ws)

    async def handler(self, ws) -> None:
        self.clients.add(ws)
        try:
            await ws.send(self.state())
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    def process_request(self, connection, request):
        if request.path == "/events":
            return None  # continue the WebSocket handshake
        if request.path in ("/", "/index.html"):
            resp = connection.respond(HTTPStatus.OK, PAGE)
            del resp.headers["Content-Type"]
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def on_message(self, ws, m: dict) -> None:
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
        elif t in ("UserStartedSpeaking",):
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


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pizza Order Demo</title>
<style>
:root{--bg:#faf7f2;--panel:#fff;--ink:#1f1b16;--muted:#7a6f63;--line:#ece4d8;
--accent:#c8412b;--accent-soft:#fbe3dc;--ok:#2f7d4f;--ok-soft:#dff1e6;--mono:ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#16130f;--panel:#201c17;--ink:#f3ede4;
--muted:#a89c8d;--line:#332d25;--accent:#ff7a5c;--accent-soft:#3a221b;--ok:#6fcf97;--ok-soft:#1d3326}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0}
#status{margin-left:auto;font-size:13px;padding:3px 10px;border-radius:99px;background:var(--line);color:var(--muted)}
#status.agent{background:var(--accent-soft);color:var(--accent)}
#status.caller{background:var(--ok-soft);color:var(--ok)}
main{display:grid;grid-template-columns:1fr 1.1fr 1fr;gap:16px;padding:16px 20px;max-width:1400px;margin:0 auto}
@media (max-width:980px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}
h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
h3{font-size:14px;margin:14px 0 6px}
.row{display:flex;justify-content:space-between;gap:10px;padding:3px 0}
.price{font-variant-numeric:tabular-nums;color:var(--muted);white-space:nowrap}
.small{font-size:13px;color:var(--muted)}
#lines .row{padding:8px 10px;border-radius:8px;margin:0 -10px}
#lines .n{color:var(--muted);width:1.6em;flex:none}
#lines .item{flex:1}
.flash{animation:flash 1.6s ease-out}
@keyframes flash{from{background:var(--accent-soft)}to{background:transparent}}
.empty{color:var(--muted);font-style:italic;padding:8px 0}
.totals{border-top:1px solid var(--line);margin-top:10px;padding-top:8px}
.total{font-weight:700;font-size:18px}
#submitted{display:none;margin-top:12px;padding:12px;border-radius:10px;background:var(--ok-soft);color:var(--ok);font-weight:600}
#customer{margin-top:10px}
#log{max-height:70vh;overflow-y:auto;display:flex;flex-direction:column;gap:6px}
.msg{padding:7px 10px;border-radius:10px;max-width:92%}
.msg.user{align-self:flex-end;background:var(--ok-soft)}
.msg.assistant{align-self:flex-start;background:var(--line)}
.fn{font:12px/1.4 var(--mono);padding:6px 9px;border-left:3px solid var(--accent);background:var(--bg);border-radius:4px;overflow-wrap:anywhere}
.fn.err{border-color:#d9a400}
.fn b{color:var(--accent)}
</style></head><body>
<header><h1>🍕 Slice of Heaven Pizza</h1><span class="small">voice order demo</span>
<span id="status">connecting</span></header>
<main>
<section><h2>Menu</h2><div id="menu"></div></section>
<section><h2>Current order</h2><div id="lines"></div>
<div class="totals" id="totals"></div><div id="customer" class="small"></div>
<div id="submitted"></div></section>
<section><h2>Conversation &amp; function calls</h2><div id="log"></div></section>
</main>
<script>
const $ = id => document.getElementById(id);
const money = v => "$" + Number(v).toFixed(2);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let menuDrawn = false, lastSig = "";
function drawMenu(m){
  let h = "<h3>Pizzas <span class='small'>S / M / L</span></h3>";
  for (const [n,p] of Object.entries(m.pizzas))
    h += `<div class="row"><span>${esc(n)}</span><span class="price">${p.map(money).join(" / ")}</span></div>`;
  h += `<div class="small">Extra toppings ${m.topping_price.map(money).join(" / ")}: ${m.toppings.map(esc).join(", ")}. Stuffed crust +$2.00.</div>`;
  for (const sec of ["sides","drinks","desserts"]) {
    h += `<h3>${sec[0].toUpperCase()+sec.slice(1)}</h3>`;
    for (const [n,p] of Object.entries(m[sec]))
      h += `<div class="row"><span>${esc(n)}</span><span class="price">${money(p)}</span></div>`;
  }
  $("menu").innerHTML = h; menuDrawn = true;
}
function drawOrder(o){
  const sig = JSON.stringify(o.lines);
  $("lines").innerHTML = o.lines.length ? o.lines.map(l =>
    `<div class="row" data-line="${l.line}"><span class="n">${l.line}.</span><span class="item">${esc(l.item)}</span><span class="price">${money(l.price)}</span></div>`
  ).join("") : '<div class="empty">Nothing yet — start talking.</div>';
  if (sig !== lastSig && o.changed_line) {
    const el = document.querySelector(`[data-line="${o.changed_line}"]`);
    if (el) el.classList.add("flash");
  }
  lastSig = sig;
  let t = `<div class="row"><span>Subtotal</span><span class="price">${money(o.subtotal)}</span></div>`;
  if (o.delivery_fee) t += `<div class="row"><span>Delivery</span><span class="price">${money(o.delivery_fee)}</span></div>`;
  t += `<div class="row"><span>Tax</span><span class="price">${money(o.tax)}</span></div>`;
  t += `<div class="row total"><span>Total</span><span>${money(o.total)}</span></div>`;
  $("totals").innerHTML = t;
  const c = o.customer;
  $("customer").innerHTML = c ? `${esc(c.name)} · ${esc(c.fulfillment)}${c.address ? " · " + esc(c.address) : ""}${c.phone ? " · " + esc(c.phone) : ""}` : "";
  const s = $("submitted");
  if (o.submitted) { s.style.display = "block"; s.textContent = `✓ Order #${o.submitted.number} placed — ready in about ${o.submitted.ready_in_minutes} minutes`; }
  else s.style.display = "none";
}
function drawLog(log){
  const el = $("log"), atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  el.innerHTML = log.map(e => e.kind === "fn"
    ? `<div class="fn ${e.ok ? "" : "err"}"><b>${esc(e.name)}</b>(${esc(e.args)})<br>→ ${esc(e.result)}</div>`
    : `<div class="msg ${esc(e.role)}">${esc(e.content)}</div>`).join("");
  if (atBottom) el.scrollTop = el.scrollHeight;
}
function drawStatus(s){
  const el = $("status"); el.textContent = s;
  el.className = s.startsWith("agent") ? "agent" : s.startsWith("caller") ? "caller" : "";
}
function connect(){
  const ws = new WebSocket(`ws://${location.host}/events`);
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (!menuDrawn) drawMenu(m.menu);
    drawOrder(m.order); drawLog(m.log); drawStatus(m.status);
  };
  ws.onclose = () => { drawStatus("disconnected — retrying"); setTimeout(connect, 1500); };
}
connect();
</script></body></html>
"""


async def amain(args) -> int:
    order = Order()
    hub = Hub(order)
    settings = build_settings(args)
    settings["agent"]["think"]["functions"] = FUNCTIONS
    async with serve(hub.handler, "127.0.0.1", args.web_port,
                     process_request=hub.process_request):
        page = f"http://127.0.0.1:{args.web_port}/"
        print(f"order board: {page}")
        if args.open_browser:
            webbrowser.open(page)
        rc = await run(args, settings=settings, on_message=hub.on_message)
        if order.lines:
            print("\nfinal order:\n" + json.dumps(order.snapshot(), indent=2))
        return rc


def main() -> int:
    p = build_parser(__doc__)
    p.add_argument("--web-port", type=int, default=8765,
                   help="local port for the order board (default 8765)")
    p.add_argument("--no-browser", dest="open_browser", action="store_false",
                   help="don't open the order board automatically")
    args = p.parse_args()
    args.prompt = args.prompt or PROMPT
    args.greeting = args.greeting or GREETING
    apply_scenario(args)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nhung up.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
