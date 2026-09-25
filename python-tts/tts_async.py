#!/usr/bin/env python3
"""
Deepgram SageMaker Text-to-Speech Client - Async Inference (Inline Payload)

Synthesizes speech via the SageMaker `InvokeEndpointAsync` API on an endpoint
whose `EndpointConfig` carries an `AsyncInferenceConfig`. The request payload
(`{"text": "..."}`) is sent **inline** in the API call through the `Body`
parameter, so no S3 upload is needed on the way in:
https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ai-async-inference-now-supports-inline-request-payloads/

Output is unchanged from classic async inference: the synthesized audio lands
at the `OutputLocation` under the EndpointConfig's `S3OutputPath` (or an error
under `S3FailurePath`), and this script polls S3 for it.

Flow
----
1.  Read the text to synthesize (--text, or --text-file with one phrase per
    line; requests cycle through the phrases).
2.  Call `InvokeEndpointAsync` for each request with `Body` set to the JSON
    payload, passing the voice + output settings through `CustomAttributes`
    formatted as `v1/speak?model=<voice>&key=value&...`. The path prefix is
    required — without it the Deepgram container returns 404.
3.  Poll the success + failure S3 locations returned by the call until each
    invocation lands an audio object or an `-error.out`.
4.  Download the audio (optionally save it) and print a per-request summary
    plus an aggregate latency / throughput table.

Limits
------
- Inline `Body` is capped at 128,000 bytes by SageMaker and is mutually
  exclusive with `InputLocation`. Deepgram TTS itself accepts at most 2,000
  characters of text per request, so the inline cap is never the binding one.
- Needs a recent boto3 (older releases reject `Body` on
  `InvokeEndpointAsync` with a ParamValidationError); `uv sync` installs one.

Endpoint prerequisites
----------------------
- The endpoint must be created from an `EndpointConfig` that includes an
  `AsyncInferenceConfig` block with `OutputConfig.S3OutputPath` and
  `S3FailurePath`.
- The SageMaker execution role attached to the endpoint must have
  `s3:PutObject` on the output + failure prefixes. It no longer needs read
  access to an input prefix.

This script only needs `sagemaker:InvokeEndpointAsync` on the endpoint and
`s3:GetObject` on the output/failure keys.
"""

import argparse
import concurrent.futures
import io
import itertools
import json
import logging
import os
import statistics
import sys
import threading
import time
import uuid
import wave
from urllib.parse import quote, urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    ParamValidationError,
    PartialCredentialsError,
)

# Configuration constants
DEFAULT_REGION = "us-east-2"
DEFAULT_VOICE = "aura-2-thalia-en"
DEFAULT_TEXT_FILE = "tts-input.txt"

# SageMaker limits, per AWS:
# https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html
INLINE_BODY_MAX_BYTES = 128_000
ASYNC_INVOKE_MAX_TIMEOUT_S = 3600  # 60 minutes

# Deepgram TTS per-request text limit.
TTS_MAX_CHARS = 2000

DEFAULT_INVOCATION_TIMEOUT_S = 600
DEFAULT_POLL_INTERVAL_S = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_extra(value: str) -> list[tuple[str, str]]:
    """
    Parse `--extra k=v&k2=v2` into a list of (key, value) pairs preserving order.
    """
    if not value:
        return []
    out: list[tuple[str, str]] = []
    for pair in value.split("&"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"ERROR: --extra entry '{pair}' must be in k=v form")
            raise SystemExit(1)
        k, v = pair.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _build_custom_attributes(voice: str, extra_pairs: list[tuple[str, str]]) -> str:
    """Build the `v1/speak?...` CustomAttributes string."""
    params = [f"model={quote(voice)}"]
    for k, v in extra_pairs:
        params.append(f"{quote(k)}={quote(v)}")
    return "v1/speak?" + "&".join(params)


def _build_body(text: str) -> bytes:
    """Encode one inline request payload, enforcing the SageMaker inline cap."""
    body = json.dumps({"text": text}).encode("utf-8")
    if len(body) > INLINE_BODY_MAX_BYTES:
        raise ValueError(
            f"Inline payload is {len(body):,} bytes, over the SageMaker "
            f"{INLINE_BODY_MAX_BYTES:,}-byte inline limit."
        )
    return body


def _load_phrases(args) -> list[str]:
    """Return the phrases to synthesize from --text or --text-file."""
    if args.text:
        return [args.text]
    with open(args.text_file, encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]
    if not phrases:
        raise ValueError(f"{args.text_file} contains no non-empty lines")
    return phrases


def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Split `s3://bucket/key/path` into (bucket, key). Raises on malformed URIs.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _sniff_audio(audio: bytes) -> str:
    """Guess a file extension from the audio's leading magic bytes."""
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "wav"
    if audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if audio[:4] == b"fLaC":
        return "flac"
    if audio[:4] == b"OggS":
        return "ogg"
    return "bin"


def _wav_duration(audio: bytes) -> float | None:
    """Duration in seconds for a WAV payload, or None if not parseable WAV."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AsyncTTSClient:
    """
    Submits TTS jobs with inline payloads to an async-configured SageMaker
    endpoint and collects the resulting S3 audio outputs.

    Each invocation runs on its own thread with its own boto3 clients so
    concurrent submissions do not share connection state. Each in-flight
    invocation independently polls for its output object or `-error.out`.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = DEFAULT_REGION,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        invocation_timeout_s: int = DEFAULT_INVOCATION_TIMEOUT_S,
        fips: bool = False,
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.poll_interval_s = poll_interval_s
        self.invocation_timeout_s = invocation_timeout_s
        self.fips = fips
        self._session: boto3.Session | None = None
        self._thread_local = threading.local()

    def _boto_cfg(self):
        """Per-client FIPS config. Applied per client rather than via
        AWS_USE_FIPS_ENDPOINT, which would also redirect the IAM Identity
        Center portal (portal.sso-fips.<region> does not exist)."""
        return BotoConfig(use_fips_endpoint=True) if self.fips else None

    def _initialize_client(self):
        """Resolve AWS credentials and verify identity once for the whole run."""
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            credentials.get_frozen_credentials()

            caller_identity = session.client("sts", config=self._boto_cfg()).get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

            self._session = session
            logger.info("AWS credentials resolved")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

    def _get_thread_clients(self):
        """Per-thread (s3, sagemaker-runtime) clients."""
        if not hasattr(self._thread_local, "clients"):
            self._thread_local.clients = (
                self._session.client("s3", config=self._boto_cfg()),
                self._session.client("sagemaker-runtime", config=self._boto_cfg()),
            )
        return self._thread_local.clients

    def runtime_endpoint_url(self) -> str:
        """The resolved sagemaker-runtime URL (printed so a log proves FIPS)."""
        if self._session is None:
            self._initialize_client()
        _, sm_rt = self._get_thread_clients()
        return sm_rt.meta.endpoint_url

    def _invoke_once(
        self,
        request_id: int,
        body: bytes,
        custom_attributes: str,
        inference_id_prefix: str,
        stop_event: threading.Event,
    ) -> tuple[int, float, bytes | None, str | None, Exception | None]:
        """
        Submit one async invocation with an inline payload and poll its
        result. Returns (request_id, elapsed_s, audio_or_None,
        output_s3_uri_or_None, error_or_None).
        """
        if stop_event.is_set():
            return request_id, 0.0, None, None, RuntimeError("Aborted due to prior error")

        s3, sm_rt = self._get_thread_clients()
        start = time.monotonic()
        inf_id = f"{inference_id_prefix}-{request_id:03d}"

        try:
            response = sm_rt.invoke_endpoint_async(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType="application/json",
                Accept="*/*",
                CustomAttributes=custom_attributes,
                InvocationTimeoutSeconds=self.invocation_timeout_s,
                InferenceId=inf_id,
            )
        except ParamValidationError as e:
            stop_event.set()
            hint = ""
            if "Body" in str(e):
                hint = (
                    " — this boto3 predates inline async payloads; "
                    "upgrade boto3 (uv sync)"
                )
            return request_id, time.monotonic() - start, None, None, RuntimeError(f"{e}{hint}")
        except ClientError as e:
            elapsed = time.monotonic() - start
            stop_event.set()
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            return request_id, elapsed, None, None, RuntimeError(
                f"InvokeEndpointAsync failed [{code}]: {msg}"
            )

        output_uri = response["OutputLocation"]
        failure_uri = response["FailureLocation"]
        out_bucket, out_key = _split_s3_uri(output_uri)
        fail_bucket, fail_key = _split_s3_uri(failure_uri)

        deadline = start + self.invocation_timeout_s + 60  # +60s for S3 propagation
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return request_id, time.monotonic() - start, None, None, RuntimeError(
                    "Aborted due to prior error"
                )

            # Probe both locations; first one wins.
            audio = self._get_or_none(s3, out_bucket, out_key)
            if audio is not None:
                return request_id, time.monotonic() - start, audio, output_uri, None

            failure = self._get_or_none(s3, fail_bucket, fail_key)
            if failure is not None:
                msg = failure.decode("utf-8", errors="replace").strip()
                return request_id, time.monotonic() - start, None, failure_uri, RuntimeError(
                    f"Endpoint reported failure: {msg}"
                )

            time.sleep(self.poll_interval_s)

        elapsed = time.monotonic() - start
        return request_id, elapsed, None, None, RuntimeError(
            f"Timed out after {elapsed:.0f}s waiting for {output_uri} / {failure_uri}"
        )

    @staticmethod
    def _get_or_none(s3, bucket: str, key: str) -> bytes | None:
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def run(
        self,
        bodies: list[bytes],
        custom_attributes: str,
        inference_id_prefix: str,
        num_requests: int = 1,
        concurrency: int = 1,
    ) -> list[tuple[int, float, bytes | None, str | None, Exception | None]]:
        """
        Submit `num_requests` async invocations (cycling through `bodies`)
        with up to `concurrency` in flight.

        Returns a list of (request_id, elapsed_s, audio, output_s3_uri, error)
        tuples in completion order.
        """
        if self._session is None:
            self._initialize_client()

        stop_event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="async-invoke",
        )
        futures = {}
        try:
            for i, body in zip(range(num_requests), itertools.cycle(bodies)):
                fut = executor.submit(
                    self._invoke_once,
                    i + 1,
                    body,
                    custom_attributes,
                    inference_id_prefix,
                    stop_event,
                )
                futures[fut] = i + 1
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    req_id = futures[future]
                    logger.error(f"[Request {req_id}] Unexpected thread error: {e}")
                    stop_event.set()
                    results.append((req_id, 0.0, None, None, e))
        except (KeyboardInterrupt, SystemExit):
            logger.warning("Async run interrupted — cancelling pending requests")
            stop_event.set()
            for f in futures:
                f.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize speech via the SageMaker `InvokeEndpointAsync` API with "
            "an inline request payload (no S3 input upload); the audio is "
            "collected from the endpoint's S3 output location."
        )
    )
    parser.add_argument("endpoint_name", help="SageMaker endpoint name (async-configured)")

    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text",
        help="Inline text to synthesize (overrides --text-file; single phrase).",
    )
    text_group.add_argument(
        "--text-file",
        default=DEFAULT_TEXT_FILE,
        metavar="PATH",
        help=(
            f"Text file with one phrase per line (default: {DEFAULT_TEXT_FILE}). "
            "Requests cycle through the phrases."
        ),
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Deepgram TTS voice model (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--extra",
        default="",
        metavar="k=v&k2=v2",
        help=(
            "Extra /v1/speak query parameters appended verbatim "
            "(e.g. 'encoding=linear16&container=wav&sample_rate=24000')."
        ),
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of invocations to keep in flight in parallel (default: 1)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help=(
            "Total number of invocations to submit (default: one per phrase, "
            "or --concurrency if larger)"
        ),
    )
    parser.add_argument(
        "--invocation-timeout",
        type=int,
        default=DEFAULT_INVOCATION_TIMEOUT_S,
        metavar="SECONDS",
        help=(
            f"Per-invocation timeout passed to SageMaker (default: "
            f"{DEFAULT_INVOCATION_TIMEOUT_S}; max: {ASYNC_INVOKE_MAX_TIMEOUT_S})."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        metavar="SECONDS",
        help=f"Seconds between S3 polls for each invocation's result (default: {DEFAULT_POLL_INTERVAL_S}).",
    )
    parser.add_argument(
        "--inference-id-prefix",
        default=None,
        metavar="PREFIX",
        help="Prefix for each invocation's InferenceId (default: async-<random>).",
    )
    parser.add_argument(
        "--save-audio-dir",
        default=None,
        metavar="DIR",
        help="Save each request's audio to DIR (extension sniffed from the audio).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--fips",
        action="store_true",
        help=(
            "Route AWS calls to the FIPS 140-3 endpoints (runtime-fips invoke, "
            "s3-fips output polling). OFF by default."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> int:
    args = _make_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.concurrency < 1:
        print("ERROR: --concurrency must be >= 1")
        return 1
    if not 1 <= args.invocation_timeout <= ASYNC_INVOKE_MAX_TIMEOUT_S:
        print(f"ERROR: --invocation-timeout must be 1..{ASYNC_INVOKE_MAX_TIMEOUT_S}")
        return 1

    try:
        phrases = _load_phrases(args)
        bodies = [_build_body(p) for p in phrases]
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    long_phrases = sum(1 for p in phrases if len(p) > TTS_MAX_CHARS)
    if long_phrases:
        logger.warning(
            f"{long_phrases} phrase(s) exceed the {TTS_MAX_CHARS}-character TTS "
            "limit and will be rejected by the endpoint"
        )

    if args.requests is None:
        args.requests = max(len(phrases), args.concurrency)
    if args.requests < 1:
        print("ERROR: --requests must be >= 1")
        return 1

    extra_pairs = _parse_extra(args.extra)
    custom_attributes = _build_custom_attributes(args.voice, extra_pairs)

    client = AsyncTTSClient(
        endpoint_name=args.endpoint_name,
        region=args.region,
        poll_interval_s=args.poll_interval,
        invocation_timeout_s=args.invocation_timeout,
        fips=args.fips,
    )
    try:
        runtime_url = client.runtime_endpoint_url()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    inference_id_prefix = args.inference_id_prefix or f"async-{uuid.uuid4().hex[:8]}"
    source = "--text" if args.text else args.text_file

    print("=" * 60)
    print("Deepgram SageMaker TTS Async Client (Inline Payload)")
    print("=" * 60)
    print(f"Endpoint:        {args.endpoint_name}")
    print(f"Runtime URL:     {runtime_url}")
    print(f"FIPS:            {'yes (--fips)' if args.fips else 'no'}")
    print(f"Region:          {args.region}")
    print(f"Voice:           {args.voice}")
    print(f"Text source:     {source} ({len(phrases)} phrase(s))")
    print(f"Payload size:    {min(map(len, bodies)):,}–{max(map(len, bodies)):,} bytes "
          f"(inline max {INLINE_BODY_MAX_BYTES:,})")
    print(f"Requests:        {args.requests}")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Per-inv timeout: {args.invocation_timeout}s")
    print(f"Poll interval:   {args.poll_interval}s")
    print(f"InferenceId:     {inference_id_prefix}-NNN")
    print(f"Path+Params:     {custom_attributes}")
    if args.save_audio_dir:
        print(f"Save audio to:   {args.save_audio_dir}")
    print("=" * 60)
    print(f"\nSubmitting {args.requests} request(s) ({args.concurrency} concurrent)...\n")

    wall_start = time.monotonic()
    results = client.run(
        bodies=bodies,
        custom_attributes=custom_attributes,
        inference_id_prefix=inference_id_prefix,
        num_requests=args.requests,
        concurrency=args.concurrency,
    )
    wall_elapsed = time.monotonic() - wall_start

    results.sort(key=lambda r: r[0])
    if args.save_audio_dir:
        os.makedirs(args.save_audio_dir, exist_ok=True)

    successes = 0
    failures = 0
    latencies: list[float] = []
    total_audio_s = 0.0

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for req_id, elapsed, audio, output_uri, error in results:
        if error is not None:
            failures += 1
            print(f"[Request {req_id:>3}] ERROR ({elapsed:.2f}s): {error}")
            if output_uri:
                print(f"             output uri: {output_uri}")
            continue
        if not audio:
            failures += 1
            print(f"[Request {req_id:>3}] ERROR ({elapsed:.2f}s): empty audio at {output_uri}")
            continue

        successes += 1
        latencies.append(elapsed)
        fmt = _sniff_audio(audio)
        dur = _wav_duration(audio) if fmt == "wav" else None
        if dur is not None:
            total_audio_s += dur
        dur_msg = f" duration={dur:.2f}s" if dur is not None else ""
        print(
            f"[Request {req_id:>3}] ✓ ({elapsed:.2f}s) {len(audio):,} bytes "
            f"format={fmt}{dur_msg} output={output_uri}"
        )
        if args.save_audio_dir:
            path = os.path.join(args.save_audio_dir, f"request_{req_id:03d}.{fmt}")
            with open(path, "wb") as f:
                f.write(audio)
            print(f"             saved: {path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total requests:  {args.requests}")
    print(f"Successful:      {successes}")
    print(f"Failed:          {failures}")
    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
        print(f"Min latency:     {min(latencies):.2f}s")
        print(f"Avg latency:     {statistics.mean(latencies):.2f}s")
        print(f"P95 latency:     {latencies_sorted[p95_idx]:.2f}s")
        print(f"Max latency:     {max(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"Std dev:         {statistics.stdev(latencies):.2f}s")
    if total_audio_s:
        print(f"Audio (WAV):     {total_audio_s:.2f}s total")
    print(f"Total wall time: {wall_elapsed:.2f}s")
    if args.requests > 0 and wall_elapsed > 0:
        print(f"Throughput:      {args.requests / wall_elapsed:.2f} req/s")
    print("=" * 60)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
