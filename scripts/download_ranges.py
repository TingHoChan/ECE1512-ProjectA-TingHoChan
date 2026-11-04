"""Download a public file with verified, resumable parallel byte ranges.

Chunks are kept under output.name.parts and are individually size-checked.
The final output is assembled only after all chunks are complete; an expected
SHA256 may be supplied from the publisher's LFS metadata.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


def acquire_output_lock(output):
    """Serialize writers targeting one output, including separate queue jobs."""
    handle = output.with_name(output.name+'.download.lock').open('a+b')
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    while True:
        handle.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('output', type=Path)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--sha256')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--chunk-mb', type=int, default=16)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the handle alive until main returns; process exit also releases the
    # OS lock. Another queue can safely wait and reuse the verified result.
    output_lock = acquire_output_lock(args.output)
    verified_path = args.output.with_name(args.output.name+'.verified.json')
    if args.output.exists() and verified_path.exists():
        previous = json.loads(verified_path.read_text())
        if (previous.get('verified') and previous.get('sha256') == args.sha256
                and args.output.stat().st_size == args.size):
            print(f'Already publisher-verified: {args.output}', flush=True)
            return
    parts = args.output.with_name(args.output.name + '.parts')
    parts.mkdir(exist_ok=True)
    manifest = parts / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if (args.sha256 and previous.get('sha256') == args.sha256
                and previous.get('size') == args.size):
            # Resume the existing byte layout even when a caller has a different
            # preferred chunk size; changing file URLs does not invalidate data.
            args.chunk_mb = int(previous['chunk_mb'])
    expected = {'url': args.url, 'size': args.size, 'sha256': args.sha256,
                'chunk_mb': args.chunk_mb}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        # A moving source URL can be replaced by an immutable revision URL when
        # the publisher SHA256, size, and chunk layout are identical.
        compatible = args.sha256 and all(previous.get(k) == expected[k]
                         for k in ['size','sha256','chunk_mb'])
        if previous != expected and not compatible:
            raise RuntimeError('Existing download manifest differs; use another output path.')
    manifest.write_text(json.dumps(expected, indent=2))
    chunk_size = args.chunk_mb * 1024 * 1024
    chunks = [(i, start, min(start + chunk_size, args.size) - 1)
              for i, start in enumerate(range(0, args.size, chunk_size))]
    downloaded = [0]
    started = time.monotonic()
    last_report = [0.0]

    # Preflight the byte length. Each chunk uses the original URL so expiring
    # signed redirects refresh automatically during a long, resumable transfer.
    with urllib.request.urlopen(urllib.request.Request(args.url, method='HEAD'), timeout=60) as response:
        published_size = response.headers.get('Content-Length')
        if published_size is not None and int(published_size) != args.size:
            raise RuntimeError(f'Publisher size changed: {published_size} != {args.size}')
    print(f'Start {args.output.name}: {args.size:,} bytes; {len(chunks)} chunks; {args.workers} workers', flush=True)

    def one(chunk):
        index, start, end = chunk
        path = parts / f'{index:05d}.part'
        length = end - start + 1
        if path.exists() and path.stat().st_size == length:
            return length
        temp = path.with_suffix('.tmp')
        for attempt in range(12):
            try:
                resume = temp.stat().st_size if temp.exists() else 0
                if resume == length:
                    temp.replace(path)
                    return length
                if resume > length:
                    raise RuntimeError(f'Oversized partial chunk {index}')
                range_start = start + resume
                request = urllib.request.Request(args.url, headers={
                    'Range': f'bytes={range_start}-{end}', 'User-Agent': 'ECE1512-project-reproducibility/1.0'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    content_range = response.headers.get('Content-Range', '')
                    if response.status != 206 or content_range != f'bytes {range_start}-{end}/{args.size}':
                        raise RuntimeError(f'Unexpected range response: {response.status} {content_range}')
                    with temp.open('ab') as output:
                        while True:
                            data = response.read(64 * 1024)
                            if not data:
                                break
                            output.write(data)
                if temp.stat().st_size != length:
                    raise RuntimeError(f'Chunk {index} length mismatch')
                temp.replace(path)
                return length
            except Exception as exc:
                print(f'Chunk {index} attempt {attempt+1}: {type(exc).__name__}: {exc}', flush=True)
                if attempt == 11:
                    raise
                time.sleep(min(2 ** attempt, 30))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, chunk) for chunk in chunks]
        for future in as_completed(futures):
            downloaded[0] += future.result()
            now = time.monotonic()
            if now - last_report[0] >= 15 or downloaded[0] == args.size:
                elapsed = now - started
                print(f'{downloaded[0]/args.size:.1%} {downloaded[0]/1e9:.2f}/{args.size/1e9:.2f} GB '
                      f'{downloaded[0]/1e6/max(elapsed,.001):.2f} MB/s elapsed={elapsed:.0f}s', flush=True)
                last_report[0] = now
    assembled = args.output.with_name(args.output.name + '.assembling')
    sha = hashlib.sha256()
    with assembled.open('wb') as output:
        for index, _, _ in chunks:
            with (parts / f'{index:05d}.part').open('rb') as stream:
                for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    output.write(data)
                    sha.update(data)
    digest = sha.hexdigest()
    if assembled.stat().st_size != args.size:
        raise RuntimeError('Assembled size differs from publisher size')
    if args.sha256 and digest != args.sha256:
        raise RuntimeError(f'SHA256 mismatch: expected {args.sha256}, got {digest}')
    assembled.replace(args.output)
    verification = {'path': str(args.output.resolve()), 'size': args.size, 'sha256': digest,
                    'publisher_sha256': args.sha256, 'verified': bool(args.sha256),
                    'seconds': time.monotonic() - started, 'source': args.url}
    args.output.with_name(args.output.name + '.verified.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
