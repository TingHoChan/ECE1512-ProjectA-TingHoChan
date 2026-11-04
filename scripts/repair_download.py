"""Repair interrupted zero-filled download spans, then require publisher SHA256.

This is a recovery helper for generated .parts files, never for trained weights
or an already verified raw dataset. Zero runs are only candidates: bytes are
replaced from the immutable publisher source, and legitimate zeros stay equal.
No local feature values or labels are inferred or synthesized.

Example: python scripts/repair_download.py feature/C16/patch_feats_pretrain_medical_ssl.h5 --patch
Without --patch, only candidate spans are recorded. The final complete-file
SHA256 is always the authority; the aligned scan is a recovery heuristic.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from download_ranges import acquire_output_lock


def scan_spans(path, block_bytes=4096):
    data = path.read_bytes()
    zero = bytes(block_bytes)
    spans = []
    for offset in range(0, len(data)-block_bytes+1, block_bytes):
        if data[offset:offset+block_bytes] != zero:
            continue
        start, end = max(0, offset-block_bytes), min(len(data), offset+2*block_bytes)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', help='Relative asset path from output/data_manifest.json')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--block-bytes', type=int, default=4096)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    publisher = json.loads((root/'output/data_manifest.json').read_text())
    if args.asset not in publisher['files']:
        raise ValueError('Choose an exact asset path from the publisher manifest')
    expected = publisher['files'][args.asset]
    if 'sha256' not in expected:
        raise ValueError('This helper applies to the SHA256-addressed binary files')
    target = (root/args.asset).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes project directory')
    verified_path = target.with_name(target.name+'.verified.json')
    if verified_path.exists():
        prior = json.loads(verified_path.read_text())
        if prior.get('verified') and prior.get('sha256') == expected['sha256']:
            print('Already publisher-verified; repair is unnecessary.')
            return
    parts = target.with_name(target.name+'.parts')
    layout = json.loads((parts/'manifest.json').read_text())
    if layout['size'] != expected['size'] or layout['sha256'] != expected['sha256']:
        raise ValueError('Chunk layout does not match the publisher asset')
    chunk_bytes = int(layout['chunk_mb'])*1024*1024
    url = publisher['source']+'/resolve/'+publisher['revision']+'/'+args.asset+'?download=true'
    lock = acquire_output_lock(target)
    try:
        candidates = []
        for path in sorted([*parts.glob('*.part'), *parts.glob('*.tmp')]):
            for start, end in scan_spans(path, args.block_bytes):
                candidates.append((path, start, end))
        records = []

        def recover(job):
            path, start, end = job
            index = int(path.stem)
            absolute_start, absolute_end = index*chunk_bytes+start, index*chunk_bytes+end-1
            if not args.patch:
                return {'chunk':path.name, 'start':start, 'end':end, 'status':'candidate_only'}
            for attempt in range(5):
                try:
                    request = urllib.request.Request(url, headers={'Range':f'bytes={absolute_start}-{absolute_end}'})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        expected_range = f'bytes {absolute_start}-{absolute_end}/{expected["size"]}'
                        if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                            raise RuntimeError('Publisher response did not match requested byte range')
                        data = response.read()
                    if len(data) != end-start:
                        raise RuntimeError('Short publisher response')
                    with path.open('r+b') as handle:
                        handle.seek(start)
                        old = handle.read(end-start)
                        if old != data:
                            handle.seek(start)
                            handle.write(data)
                            handle.flush()
                            os.fsync(handle.fileno())
                    return {'chunk':path.name, 'start':start, 'end':end,
                            'changed':old != data, 'before_sha256':hashlib.sha256(old).hexdigest(),
                            'publisher_range_sha256':hashlib.sha256(data).hexdigest()}
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(recover, item) for item in candidates]):
                records.append(future.result())
    finally:
        lock.close()
    audit = {'asset':args.asset, 'publisher_revision':publisher['revision'],
             'publisher_sha256':expected['sha256'], 'patch_requested':args.patch,
             'method':'Aligned zero-span candidates with one-block boundary padding; immutable HTTP Range replacement',
             'records':records}
    output = root/'output'/('download_repair_'+target.parent.name+'_'+target.stem+'.json')
    output.write_text(json.dumps(audit, indent=2), encoding='utf8')
    print('Candidate spans:',len(records),'changed:',sum(bool(r.get('changed')) for r in records))
    if args.patch:
        subprocess.run([sys.executable,str(root/'scripts/download_ranges.py'),url,str(target),
                        '--size',str(expected['size']),'--sha256',expected['sha256'],
                        '--workers',str(args.workers),'--chunk-mb',str(layout['chunk_mb'])],check=True)
        print('Complete file passed publisher SHA256; recovery is verified.')


if __name__ == '__main__':
    main()
