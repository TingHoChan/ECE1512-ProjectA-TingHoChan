"""Download the exact course data revision and verify publisher hashes.

Run: python scripts/download_data.py --include-slides
The feature files are always downloaded; --include-slides adds the two WSIs
and tumor contours required for attention overlays. Large downloads resume
completed and partial chunks. No tokens or account credentials are required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-slides', action='store_true')
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'output/data_manifest.json').read_text())
    names = ['feature/C17/patch_feats_pretrain_medical_ssl.h5',
             'feature/BRACS/patch_feats_pretrain_medical_ssl.h5',
             'feature/C16/patch_feats_pretrain_medical_ssl.h5']
    if args.include_slides:
        names += ['slide/test_068.tif','slide/test_067.tif','slide/test_068.xml']
    for name in names:
        expected = manifest['files'][name]
        target = root/name
        url = manifest['source']+'/resolve/'+manifest['revision']+'/'+name+'?download=true'
        target.parent.mkdir(parents=True,exist_ok=True)
        sidecar = target.with_name(target.name+'.verified.json')
        if target.exists() and sidecar.exists():
            previous = json.loads(sidecar.read_text())
            if (previous.get('verified') and target.stat().st_size == expected['size']
                    and previous.get('sha256') == expected.get('sha256')):
                print('Already verified: '+name,flush=True)
                continue
        if 'sha256' in expected:
            chunk_mb = '4' if name.startswith('slide/') else '16'
            command = [sys.executable,str(root/'scripts/download_ranges.py'),url,str(target),
                       '--size',str(expected['size']),'--sha256',expected['sha256'],
                       '--workers',str(args.workers),'--chunk-mb',chunk_mb]
            subprocess.run(command,check=True)
        else:
            # Small XML is stored as a Git blob rather than Git LFS. Verify the
            # publisher's Git object ID as well as recording a local SHA256.
            with urllib.request.urlopen(url,timeout=60) as response:
                content = response.read()
            git_digest = hashlib.sha1(('blob '+str(len(content))+'\0').encode()+content).hexdigest()
            if len(content)!=expected['size'] or git_digest!=expected['git_blob_sha1']:
                raise RuntimeError('Publisher XML hash/size mismatch')
            target.write_bytes(content)
            sidecar.write_text(json.dumps({'path':str(target),'size':len(content),
                'sha256':hashlib.sha256(content).hexdigest(),'git_blob_sha1':git_digest,
                'verified':True,'source':url},indent=2))


if __name__=='__main__': main()
