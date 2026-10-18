#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Publish manifest-listed course materials to the existing personal Pages repository."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def api(endpoint, payload=None, method=None):
    command = ['gh', 'api', endpoint]
    if method:
        command += ['--method', method]
    if payload is not None:
        command += ['--input', '-']
    result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                            text=True, capture_output=True, timeout=120, check=True)
    return json.loads(result.stdout)


def records(config):
    result = []
    for topic in config['topics']:
        for doc in topic['documents'].values():
            for ext in ['md', 'pdf']:
                data = (ROOT / doc[ext]).read_bytes()
                # A leading blank line keeps YAML-bearing Markdown static in Jekyll.
                if ext == 'md' and data.startswith(b'---\n'):
                    data = b'\n' + data
                result.append((doc['site_' + ext], data))
    return result


def publish(config, files):
    repo = config['website_repository']
    branch = config['website_branch']
    ref = api(f'repos/{repo}/git/ref/heads/{branch}')['object']['sha']
    base = api(f'repos/{repo}/git/commits/{ref}')['tree']['sha']
    tree = api(f'repos/{repo}/git/trees/{base}?recursive=1')
    if tree.get('truncated'):
        raise RuntimeError('Website tree is truncated; cannot compare safely')
    current = {item['path']: item['sha'] for item in tree['tree']}
    import base64
    changes = []
    for dest, data in files:
        blob_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if current.get(dest) == blob_sha:
            continue
        blob = api(f'repos/{repo}/git/blobs', {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        changes.append({'path': dest, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    if not changes:
        print('Website repository already matches all materials.', flush=True)
        return None
    new_tree = api(f'repos/{repo}/git/trees', {'base_tree': base, 'tree': changes})
    commit = api(f'repos/{repo}/git/commits', {
        'message': 'Synchronize licensed bilingual course materials and PDF builds',
        'tree': new_tree['sha'], 'parents': [ref],
    })
    # No force update: a concurrent website commit causes a safe rejection.
    api(f'repos/{repo}/git/refs/heads/{branch}', {'sha': commit['sha'], 'force': False}, 'PATCH')
    print(f'Published {len(changes)} files: {commit["html_url"]}', flush=True)
    return commit['sha']


def wait_for_pages(config, commit, deadline):
    while time.monotonic() < deadline:
        runs = api(f'repos/{config["website_repository"]}/actions/runs?head_sha={commit}&per_page=20')['workflow_runs']
        matching = [r for r in runs if r['name'] == 'Deploy Jekyll site to Pages']
        if matching:
            run = matching[0]
            if run['status'] == 'completed':
                if run['conclusion'] != 'success':
                    raise RuntimeError(f'Pages deployment failed: {run["html_url"]}')
                print(f'Pages deployment succeeded: {run["html_url"]}', flush=True)
                return
            print(f'Pages deployment: {run["status"]}', flush=True)
        time.sleep(10)
    raise TimeoutError('Pages deployment did not finish within the requested timeout')


def verify(config, files):
    def check(record):
        dest, expected = record
        url = config['site_base'] + quote(dest)
        try:
            request = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
            with urllib.request.urlopen(request, timeout=45) as response:
                actual = response.read()
            return url if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest() else None
        except Exception as exc:
            return f'{url}: {exc}'
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return [error for error in pool.map(check, files) if error]


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--publish', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    parser.add_argument('--wait', action='store_true', help='Wait for the known website deployment workflow and CDN verification')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    subprocess.run(['python3', str(ROOT / 'scripts/check.py')], check=True)
    config = json.loads((ROOT / 'materials.json').read_text())
    files = records(config)
    deadline = time.monotonic() + args.timeout
    if args.publish:
        commit = publish(config, files)
        if commit and args.wait:
            wait_for_pages(config, commit, deadline)
        elif commit:
            print('Deployment pending. Run --verify-only after Pages finishes.', flush=True)
            return
    while True:
        errors = verify(config, files)
        if not errors:
            print(f'PASS: all {len(files)} public files match the current source/PDF bytes.', flush=True)
            return
        if not args.wait or time.monotonic() >= deadline:
            raise RuntimeError('Online verification failed:\n' + '\n'.join(errors))
        print(f'Waiting for website/CDN: {len(errors)} files not current yet.', flush=True)
        time.sleep(10)


if __name__ == '__main__':
    main()
