"""Fixture workspace tools: bounded, path-checked, receipted (E3 §4).

``read_file`` and ``write_file`` are restricted to the writable ``workspace/``
subtree; the evaluator and ``case.json`` sit outside both read and write
authority, so the only way to observe the evaluator's bound is to actually run
``run_test`` (or the host-only ``verify_success`` path, which is never exposed
as a tool). All receipts are canonical JSON carrying their own sha256.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import subprocess
import sys
import time

from ..architecture.contracts import fingerprint
from .actions import TOOL_ARGUMENT_SCHEMAS, validate_tool_arguments

CASE_FORMAT = 'replica.e3.fixture.v1'
ROLES = ('train', 'dev', 'heldout', 'interrupted')
KNOWN_TOOLS = ('read_file', 'write_file', 'set_constant', 'run_test')
MAX_READ_BYTES = 16 * 1024
MAX_WRITE_BYTES = 8 * 1024
MAX_OUTPUT_BYTES = 20 * 1024
MAX_EXCERPT = 600
TEST_TIMEOUT_SECONDS = 30
MAX_PATH = 512
MAX_CASE_JSON_BYTES = 64 * 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def excerpt(value: bytes) -> str:
    return value[:MAX_OUTPUT_BYTES].decode('utf-8', errors='replace')[:MAX_EXCERPT]


def _case_workspace_parts(raw, name: str) -> tuple[str, ...]:
    """Normalize one case-root-relative workspace entry to workspace-relative parts.

    ``case.json`` describes ``workspace_files``/``workspace_entry`` relative to
    the case root (``workspace/app_settings.py``). Bare names are accepted as
    shorthand but always resolve under ``workspace/``.
    """
    if not isinstance(raw, str) or not raw.strip() or '\x00' in raw:
        raise ValueError(f'Fixture {name} must be a nonempty path string')
    if len(raw) > MAX_PATH:
        raise ValueError(f'Fixture {name} exceeds the bounded path length')
    pure = PurePosixPath(raw)
    if pure.is_absolute() or raw.startswith('/') or (len(raw) > 1 and raw[1] == ':'):
        raise ValueError(f'Fixture {name} must be case-relative')
    parts = pure.parts
    if any(part == '..' for part in parts):
        raise ValueError(f'Fixture {name} may not traverse outside the case root')
    if parts and parts[0] == 'workspace':
        parts = parts[1:]
    if not parts:
        raise ValueError(f'Fixture {name} must name a file under workspace/')
    return tuple(parts)


def _receipt(base: dict) -> dict:
    value = dict(base)
    value['receipt_sha256'] = fingerprint(value)
    return value


def sanitized_env(case_root: Path, host_tmp: Path) -> dict:
    return {
        'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
        'HOME': str(host_tmp),
        'TMPDIR': str(host_tmp),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONHASHSEED': '0',
        'PYTHONPATH': str(case_root),
    }


class ToolRegistry:
    """Name-bounded dispatch; only registered tools are callable by the model."""

    def __init__(self, handlers: dict):
        self._handlers = dict(handlers)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def schema(self, name: str) -> dict:
        return dict(TOOL_ARGUMENT_SCHEMAS[name])

    def call(self, name: str, arguments: dict) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            return {'status': 'denied', 'kind': name, 'error': 'tool_not_available'}
        return handler(dict(arguments or {}))


class FixtureWorkspace:
    """One connected fixture case; host-owned authority over the evaluator."""

    def __init__(self, case_root):
        self.case_root = Path(case_root).resolve()
        case_path = self.case_root / 'case.json'
        if not case_path.is_file():
            raise ValueError(f'Fixture case.json missing under {self.case_root}')
        raw = case_path.read_bytes()
        if len(raw) > MAX_CASE_JSON_BYTES:
            raise ValueError('case.json exceeds the bounded size')
        case = json.loads(raw.decode('utf-8'))
        if case.get('format') != CASE_FORMAT:
            raise ValueError(f'Fixture format must be {CASE_FORMAT}')
        if case.get('role') not in ROLES:
            raise ValueError(f'Fixture role must be one of {ROLES}')
        self.case_id = case['case_id']
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError('Fixture case_id required')
        self.role = case['role']
        self.goal = case['goal']
        if not isinstance(self.goal, str) or not self.goal:
            raise ValueError('Fixture goal required')
        condition = case.get('success_condition')
        if (not isinstance(condition, dict)
                or not isinstance(condition.get('condition_id'), str)
                or not isinstance(condition.get('description'), str)):
            raise ValueError('Fixture success_condition requires condition_id/description')
        self.success_condition = {'condition_id': condition['condition_id'],
                                  'description': condition['description']}
        self.constraints = tuple(case.get('constraints') or ())
        if not all(isinstance(c, str) for c in self.constraints):
            raise ValueError('Fixture constraints must be strings')
        self.allowed_tools = tuple(case.get('allowed_tools') or ())
        if not self.allowed_tools or not set(self.allowed_tools) <= set(KNOWN_TOOLS):
            raise ValueError(f'Fixture allowed_tools must be a subset of {KNOWN_TOOLS}')
        evaluator = case.get('evaluator')
        if not isinstance(evaluator, str) or not evaluator:
            raise ValueError('Fixture evaluator path required')
        evaluator_path = PurePosixPath(evaluator)
        if evaluator_path.is_absolute() or '..' in evaluator_path.parts:
            raise ValueError('Fixture evaluator must be a relative in-root path')
        if evaluator_path.parts and evaluator_path.parts[0] == 'workspace':
            raise ValueError('The evaluator must live outside the writable workspace')
        self.evaluator_rel = evaluator
        self.evaluator_path = self.case_root / Path(*evaluator_path.parts)
        if not self.evaluator_path.is_file():
            raise ValueError(f'Fixture evaluator missing: {evaluator}')
        self.variant = dict(case.get('variant') or {})
        reference = case.get('reference_repair') or {}
        if not isinstance(reference, dict):
            raise ValueError('Fixture reference_repair must be an object')
        self.reference_repair = dict(reference)
        self.workspace_dir = self.case_root / 'workspace'
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        declared_files = case.get('workspace_files') or ()
        if not isinstance(declared_files, (list, tuple)):
            raise ValueError('Fixture workspace_files must be a list')
        workspace_files = []
        for raw in declared_files:
            parts = _case_workspace_parts(raw, 'workspace file')
            if not self.workspace_dir.joinpath(*parts).is_file():
                raise ValueError(f'Fixture workspace file missing: {raw}')
            workspace_files.append('workspace/' + '/'.join(parts))
        self.workspace_files = tuple(workspace_files)
        entry = case.get('workspace_entry')
        if entry is not None:
            parts = _case_workspace_parts(entry, 'workspace entry')
            if not self.workspace_dir.joinpath(*parts).is_file():
                raise ValueError(f'Fixture workspace entry missing: {entry}')
            entry = 'workspace/' + '/'.join(parts)
        self.workspace_entry = entry
        self.host_tmp = self.case_root / '.host_tmp'
        self.host_tmp.mkdir(parents=True, exist_ok=True)
        self._env = sanitized_env(self.case_root, self.host_tmp)

    # ---- path authority ----------------------------------------------------
    def _workspace_path(self, raw) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('path must be a nonempty string')
        if len(raw) > MAX_PATH or '\x00' in raw:
            raise ValueError('path exceeds bounds')
        pure = PurePosixPath(raw)
        if pure.is_absolute() or raw.startswith('/') or (len(raw) > 1 and raw[1] == ':'):
            raise ValueError('absolute paths are outside workspace authority')
        parts = pure.parts
        if not parts or parts[0] != 'workspace' or len(parts) < 2:
            raise ValueError('model paths must start with workspace/')
        if any(part == '..' for part in parts):
            raise ValueError('parent traversal is outside workspace authority')
        parts = parts[1:]
        target = self.workspace_dir.joinpath(*parts)
        resolved = target.resolve()
        workspace_real = self.workspace_dir.resolve()
        if not resolved.is_relative_to(workspace_real) or resolved == workspace_real:
            raise ValueError('target resolves outside the writable workspace')
        current = workspace_real
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('symlink components are outside workspace authority')
        return resolved

    def workspace_hashes(self) -> dict:
        hashes = {}
        for path in sorted(self.workspace_dir.rglob('*')):
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(self.workspace_dir).as_posix()
            hashes[rel] = sha256_bytes(path.read_bytes())
        return hashes

    # ---- tools -------------------------------------------------------------
    def read_file(self, arguments: dict) -> dict:
        try:
            args = validate_tool_arguments('read_file', arguments)
            target = self._workspace_path(args['path'])
        except ValueError as error:
            return {'status': 'denied', 'kind': 'read_file', 'error': str(error)}
        if not target.exists() or target.is_symlink() or not target.is_file():
            return {'status': 'error', 'kind': 'read_file', 'error': 'not_a_regular_file',
                    'path': args['path']}
        content = target.read_bytes()
        if len(content) > MAX_READ_BYTES:
            return {'status': 'error', 'kind': 'read_file', 'error': 'file_too_large',
                    'path': args['path'], 'size': len(content)}
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            return {'status': 'error', 'kind': 'read_file', 'error': 'not_utf8',
                    'path': args['path']}
        return {'status': 'ok', 'kind': 'read_file', 'path': args['path'],
                'sha256': sha256_bytes(content), 'size': len(content), 'content': text,
                'excerpt': text[:MAX_EXCERPT]}

    def write_file(self, arguments: dict) -> dict:
        try:
            args = validate_tool_arguments('write_file', arguments)
            target = self._workspace_path(args['path'])
        except ValueError as error:
            return {'status': 'denied', 'kind': 'write_file', 'error': str(error)}
        content = args['content'].encode('utf-8')
        if len(content) > MAX_WRITE_BYTES:
            return {'status': 'denied', 'kind': 'write_file', 'error': 'content_too_large',
                    'size': len(content)}
        created = not target.exists()
        if target.is_symlink() or (target.exists() and not target.is_file()):
            return {'status': 'error', 'kind': 'write_file', 'error': 'target_not_regular',
                    'path': args['path']}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            for parent in target.parents:
                if parent.is_symlink():
                    return {'status': 'error', 'kind': 'write_file',
                            'error': 'symlink_components', 'path': args['path']}
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                                 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(content)
        except OSError as error:
            return {'status': 'error', 'kind': 'write_file', 'error': f'io_error: {error}',
                    'path': args['path']}
        return {'status': 'ok', 'kind': 'write_file', 'path': args['path'],
                'sha256': sha256_bytes(content), 'size': len(content), 'created': created}

    def set_constant(self, arguments: dict) -> dict:
        """Bounded repair: replace exactly one numeric module-level assignment."""
        import re
        try:
            args = validate_tool_arguments('set_constant', arguments)
        except ValueError as error:
            return {'status': 'denied', 'kind': 'set_constant', 'error': str(error)}
        parts = _case_workspace_parts(self.workspace_entry, 'workspace_entry')
        relative = 'workspace/' + '/'.join(parts)
        target = self.workspace_dir.joinpath(*parts)
        if not target.exists() or target.is_symlink() or not target.is_file():
            return {'status': 'error', 'kind': 'set_constant', 'error': 'not_a_regular_file',
                    'path': relative}
        raw = target.read_bytes()
        if len(raw) > MAX_WRITE_BYTES:
            return {'status': 'denied', 'kind': 'set_constant', 'error': 'file_too_large'}
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return {'status': 'error', 'kind': 'set_constant', 'error': 'not_utf8'}
        pattern = re.compile(
            r'^([ \t]*)' + re.escape(args['name']) + r'([ \t]*=[ \t]*)'
            r'(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)([ \t]*)$',
            re.MULTILINE)
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            return {'status': 'denied', 'kind': 'set_constant',
                    'error': f'constant_must_occur_exactly_once:{args["name"]}',
                    'occurrences': len(matches)}
        match = matches[0]
        old_value = match.group(3)
        new_value = str(int(args['value'])) if type(args['value']) is int else repr(args['value'])
        updated = text[:match.start(3)] + new_value + text[match.end(3):]
        written = self.write_file({'path': relative, 'content': updated})
        if written.get('status') != 'ok':
            return {'status': 'error', 'kind': 'set_constant',
                    'error': written.get('error', 'write_failed')}
        return {'status': 'ok', 'kind': 'set_constant', 'name': args['name'],
                'old_value': old_value, 'new_value': new_value,
                'path': relative, 'sha256': written['sha256'],
                'receipt': written}

    def _run_evaluator(self) -> dict:
        command = [sys.executable, self.evaluator_rel]
        workspace_before = self.workspace_hashes()
        started = time.monotonic()
        timed_out = False
        returncode = None
        stdout, stderr = b'', b''
        try:
            completed = subprocess.run(
                command, cwd=str(self.case_root), env=self._env,
                capture_output=True, timeout=TEST_TIMEOUT_SECONDS)
            returncode = completed.returncode
            stdout, stderr = completed.stdout or b'', completed.stderr or b''
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            stdout = expired.stdout or b''
            stderr = expired.stderr or b''
        duration_ms = int((time.monotonic() - started) * 1000)
        workspace_after = self.workspace_hashes()
        return _receipt({
            'kind': 'run_test', 'command': command, 'cwd': str(self.case_root),
            'exit_code': returncode, 'timed_out': timed_out, 'duration_ms': duration_ms,
            'stdout_sha256': sha256_bytes(stdout[:MAX_OUTPUT_BYTES]),
            'stderr_sha256': sha256_bytes(stderr[:MAX_OUTPUT_BYTES]),
            'stdout_excerpt': excerpt(stdout), 'stderr_excerpt': excerpt(stderr),
            'workspace_hashes_before': workspace_before,
            'workspace_hashes_after': workspace_after,
        })

    def run_test(self, arguments: dict) -> dict:
        try:
            validate_tool_arguments('run_test', arguments)
        except ValueError as error:
            return {'status': 'denied', 'kind': 'run_test', 'error': str(error)}
        receipt = self._run_evaluator()
        status = 'ok' if receipt['exit_code'] == 0 and not receipt['timed_out'] else 'error'
        return {'status': status, 'kind': 'run_test', 'receipt': receipt}

    def verify_success(self, condition_ids=None) -> dict:
        """Separate host path; the only authority that may satisfy a success condition."""
        receipt = self._run_evaluator()
        passed = receipt['exit_code'] == 0 and not receipt['timed_out']
        receipt['kind'] = 'verify_success'
        receipt['passed'] = passed
        receipt['condition_ids'] = list(condition_ids or [self.success_condition['condition_id']])
        receipt = _receipt({key: value for key, value in receipt.items()
                            if key != 'receipt_sha256'})
        return {'passed': passed, 'receipt': receipt}

    def host_reference_repair(self) -> dict:
        """Host validator only; never enters a prompt, tool result or journal payload."""
        return dict(self.reference_repair)

    def registry(self, allowed_tools=None) -> ToolRegistry:
        allowed = tuple(allowed_tools) if allowed_tools is not None else self.allowed_tools
        handlers = {}
        if 'read_file' in allowed:
            handlers['read_file'] = self.read_file
        if 'write_file' in allowed:
            handlers['write_file'] = self.write_file
        if 'set_constant' in allowed:
            handlers['set_constant'] = self.set_constant
        if 'run_test' in allowed:
            handlers['run_test'] = self.run_test
        return ToolRegistry(handlers)

    def file_hash_if_exists(self, raw_path: str) -> str | None:
        try:
            target = self._workspace_path(raw_path)
        except ValueError:
            return None
        if target.is_file() and not target.is_symlink():
            return sha256_bytes(target.read_bytes())
        return None

    def reconcile_write(self, raw_path, content, *, expected_sha256=None) -> dict:
        """Reconcile one journaled write intent by observable target content.

        A target already holding the intended bytes is never written again. The
        returned mapping includes ``tool_calls`` for exact budget accounting and
        ``reconciled`` when the landed effect was recognised without a write.
        """
        if not isinstance(content, str):
            return {'status': 'error', 'kind': 'write_file',
                    'error': 'unreconcilable_write_payload', 'tool_calls': 0}
        digest = sha256_bytes(content.encode('utf-8'))
        expected = expected_sha256 or digest
        if digest != expected:
            return {'status': 'error', 'kind': 'write_file', 'path': raw_path,
                    'error': 'content_hash_mismatch', 'tool_calls': 0}
        current = self.file_hash_if_exists(raw_path)
        if current is not None and current == expected:
            return {'status': 'ok', 'kind': 'write_file', 'path': raw_path,
                    'sha256': current, 'reconciled': True, 'tool_calls': 0}
        result = self.write_file({'path': raw_path, 'content': content})
        result = dict(result)
        if result.get('status') == 'ok' and result.get('sha256') != expected:
            result = dict(result, status='error', error='content_hash_mismatch')
        result['tool_calls'] = 1 if result.get('status') == 'ok' else 0
        result['reconciled'] = False
        return result


__all__ = ['CASE_FORMAT', 'FixtureWorkspace', 'KNOWN_TOOLS', 'MAX_READ_BYTES',
           'MAX_WRITE_BYTES', 'TEST_TIMEOUT_SECONDS', 'ToolRegistry', 'sanitized_env',
           'sha256_bytes']
