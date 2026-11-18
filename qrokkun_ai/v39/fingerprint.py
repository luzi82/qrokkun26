"""Explicit entrypoint closure including dynamic hook dependencies."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def source_files():
    pending = ['qrokkun_ai.v39.'+p.stem for p in Path(__file__).parent.glob('*.py')]
    # prepare_control dynamically enumerates every v37/v38 production entrypoint.
    from qrokkun_ai.v37.fingerprint import source_files as old_source_files
    found = set(old_source_files())
    # Traverse those entrypoints too, rather than treating them as already visited.
    pending.extend('.'.join(p.relative_to(ROOT).with_suffix('').parts) for p in found if p.suffix == '.py')
    old_closure = found
    found = set()
    while pending:
        module = pending.pop()
        path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        if not path.is_file(): path = ROOT.joinpath(*module.split('.'), '__init__.py')
        if not path.is_file() or path in found or 'tests' in path.parts: continue
        found.add(path)
        package = module if path.name == '__init__.py' else module.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import): names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level:
                    prefix = '.'.join(package.split('.')[:len(package.split('.'))-node.level+1]+([prefix] if prefix else []))
                names = [prefix]+[prefix+'.'+a.name for a in node.names]
            pending.extend(n for n in names if n.startswith(('qrokkun_ai', 'qrokkun_env')))
    found.update(old_closure)
    found.update(ROOT/p for p in ('qrokkun_ai/v25/reference.json', 'qrokkun_ai/v39/contract.json'))
    return sorted(found)

def source_digest():
    h = hashlib.sha256()
    for path in source_files():
        h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    return h.hexdigest()
