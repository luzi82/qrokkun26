"""Static production dependency closure, independent of .git and import history."""
import ast
import hashlib
from pathlib import Path


def source_digest():
    root=Path(__file__).resolve().parents[2]
    pending=['qrokkun_ai.v34.'+p.stem for p in Path(__file__).parent.glob('*.py')]
    found=set()
    while pending:
        module=pending.pop()
        path=root.joinpath(*module.split('.')).with_suffix('.py')
        if not path.is_file():
            path=root.joinpath(*module.split('.'),'__init__.py')
        if not path.is_file() or path in found:
            continue
        found.add(path)
        package=module if path.name=='__init__.py' else module.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):
                names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=node.module or ''
                if node.level:
                    parts=package.split('.')
                    prefix='.'.join(parts[:len(parts)-node.level+1]+([prefix] if prefix else []))
                names=[prefix]+[prefix+'.'+a.name for a in node.names]
            pending.extend(n for n in names if n.startswith(('qrokkun_ai','qrokkun_env')))
    found.update((root/'qrokkun_ai/v25/reference.json',Path(__file__).with_name('control-reference.json')))
    h=hashlib.sha256()
    for p in sorted(found):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()
