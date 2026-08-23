from pathlib import Path

from .registry import ENVIRONMENT_REGISTRY, make_environment, register_environment

# Auto-discover and register all environments
root = Path(__file__).parent
discovered = [(root, __package__)]
seen_dirs = {root.resolve()}

# Find all Python files that contain "register_environment"
for py in root.rglob("*.py"):
    try:
        if "register_environment" not in py.read_text(encoding="utf-8"):
            continue
    except Exception:
        continue

    rel = py.parent.relative_to(root)
    pkg = __package__ + ("" if not rel.parts else "." + ".".join(rel.parts))
    dir_path = py.parent.resolve()

    if dir_path not in seen_dirs:
        discovered.append((dir_path, pkg))
        seen_dirs.add(dir_path)

# Trigger discovery (imports all environment modules)
ENVIRONMENT_REGISTRY.discover(discovered)

__all__ = ["ENVIRONMENT_REGISTRY", "make_environment", "register_environment"]
