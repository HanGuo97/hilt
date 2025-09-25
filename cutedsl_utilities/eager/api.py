import inspect
import cutlass.cute as cute
from .core import cute_apply


_exported_attrs = []
for name in dir(cute):

    if name.startswith("_"):
        continue

    attr = getattr(cute, name)
    if inspect.isfunction(attr):
        attr = cute_apply()(attr)

    globals()[name] = attr
    _exported_attrs.append(name)

__all__ = _exported_attrs
