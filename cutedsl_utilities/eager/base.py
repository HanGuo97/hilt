import torch
from typing import Self


class CuTeEager(object):

    def get_metadata(self) -> dict[str, object]:
        raise NotImplementedError

    def get_annotation(self, name: str | None = None) -> str:
        if name is None:
            name = type(self).__name__
        metadata_list = []
        for key, value in self.get_metadata().items():
            if isinstance(value, torch.Size):
                value = tuple(value)
            metadata_list.append(f"{key}={value}")
        metadata = ", ".join(metadata_list)
        return f"{name}[{metadata}]"

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        raise NotImplementedError

    @classmethod
    def from_struct(cls: type[Self], struct: object) -> Self:
        assert isinstance(struct, cls.get_struct_type())
        return cls(**struct._asdict())

    def to_struct(self) -> object:
        struct_cls = self.get_struct_type()
        metadata = self.get_metadata()
        return struct_cls(**metadata)

    def __str__(self) -> str:
        return self.get_annotation()

    def __repr__(self) -> str:
        from .core import cute_apply
        return cute_apply()(str)(self)
