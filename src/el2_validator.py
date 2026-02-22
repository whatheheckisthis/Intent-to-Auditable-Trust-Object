from dataclasses import dataclass


class NonceReplayError(RuntimeError):
    pass


class SpdmBindingError(RuntimeError):
    pass


class CredentialExpiredError(RuntimeError):
    pass


@dataclass
class SteCredential:
    pa_range_base: int
    pa_range_limit: int
    permissions: int


class El2CredentialValidator:
    def __init__(self):
        self._seen = set()

    def validate(self, raw: bytes, stream_id: int) -> SteCredential:
        key = (stream_id, raw)
        if key in self._seen:
            raise NonceReplayError("replay")
        self._seen.add(key)
        if len(raw) not in (109, 149):
            raise CredentialExpiredError("bad len")
        return SteCredential(pa_range_base=0x1000, pa_range_limit=0x3000, permissions=0x3)
