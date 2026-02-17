"""Decorator-based RBAC controls for SDN security operations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable


@dataclass(frozen=True)
class UserContext:
    """Authenticated operator context."""

    username: str
    role: str
    hsm_pin: str


def require_role(required_role: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Ensure the caller has the expected role and a validated HSM PIN.

    The decorated function must receive ``user`` and expose an HSM wrapper (``hsm`` kwarg or ``self.hsm``).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user: UserContext = kwargs["user"]
            hsm = kwargs.get("hsm") or getattr(args[0], "hsm", None)
            if hsm is None:
                raise ValueError("HSM reference not found for RBAC PIN validation.")
            if user.role != required_role:
                raise PermissionError(
                    f"{user.username} has role {user.role}; required role is {required_role}."
                )
            if not hsm.verify_user_pin(user.hsm_pin):
                raise PermissionError("HSM User PIN verification failed.")
            return func(*args, **kwargs)

        return wrapper

    return decorator
