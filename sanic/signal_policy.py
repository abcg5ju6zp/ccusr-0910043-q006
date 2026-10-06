"""运行时信号订阅者的派发策略声明。

订阅者在注册信号处理器时可以声明：

* ``order``: 同一事件上的执行顺序（数值小的先执行）
* ``critical``: 关键级别，关键故障未被解决时整个派发失败
* ``domain``: 故障域，用于把故障归属到独立的隔离边界
* ``failure``: 故障模式

  - :attr:`FailureMode.FAIL`: 立即失败（关键订阅者会中断本次派发）
  - :attr:`FailureMode.ISOLATE`: 隔离故障，记录并上报后继续执行其余订阅者
  - :attr:`FailureMode.COMPENSATE`: 执行登记的补偿，补偿成功则视为已解决
* ``compensation``: 补偿处理器（仅 ``COMPENSATE`` 模式允许登记）
* ``propagate``: 故障是否向嵌套派发的外层帧传播（故障域边界）

没有声明策略的旧式信号处理器不经过本模块，维持既有的“抛出即中断”行为。
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from enum import Enum
from inspect import isawaitable
from typing import Any, Callable

from sanic.exceptions import InvalidSignal, SanicException


class FailureMode(str, Enum):
    """订阅者对自身故障的处置方式。"""

    FAIL = "fail"
    ISOLATE = "isolate"
    COMPENSATE = "compensate"


SignalCompensation = Callable[..., Any]


@dataclass(frozen=True)
class SignalPolicy:
    """订阅者对派发内核声明的策略。"""

    critical: bool = False
    failure: FailureMode = FailureMode.FAIL
    domain: str = "default"
    order: int = 0
    propagate: bool = True
    compensation: SignalCompensation | None = None

    def __post_init__(self) -> None:
        mode = self.failure
        if not isinstance(mode, FailureMode):
            try:
                mode = FailureMode(mode)
            except ValueError:
                raise InvalidSignal(
                    f"Unknown signal failure mode: {self.failure!r}. "
                    f"Expected one of: "
                    f"{', '.join(m.value for m in FailureMode)}"
                ) from None
        object.__setattr__(self, "failure", mode)

        if (
            self.compensation is not None
            and mode is not FailureMode.COMPENSATE
        ):
            raise InvalidSignal(
                "A compensation can only be registered with "
                "failure=FailureMode.COMPENSATE"
            )
        if mode is FailureMode.COMPENSATE and self.compensation is None:
            raise InvalidSignal(
                "failure=FailureMode.COMPENSATE requires a compensation "
                "handler to be registered"
            )

    @property
    def isolated(self) -> bool:
        return self.failure is FailureMode.ISOLATE

    @property
    def compensatory(self) -> bool:
        return self.failure is FailureMode.COMPENSATE


@dataclass
class SignalFailure:
    """一次派发中记录的单个故障。"""

    handler: Callable[..., Any]
    handler_name: str
    event: str
    domain: str
    critical: bool
    error: BaseException
    phase: str = "handler"
    resolved: bool = False
    propagate: bool = True
    absorbed: bool = False


class SignalDispatchError(SanicException):
    """一次派发结束后仍存在未解决的关键故障。

    最初的原因始终是 :attr:`primary`，同时挂在异常链的 ``__cause__`` 上；
    补偿阶段产生的错误只作为附加故障，永远不会掩盖最初原因。
    """

    def __init__(self, event: str, failures: list[SignalFailure]):
        self.event = event
        self.failures: tuple[SignalFailure, ...] = tuple(failures)
        self.handler_failures = tuple(
            f for f in self.failures if f.phase == "handler"
        )
        self.compensation_failures = tuple(
            f for f in self.failures if f.phase == "compensation"
        )
        primary_failure = next(
            (
                f
                for f in self.handler_failures
                if f.critical and not f.resolved
            ),
            self.handler_failures[0] if self.handler_failures else None,
        )
        self.primary_failure = primary_failure
        self.primary = primary_failure.error if primary_failure else None
        self.domains = tuple(dict.fromkeys(f.domain for f in self.failures))

        messages = [
            f"{f.handler_name} [{f.domain}] raised "
            f"{type(f.error).__name__}: {f.error}"
            + (" (compensation failed)" if f.phase == "compensation" else "")
            for f in self.failures
        ]
        super().__init__(
            f"Unresolved critical failure(s) dispatching signal {event!r}:\n"
            + "\n".join(f" - {message}" for message in messages)
        )


async def run_compensation(
    policy: SignalPolicy,
    params: dict[str, Any],
    error: BaseException,
) -> BaseException | None:
    """执行已登记的补偿。

    成功返回 ``None``；补偿自身失败时返回补偿错误，由调用方记录为
    ``phase="compensation"`` 的附加故障。补偿处理器可以通过声明 ``error``
    关键字形参来接收被补偿的原始异常。

    与普通处理器一样，``CancelledError`` 不属于故障，直接向上传播。
    """
    compensation = policy.compensation
    if compensation is None:
        return None

    try:
        from inspect import signature

        sig = signature(compensation)
    except (TypeError, ValueError):
        sig = None

    kwargs = dict(params)
    accepts_var_kwargs = False
    accepted_names: set[str] = set()
    if sig is not None:
        for parameter in sig.parameters.values():
            if parameter.kind == parameter.VAR_KEYWORD:
                accepts_var_kwargs = True
            elif parameter.kind in (
                parameter.POSITIONAL_OR_KEYWORD,
                parameter.KEYWORD_ONLY,
            ):
                accepted_names.add(parameter.name)
        if not accepts_var_kwargs:
            # 补偿处理器通常只关心少数上下文（甚至只要 error），
            # 按其声明的形参过滤，避免多余的派发上下文导致 TypeError。
            kwargs = {
                name: value
                for name, value in kwargs.items()
                if name in accepted_names
            }
        if "error" in accepted_names or accepts_var_kwargs:
            kwargs.setdefault("error", error)

    try:
        maybe = compensation(**kwargs)
        if isawaitable(maybe):
            await maybe
    except asyncio.CancelledError:
        raise
    except BaseException as compensation_error:  # noqa: BLE001
        return compensation_error
    return None
