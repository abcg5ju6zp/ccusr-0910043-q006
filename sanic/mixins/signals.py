from __future__ import annotations

from collections.abc import Coroutine
from enum import Enum
from typing import Any, Callable

from sanic.base.meta import SanicMeta
from sanic.models.futures import FutureSignal, FutureSignalCompensation
from sanic.models.handler_types import SignalHandler
from sanic.signals import (
    Event,
    Signal,
    SignalCriticality,
    SignalFailureMode,
    SignalPolicy,
)
from sanic.types import HashableDict


class SignalMixin(metaclass=SanicMeta):
    def __init__(self, *args, **kwargs) -> None:
        self._future_signals: set[FutureSignal] = set()
        self._future_signal_compensations: set[FutureSignalCompensation] = (
            set()
        )

    def _apply_signal(self, signal: FutureSignal) -> Signal:
        raise NotImplementedError  # noqa

    def _apply_signal_compensation(
        self, compensation: FutureSignalCompensation
    ) -> Any:
        raise NotImplementedError  # noqa

    def signal(
        self,
        event: str | Enum,
        *,
        apply: bool = True,
        condition: dict[str, Any] | None = None,
        exclusive: bool = True,
        priority: int = 0,
        order: int | None = None,
        criticality: SignalCriticality | str | None = None,
        domain: str | None = None,
        on_failure: SignalFailureMode | str | None = None,
    ) -> Callable[[SignalHandler], SignalHandler]:
        """项目内部接口说明。"""
        event_value = str(event.value) if isinstance(event, Enum) else event
        policy = SignalPolicy.build(
            order=order,
            criticality=criticality,
            domain=domain,
            on_failure=on_failure,
        )

        def decorator(handler: SignalHandler):
            future_signal = FutureSignal(
                handler,
                event_value,
                HashableDict(condition or {}),
                exclusive,
                priority,
                policy,
            )
            self._future_signals.add(future_signal)

            if apply:
                self._apply_signal(future_signal)

            return handler

        return decorator

    def add_signal(
        self,
        handler: Callable[..., Any] | None,
        event: str | Enum,
        condition: dict[str, Any] | None = None,
        exclusive: bool = True,
        priority: int = 0,
        *,
        order: int | None = None,
        criticality: SignalCriticality | str | None = None,
        domain: str | None = None,
        on_failure: SignalFailureMode | str | None = None,
    ) -> Callable[..., Any]:
        """项目内部接口说明。"""
        if not handler:

            async def noop(**context): ...

            handler = noop
        self.signal(
            event=event,
            condition=condition,
            exclusive=exclusive,
            priority=priority,
            order=order,
            criticality=criticality,
            domain=domain,
            on_failure=on_failure,
        )(handler)
        return handler

    def signal_compensation(
        self,
        event: str | Enum,
        *,
        domain: str | None = None,
        apply: bool = True,
    ) -> Callable[[SignalHandler], SignalHandler]:
        """为事件登记补偿处理器的装饰器。

        补偿处理器接收唯一的 ``SignalFailure`` 入参,在声明了
        ``on_failure=COMPENSATE`` 的订阅者失败时按登记顺序执行。
        """
        event_value = str(event.value) if isinstance(event, Enum) else event

        def decorator(handler: SignalHandler):
            future_compensation = FutureSignalCompensation(
                handler,
                event_value,
                domain,
            )
            self._future_signal_compensations.add(future_compensation)

            if apply:
                self._apply_signal_compensation(future_compensation)

            return handler

        return decorator

    def add_signal_compensation(
        self,
        handler: Callable[..., Any],
        event: str | Enum,
        domain: str | None = None,
    ) -> Callable[..., Any]:
        """项目内部接口说明。"""
        self.signal_compensation(event=event, domain=domain)(handler)
        return handler

    def event(self, event: str):
        raise NotImplementedError

    def catch_exception(
        self,
        handler: Callable[[SignalMixin, Exception], Coroutine[Any, Any, None]],
    ) -> None:
        """项目内部接口说明。"""

        async def signal_handler(exception: Exception):
            await handler(self, exception)

        self.signal(Event.SERVER_EXCEPTION_REPORT)(signal_handler)
