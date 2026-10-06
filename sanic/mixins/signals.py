from __future__ import annotations

from collections.abc import Coroutine
from enum import Enum
from typing import Any, Callable

from sanic.base.meta import SanicMeta
from sanic.exceptions import SanicException
from sanic.models.futures import FutureSignal
from sanic.models.handler_types import SignalHandler
from sanic.signal_policy import FailureMode, SignalPolicy
from sanic.signals import Event, Signal
from sanic.types import HashableDict


class SignalMixin(metaclass=SanicMeta):
    def __init__(self, *args, **kwargs) -> None:
        self._future_signals: set[FutureSignal] = set()

    def _apply_signal(self, signal: FutureSignal) -> Signal:
        raise NotImplementedError  # noqa

    def signal(
        self,
        event: str | Enum,
        *,
        apply: bool = True,
        condition: dict[str, Any] | None = None,
        exclusive: bool = True,
        priority: int = 0,
        policy: SignalPolicy | None = None,
        critical: bool | None = None,
        failure: FailureMode | str | None = None,
        domain: str | None = None,
        order: int | None = None,
        propagate: bool | None = None,
        compensation: Callable[..., Any] | None = None,
    ) -> Callable[[SignalHandler], SignalHandler]:
        """登记一个运行时信号订阅者。

        除既有的 ``condition``/``exclusive``/``priority`` 外，还可以声明
        派发策略：

        :param critical: 关键级别。关键故障未被解决时，派发以失败结束。
        :param failure: 故障模式：``FailureMode.FAIL``（立即失败）、
            ``FailureMode.ISOLATE``（隔离故障后继续其余订阅者）、
            ``FailureMode.COMPENSATE``（执行已登记的补偿）。
        :param domain: 故障域名称，用于归属与隔离边界。
        :param order: 同一事件上的执行顺序，数值小的先执行；
            缺省时遵循 priority/注册序。
        :param propagate: 故障是否跨越嵌套派发边界向外传播；
            ``False`` 时故障被限制在所属派发帧/故障域内。
        :param compensation: 补偿处理器，仅 ``COMPENSATE`` 模式可登记。
        :param policy: 直接传入一个 :class:`SignalPolicy`；
            不可与上面的策略关键字同时使用。

        所有策略参数缺省时注册的是旧式订阅者，维持既有的
        “处理器抛错即中断整个派发”行为。
        """
        resolved_policy = self._resolve_policy(
            policy,
            critical=critical,
            failure=failure,
            domain=domain,
            order=order,
            propagate=propagate,
            compensation=compensation,
        )
        event_value = str(event.value) if isinstance(event, Enum) else event

        def decorator(handler: SignalHandler):
            future_signal = FutureSignal(
                handler,
                event_value,
                HashableDict(condition or {}),
                exclusive,
                priority,
                resolved_policy,
            )
            self._future_signals.add(future_signal)

            if apply:
                self._apply_signal(future_signal)

            return handler

        return decorator

    @staticmethod
    def _resolve_policy(
        policy: SignalPolicy | None,
        *,
        critical: bool | None,
        failure: FailureMode | str | None,
        domain: str | None,
        order: int | None,
        propagate: bool | None,
        compensation: Callable[..., Any] | None,
    ) -> SignalPolicy | None:
        overrides = {
            key: value
            for key, value in (
                ("critical", critical),
                ("failure", failure),
                ("domain", domain),
                ("order", order),
                ("propagate", propagate),
                ("compensation", compensation),
            )
            if value is not None
        }
        if policy is not None:
            if overrides:
                raise SanicException(
                    "Cannot pass both a policy object and individual policy "
                    f"keyword arguments: {', '.join(overrides)}"
                )
            return policy
        if not overrides:
            # 旧式处理器：不声明任何策略，维持当前行为。
            return None
        return SignalPolicy(**overrides)  # type: ignore[arg-type]

    def add_signal(
        self,
        handler: Callable[..., Any] | None,
        event: str | Enum,
        condition: dict[str, Any] | None = None,
        exclusive: bool = True,
        *,
        policy: SignalPolicy | None = None,
        critical: bool | None = None,
        failure: FailureMode | str | None = None,
        domain: str | None = None,
        order: int | None = None,
        propagate: bool | None = None,
        compensation: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """项目内部接口说明。"""
        if not handler:

            async def noop(**context): ...

            handler = noop
        self.signal(
            event=event,
            condition=condition,
            exclusive=exclusive,
            policy=policy,
            critical=critical,
            failure=failure,
            domain=domain,
            order=order,
            propagate=propagate,
            compensation=compensation,
        )(handler)
        return handler

    def replace_signal(
        self,
        previous: Callable[..., Any],
        handler: Callable[..., Any],
        *,
        policy: SignalPolicy | None = None,
        critical: bool | None = None,
        failure: FailureMode | str | None = None,
        domain: str | None = None,
        order: int | None = None,
        propagate: bool | None = None,
        compensation: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """动态替换一个已登记的信号订阅者。

        ``previous`` 是此前注册时返回的处理器对象。替换立即生效；已经在
        派发途中的事件仍使用开始时快照到的旧处理器与旧策略执行完，
        之后的派发才使用新处理器，从而保证一次派发的一致视图。

        新策略缺省时沿用被替换订阅者的策略。
        """
        router = getattr(self, "signal_router", None)
        if router is None:  # 蓝图在挂载前没有自己的路由器
            raise SanicException(
                "Cannot replace a signal subscriber before the application "
                "has been created; dynamic replacement is available on a "
                "running Sanic application"
            )

        target = next(
            (route for route in router.routes if route.handler is previous),
            None,
        )
        if target is None:
            raise SanicException(
                "Cannot replace a signal subscriber that is not registered"
            )

        new_policy = self._resolve_policy(
            policy,
            critical=critical,
            failure=failure,
            domain=domain,
            order=order,
            propagate=propagate,
            compensation=compensation,
        )
        target.handler = handler
        if new_policy is not None:
            target.ctx.policy = new_policy
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
