from __future__ import annotations

import asyncio

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from inspect import isawaitable
from typing import Any, cast

from sanic_routing import BaseRouter, Route, RouteGroup
from sanic_routing.exceptions import NotFound
from sanic_routing.utils import path_to_parts

from sanic.exceptions import InvalidSignal
from sanic.log import error_logger, logger
from sanic.models.handler_types import SignalHandler


class Event(Enum):
    """项目内部接口说明。"""

    SERVER_EXCEPTION_REPORT = "server.exception.report"
    SERVER_INIT_AFTER = "server.init.after"
    SERVER_INIT_BEFORE = "server.init.before"
    SERVER_SHUTDOWN_AFTER = "server.shutdown.after"
    SERVER_SHUTDOWN_BEFORE = "server.shutdown.before"
    HTTP_LIFECYCLE_BEGIN = "http.lifecycle.begin"
    HTTP_LIFECYCLE_COMPLETE = "http.lifecycle.complete"
    HTTP_LIFECYCLE_EXCEPTION = "http.lifecycle.exception"
    HTTP_LIFECYCLE_HANDLE = "http.lifecycle.handle"
    HTTP_LIFECYCLE_READ_BODY = "http.lifecycle.read_body"
    HTTP_LIFECYCLE_READ_HEAD = "http.lifecycle.read_head"
    HTTP_LIFECYCLE_REQUEST = "http.lifecycle.request"
    HTTP_LIFECYCLE_RESPONSE = "http.lifecycle.response"
    HTTP_ROUTING_AFTER = "http.routing.after"
    HTTP_ROUTING_BEFORE = "http.routing.before"
    HTTP_HANDLER_AFTER = "http.handler.after"
    HTTP_HANDLER_BEFORE = "http.handler.before"
    HTTP_LIFECYCLE_SEND = "http.lifecycle.send"
    HTTP_MIDDLEWARE_AFTER = "http.middleware.after"
    HTTP_MIDDLEWARE_BEFORE = "http.middleware.before"
    WEBSOCKET_HANDLER_AFTER = "websocket.handler.after"
    WEBSOCKET_HANDLER_BEFORE = "websocket.handler.before"
    WEBSOCKET_HANDLER_EXCEPTION = "websocket.handler.exception"


RESERVED_NAMESPACES = {
    "server": (
        Event.SERVER_EXCEPTION_REPORT.value,
        Event.SERVER_INIT_AFTER.value,
        Event.SERVER_INIT_BEFORE.value,
        Event.SERVER_SHUTDOWN_AFTER.value,
        Event.SERVER_SHUTDOWN_BEFORE.value,
    ),
    "http": (
        Event.HTTP_LIFECYCLE_BEGIN.value,
        Event.HTTP_LIFECYCLE_COMPLETE.value,
        Event.HTTP_LIFECYCLE_EXCEPTION.value,
        Event.HTTP_LIFECYCLE_HANDLE.value,
        Event.HTTP_LIFECYCLE_READ_BODY.value,
        Event.HTTP_LIFECYCLE_READ_HEAD.value,
        Event.HTTP_LIFECYCLE_REQUEST.value,
        Event.HTTP_LIFECYCLE_RESPONSE.value,
        Event.HTTP_ROUTING_AFTER.value,
        Event.HTTP_ROUTING_BEFORE.value,
        Event.HTTP_HANDLER_AFTER.value,
        Event.HTTP_HANDLER_BEFORE.value,
        Event.HTTP_LIFECYCLE_SEND.value,
        Event.HTTP_MIDDLEWARE_AFTER.value,
        Event.HTTP_MIDDLEWARE_BEFORE.value,
    ),
    "websocket": {
        Event.WEBSOCKET_HANDLER_AFTER.value,
        Event.WEBSOCKET_HANDLER_BEFORE.value,
        Event.WEBSOCKET_HANDLER_EXCEPTION.value,
    },
}

GENERIC_SIGNAL_FORMAT = "__generic__.__signal__.%s"


def _blank(): ...


class Signal(Route):
    """项目内部接口说明。"""


class SignalFailureMode(Enum):
    """订阅者抛出异常时,单次派发应当如何应对。"""

    PROPAGATE = "propagate"
    """立即失败:中断本次派发并把异常抛给派发方(旧式默认行为)。"""

    ISOLATE = "isolate"
    """隔离后继续:把失败限制在其故障域内,其余订阅者继续执行。"""

    COMPENSATE = "compensate"
    """执行已登记的补偿:先运行该事件登记的补偿处理器,再继续派发。"""


class SignalCriticality(Enum):
    """订阅者对于一次派发整体成败的关键级别。"""

    CRITICAL = "critical"
    NON_CRITICAL = "non_critical"


@dataclass(frozen=True)
class SignalPolicy:
    """信号订阅者声明的执行顺序、关键级别、故障域与失败策略。

    未声明任何策略字段的旧式订阅者不会得到 ``SignalPolicy``
    (``signal.ctx.policy is None``),派发行为与历史版本完全一致。

    ``order`` 越小越先执行;它与旧式 ``priority`` 共用同一排序通道
    (``order = N`` 等价于 ``priority = -N``,而 ``priority``
    大者先执行),因此两者混用时顺序依然可预期。
    """

    order: int = 0
    criticality: SignalCriticality = SignalCriticality.CRITICAL
    domain: str | None = None
    on_failure: SignalFailureMode = SignalFailureMode.PROPAGATE

    @classmethod
    def build(
        cls,
        *,
        order: int | None = None,
        criticality: SignalCriticality | str | None = None,
        domain: str | None = None,
        on_failure: SignalFailureMode | str | None = None,
    ) -> SignalPolicy | None:
        """根据声明构建策略;什么都没声明时返回 ``None``(旧式行为)。"""
        if (
            order is None
            and criticality is None
            and domain is None
            and on_failure is None
        ):
            return None

        crit = cls._coerce_criticality(criticality)
        mode = cls._coerce_mode(on_failure)
        if mode is None:
            # 非关键订阅者默认隔离失败,关键订阅者默认立即失败
            mode = (
                SignalFailureMode.ISOLATE
                if crit is SignalCriticality.NON_CRITICAL
                else SignalFailureMode.PROPAGATE
            )
        return cls(
            order=order or 0,
            criticality=crit,
            domain=domain,
            on_failure=mode,
        )

    @staticmethod
    def _coerce_criticality(
        value: SignalCriticality | str | None,
    ) -> SignalCriticality:
        if value is None:
            return SignalCriticality.CRITICAL
        if isinstance(value, SignalCriticality):
            return value
        try:
            return SignalCriticality(str(value))
        except ValueError:
            raise InvalidSignal(
                "Invalid signal criticality: %s. Must be one of: %s"
                % (
                    value,
                    ", ".join(member.value for member in SignalCriticality),
                )
            ) from None

    @staticmethod
    def _coerce_mode(
        value: SignalFailureMode | str | None,
    ) -> SignalFailureMode | None:
        if value is None or isinstance(value, SignalFailureMode):
            return value
        try:
            return SignalFailureMode(str(value))
        except ValueError:
            raise InvalidSignal(
                "Invalid signal failure mode: %s. Must be one of: %s"
                % (
                    value,
                    ", ".join(member.value for member in SignalFailureMode),
                )
            ) from None


@dataclass(frozen=True)
class SignalCompensation:
    """针对某个事件(可选地限定故障域)登记的补偿处理器。"""

    handler: SignalHandler
    domain: str | None = None


@dataclass
class SignalFailure:
    """派发给补偿处理器的故障上下文。

    补偿处理器以此对象为唯一入参,避免与派发参数发生关键字冲突;
    该对象同时挂载到原始异常的 ``__signal_failure__`` 属性上,
    供异常上报链路观察。
    """

    exception: Exception
    event: str
    definition: str
    domain: str | None
    criticality: SignalCriticality
    params: dict[str, Any]


@dataclass
class _DispatchEntry:
    """一次派发中待执行的订阅者及其已解析策略。"""

    signal: Signal
    policy: SignalPolicy | None


@dataclass
class _DispatchFrame:
    """一次派发的一致视图。

    在派发开始时对订阅者列表(条件过滤后)与补偿登记表做快照,
    派发期间的故障域状态只保存在帧上。嵌套派发各自持有独立的帧,
    监听器动态替换不会影响正在进行的派发。
    """

    event: str
    entries: list[_DispatchEntry]
    compensations: dict[str, tuple[SignalCompensation, ...]]
    failed_domains: set[str] = field(default_factory=set)

    def compensations_for(self, entry: _DispatchEntry):
        registered = self.compensations.get(entry.signal.ctx.definition, ())
        domain = entry.policy.domain if entry.policy else None
        return tuple(
            compensation
            for compensation in registered
            if compensation.domain is None or compensation.domain == domain
        )


@dataclass
class SignalWaiter:
    """项目内部接口说明。"""

    signal: Signal
    event_definition: str
    trigger: str = ""
    requirements: dict[str, str] | None = None
    exclusive: bool = True

    future: asyncio.Future | None = None

    async def wait(self):
        """项目内部接口说明。"""
        loop = asyncio.get_running_loop()
        self.future = loop.create_future()
        self.signal.ctx.waiters.append(self)
        try:
            return await self.future
        finally:
            self.signal.ctx.waiters.remove(self)

    def matches(self, event, condition):
        return (
            (condition is None and not self.exclusive)
            or (condition is None and not self.requirements)
            or condition == self.requirements
        ) and (self.trigger or event == self.event_definition)


class SignalGroup(RouteGroup):
    """项目内部接口说明。"""


class SignalRouter(BaseRouter):
    """项目内部接口说明。"""

    def __init__(self) -> None:
        super().__init__(
            delimiter=".",
            route_class=Signal,
            group_class=SignalGroup,
            stacking=True,
        )
        self.allow_fail_builtin = True
        self.ctx.loop = None
        self.ctx.compensations: dict[str, tuple[SignalCompensation, ...]] = {}

    @staticmethod
    def format_event(event: str | Enum) -> str:
        """项目内部接口说明。"""
        if isinstance(event, Enum):
            event = str(event.value)
        if "." not in event:
            event = GENERIC_SIGNAL_FORMAT % event
        return event

    def get(  # type: ignore
        self,
        event: str | Enum,
        condition: dict[str, str] | None = None,
    ):
        """项目内部接口说明。"""
        event = self.format_event(event)
        extra = condition or {}
        try:
            group, param_basket = self.find_route(
                f".{event}",
                self.DEFAULT_METHOD,
                self,
                {"__params__": {}, "__matches__": {}},
                extra=extra,
            )
        except NotFound:
            message = "Could not find signal %s"
            terms: list[str | dict[str, str] | None] = [event]
            if extra:
                message += " with %s"
                terms.append(extra)
            raise NotFound(message % tuple(terms))

        # Regex routes evaluate and can extract params directly. They are set
        # on param_basket["__params__"]
        params = param_basket["__params__"]
        if not params:
            # If param_basket["__params__"] does not exist, we might have
            # param_basket["__matches__"], which are indexed based matches
            # on path segments. They should already be cast types.
            params = {
                param.name: param_basket["__matches__"][idx]
                for idx, param in group.params.items()
            }

        return group, [route.handler for route in group], params

    def _resolve_entries(
        self,
        signals,
        event: str,
        condition: dict[str, str] | None,
    ) -> list[_DispatchEntry]:
        """按当前条件过滤订阅者并快照成本次派发的执行条目。"""
        entries: list[_DispatchEntry] = []
        for signal in signals:
            requirements = signal.extra.requirements
            if (
                (condition is None and signal.ctx.exclusive is False)
                or (condition is None and not requirements)
                or (condition == requirements)
            ) and (signal.ctx.trigger or event == signal.ctx.definition):
                entries.append(
                    _DispatchEntry(
                        signal=signal,
                        policy=getattr(signal.ctx, "policy", None),
                    )
                )
        return entries

    async def _report_failure(self, frame: _DispatchFrame, error: Exception):
        """上报被隔离/被补偿的异常,使其可见但不再向外抛出。"""
        if self.ctx.app.debug and self.ctx.app.state.verbosity >= 1:
            error_logger.exception(error)

        if frame.event != Event.SERVER_EXCEPTION_REPORT.value:
            await self.dispatch(
                Event.SERVER_EXCEPTION_REPORT.value,
                context={"exception": error},
            )
            setattr(error, "__dispatched__", True)

    async def _run_compensations(
        self,
        compensations: tuple[SignalCompensation, ...],
        failure: SignalFailure,
    ) -> list[Exception]:
        """尽力执行全部补偿,收集补偿自身的失败而不中断后续补偿。"""
        compensation_errors: list[Exception] = []
        for compensation in compensations:
            try:
                maybe_coroutine = compensation.handler(failure)
                if isawaitable(maybe_coroutine):
                    await maybe_coroutine
            except Exception as compensation_error:
                error_logger.exception(
                    "Compensation failed for signal event %s: %s",
                    failure.event,
                    compensation_error,
                )
                compensation_errors.append(compensation_error)
        return compensation_errors

    async def _handle_subscriber_failure(
        self,
        frame: _DispatchFrame,
        entry: _DispatchEntry,
        error: Exception,
        params: dict[str, Any],
    ) -> None:
        """按订阅者声明的策略处理失败;需要立即失败时重新抛出原始异常。"""
        policy = entry.policy
        if policy is None or policy.on_failure is SignalFailureMode.PROPAGATE:
            raise error

        failure = SignalFailure(
            exception=error,
            event=frame.event,
            definition=entry.signal.ctx.definition,
            domain=policy.domain,
            criticality=policy.criticality,
            params=dict(params),
        )
        setattr(error, "__signal_failure__", failure)

        if policy.on_failure is SignalFailureMode.COMPENSATE:
            compensations = frame.compensations_for(entry)
            if not compensations:
                error_logger.warning(
                    "Signal %s declared compensation but none is registered "
                    "for event %s",
                    failure.definition,
                    failure.event,
                )
                raise error
            compensation_errors = await self._run_compensations(
                compensations, failure
            )
            if compensation_errors:
                # 补偿失败不得掩盖最初原因:重新抛出原始异常,
                # 补偿异常仅作为附注与属性保留
                setattr(error, "__compensation_errors__", compensation_errors)
                add_note = getattr(error, "add_note", None)
                if add_note is not None:
                    for compensation_error in compensation_errors:
                        add_note(
                            "Signal compensation failed: "
                            f"{compensation_error!r}"
                        )
                raise error
            await self._report_failure(frame, error)
            return

        # ISOLATE:把失败限制在故障域内,同域其余订阅者本次跳过
        if policy.domain is not None:
            frame.failed_domains.add(policy.domain)
        await self._report_failure(frame, error)

    async def _dispatch(
        self,
        event: str,
        context: dict[str, Any] | None = None,
        condition: dict[str, str] | None = None,
        fail_not_found: bool = True,
        reverse: bool = False,
    ) -> Any:
        event = self.format_event(event)
        try:
            group, handlers, params = self.get(event, condition=condition)
        except NotFound as e:
            is_reserved = event.split(".", 1)[0] in RESERVED_NAMESPACES
            if fail_not_found and (not is_reserved or self.allow_fail_builtin):
                raise e
            else:
                if self.ctx.app.debug and self.ctx.app.state.verbosity >= 1:
                    error_logger.warning(str(e))
                return None

        if context:
            params.update(context)
        params.pop("__trigger__", None)

        signals = group.routes
        if not reverse:
            signals = signals[::-1]

        # 一次派发的一致视图:订阅者与补偿登记表在派发开始时快照,
        # 之后的动态替换只影响后续派发
        frame = _DispatchFrame(
            event=event,
            entries=self._resolve_entries(signals, event, condition),
            compensations=self.ctx.compensations,
        )

        try:
            for signal in signals:
                for waiter in tuple(signal.ctx.waiters):
                    if (
                        waiter.future is not None
                        and not waiter.future.done()
                        and waiter.matches(event, condition)
                    ):
                        waiter.future.set_result(dict(params))

            for entry in frame.entries:
                policy = entry.policy
                if (
                    policy is not None
                    and policy.domain is not None
                    and policy.domain in frame.failed_domains
                ):
                    continue
                try:
                    maybe_coroutine = entry.signal.handler(**params)
                    if isawaitable(maybe_coroutine):
                        retval = await maybe_coroutine
                        if retval:
                            return retval
                    elif maybe_coroutine:
                        return maybe_coroutine
                except Exception as e:
                    await self._handle_subscriber_failure(
                        frame, entry, e, params
                    )
            return None
        except Exception as e:
            if self.ctx.app.debug and self.ctx.app.state.verbosity >= 1:
                error_logger.exception(e)

            if event != Event.SERVER_EXCEPTION_REPORT.value and not getattr(
                e, "__dispatched__", False
            ):
                await self.dispatch(
                    Event.SERVER_EXCEPTION_REPORT.value,
                    context={"exception": e},
                )
                setattr(e, "__dispatched__", True)
            raise e

    async def dispatch(
        self,
        event: str | Enum,
        *,
        context: dict[str, Any] | None = None,
        condition: dict[str, str] | None = None,
        fail_not_found: bool = True,
        inline: bool = False,
        reverse: bool = False,
    ) -> asyncio.Task | Any:
        """项目内部接口说明。"""

        event = self.format_event(event)
        dispatch = self._dispatch(
            event,
            context=context,
            condition=condition,
            fail_not_found=fail_not_found and inline,
            reverse=reverse,
        )
        logger.debug(f"Dispatching signal: {event}", extra={"verbosity": 1})

        if inline:
            return await dispatch

        task = asyncio.get_running_loop().create_task(dispatch)
        await asyncio.sleep(0)
        return task

    def get_waiter(
        self,
        event: str | Enum,
        condition: dict[str, Any] | None = None,
        exclusive: bool = True,
    ) -> SignalWaiter | None:
        event_definition = self.format_event(event)
        name, trigger, _ = self._get_event_parts(event_definition)
        signal = cast(Signal, self.name_index.get(name))
        if not signal:
            return None

        if event_definition.endswith(".*") and not trigger:
            trigger = "*"
        return SignalWaiter(
            signal=signal,
            event_definition=event_definition,
            trigger=trigger,
            requirements=condition,
            exclusive=bool(exclusive),
        )

    def _get_event_parts(self, event: str) -> tuple[str, str, str]:
        parts = self._build_event_parts(event)
        if parts[2].startswith("<"):
            name = ".".join([*parts[:-1], "*"])
            trigger = self._clean_trigger(parts[2])
        else:
            name = event
            trigger = ""

        if not trigger:
            event = ".".join([*parts[:2], "<__trigger__>"])

        return name, trigger, event

    def add(  # type: ignore
        self,
        handler: SignalHandler,
        event: str | Enum,
        condition: dict[str, Any] | None = None,
        exclusive: bool = True,
        *,
        priority: int = 0,
        order: int | None = None,
        criticality: SignalCriticality | str | None = None,
        domain: str | None = None,
        on_failure: SignalFailureMode | str | None = None,
        policy: SignalPolicy | None = None,
    ) -> Signal:
        event_definition = self.format_event(event)
        name, trigger, event_string = self._get_event_parts(event_definition)

        if policy is not None and any(
            declared is not None
            for declared in (order, criticality, domain, on_failure)
        ):
            raise InvalidSignal(
                "Cannot combine policy= with order/criticality/domain/"
                "on_failure declarations"
            )
        if policy is None:
            policy = SignalPolicy.build(
                order=order,
                criticality=criticality,
                domain=domain,
                on_failure=on_failure,
            )
            effective_priority = -order if order is not None else priority
        else:
            effective_priority = -policy.order if policy.order else priority

        signal = super().add(
            event_string,
            handler,
            name=name,
            append=True,
            priority=effective_priority,
        )  # type: ignore

        signal.ctx.exclusive = exclusive
        signal.ctx.trigger = trigger
        signal.ctx.definition = event_definition
        signal.ctx.policy = policy
        signal.extra.requirements = condition

        return cast(Signal, signal)

    def add_compensation(
        self,
        handler: SignalHandler,
        event: str | Enum,
        domain: str | None = None,
    ) -> SignalCompensation:
        """为事件登记补偿处理器。

        登记表写时复制:正在进行的派发仍持有派发开始时的快照,
        新登记的补偿只影响后续派发。
        """
        event_definition = self.format_event(event)
        compensation = SignalCompensation(handler=handler, domain=domain)
        registry = dict(self.ctx.compensations)
        registry[event_definition] = registry.get(event_definition, ()) + (
            compensation,
        )
        self.ctx.compensations = registry
        return compensation

    def finalize(self, do_compile: bool = True, do_optimize: bool = False):
        """项目内部接口说明。"""
        self.add(_blank, "sanic.__signal__.__init__")

        try:
            self.ctx.loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("Cannot finalize signals outside of event loop")

        for signal in self.routes:
            signal.ctx.waiters = deque()

        return super().finalize(do_compile=do_compile, do_optimize=do_optimize)

    def _build_event_parts(self, event: str) -> tuple[str, str, str]:
        parts = path_to_parts(event, self.delimiter)
        if (
            len(parts) != 3
            or parts[0].startswith("<")
            or parts[1].startswith("<")
        ):
            raise InvalidSignal("Invalid signal event: %s" % event)

        if (
            parts[0] in RESERVED_NAMESPACES
            and event not in RESERVED_NAMESPACES[parts[0]]
            and not (parts[2].startswith("<") and parts[2].endswith(">"))
        ):
            raise InvalidSignal(
                "Cannot declare reserved signal event: %s" % event
            )
        return parts

    def _clean_trigger(self, trigger: str) -> str:
        trigger = trigger[1:-1]
        if ":" in trigger:
            trigger, _ = trigger.split(":")
        return trigger
