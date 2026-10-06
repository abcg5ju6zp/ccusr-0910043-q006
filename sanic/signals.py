from __future__ import annotations

import asyncio

from collections import deque
from contextvars import ContextVar
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
from sanic.signal_policy import (
    FailureMode,
    SignalDispatchError,
    SignalFailure,
    SignalPolicy,
    run_compensation,
)


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


def _name_of(handler: Any) -> str:
    handler = handler or repr(None)
    name = getattr(handler, "__qualname__", None) or getattr(
        handler, "__name__", None
    )
    return name or repr(handler)


# 嵌套内联派发通过该上下文变量复用最外层派发的一致视图；
# 后台任务形式的派发会在任务体内显式重置它，不继承调用方视图。
_current_view: ContextVar["SignalRouter._DispatchView | None"] = ContextVar(
    "sanic_signal_dispatch_view", default=None
)


def _reset_view(token: Any) -> None:
    """还原派发视图上下文变量。

    Sanic 的任务/请求协程可能在与其创建处不同的 context 中被驱动，
    对在别的 context 中取得的 token 调用 :meth:`ContextVar.reset` 会抛
    ``ValueError``。此时无需也无法还原（那是另一个 context），忽略即可。
    """
    try:
        _current_view.reset(token)
    except ValueError:
        pass


class Signal(Route):
    """项目内部接口说明。"""


@dataclass
class _ExecutionEntry:
    """单次派发执行计划中的一个快照条目。

    持有的是派发开始那一刻解析出的处理器与策略引用。派发进行期间即使
    监听器被动态替换（``route.handler`` 被改写），本次派发仍按快照执行，
    从而保证“一次派发的一致视图”。
    """

    signal: Signal
    handler: SignalHandler
    policy: SignalPolicy | None

    @property
    def name(self) -> str:
        return _name_of(self.handler)


# 旧式处理器的等效策略：任何异常都立即中断派发（既有行为）。
_LEGACY_POLICY = SignalPolicy(critical=True, failure=FailureMode.FAIL)

# 执行计划：参与执行的订阅者快照条目 + 用于通知 waiter 的全部信号快照。
_ExecutionPlan = tuple[tuple[_ExecutionEntry, ...], tuple[Signal, ...]]


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

    @dataclass
    class _DispatchView:
        """一次派发（含其内联嵌套派发）共享的一致视图。

        * 事件首次解析后，执行计划被缓存，嵌套派发复用同一快照；
        * 派发进行期间的监听器动态替换/增删不影响已解析出的计划；
        * 嵌套帧的故障被外层订阅者吸收时，仍保留在同一记录集合中，
          补偿失败之类的附加故障不会丢失或掩盖最初原因。
        """

        plans: dict[Any, _ExecutionPlan] = field(default_factory=dict)

    def _matches(self, signal: Signal, event: str, condition: Any) -> bool:
        requirements = signal.extra.requirements
        return (
            (condition is None and signal.ctx.exclusive is False)
            or (condition is None and not requirements)
            or (condition == requirements)
        ) and (signal.ctx.trigger or event == signal.ctx.definition)

    def _build_plan(
        self,
        view: _DispatchView,
        group: SignalGroup,
        event: str,
        condition: dict[str, str] | None,
        reverse: bool,
    ) -> _ExecutionPlan:
        from sanic.types import HashableDict

        key = (event, HashableDict(condition) if condition else None, reverse)
        cached = view.plans.get(key)
        if cached is not None:
            return cached

        # 快照 group.routes：amend() 重建路由器后，旧 group 仍被本计划持有。
        signals = cast(tuple[Signal, ...], tuple(group.routes))
        ordered = signals if reverse else signals[::-1]
        entries = [
            _ExecutionEntry(
                signal=signal,
                # 快照处理器与策略引用：派发途中替换 route.handler 或
                # 重写 ctx.policy 都不影响本计划。
                handler=cast(SignalHandler, signal.handler),
                policy=getattr(signal.ctx, "policy", None),
            )
            for signal in ordered
            if self._matches(signal, event, condition)
        ]
        # 显式声明的 order 在基础顺序（priority、注册序）之上做稳定排序；
        # 旧式处理器没有策略，order 视为 0，混合编排时相对次序仍可预测。
        entries.sort(
            key=lambda entry: entry.policy.order if entry.policy else 0,
            reverse=reverse,
        )
        plan = (tuple(entries), signals)
        view.plans[key] = plan
        return plan

    async def _notify_waiters(
        self,
        signals: tuple[Signal, ...],
        event: str,
        condition: dict[str, str] | None,
        params: dict[str, Any],
    ) -> None:
        # 与既有行为一致：在该事件组的全部信号上通知匹配的 waiter
        # （waiter 通过 name_index 挂在其中一个信号上，未必是本次条件
        # 匹配执行的那个），但遍历的是派发开始时的信号快照。请求取消期间
        # waiter 可能已注销/其 future 已结束，跳过它，不允许向已结束的
        # future 再写结果而污染本次派发。
        for signal in signals:
            for waiter in tuple(signal.ctx.waiters):
                future = waiter.future
                if (
                    waiter.matches(event, condition)
                    and future is not None
                    and not future.done()
                ):
                    future.set_result(dict(params))

    async def _report_failure(self, error: BaseException) -> None:
        """上报故障到 server.exception.report。

        上报本身失败只记录日志，绝不掩盖正在处理的原始故障。
        """
        try:
            await self.dispatch(
                Event.SERVER_EXCEPTION_REPORT.value,
                context={"exception": error},
            )
        except Exception:  # noqa: BLE001
            error_logger.exception(
                "Failed to report signal handler failure: %r", error
            )

    def _record_failure(
        self,
        failures: list[SignalFailure],
        entry: _ExecutionEntry,
        error: BaseException,
        *,
        phase: str = "handler",
        absorbed: bool = False,
    ) -> SignalFailure:
        policy = entry.policy or _LEGACY_POLICY
        if phase == "handler":
            name = entry.name
        else:
            name = _name_of(policy.compensation)
        failure = SignalFailure(
            handler=entry.handler,
            handler_name=name,
            event=entry.signal.ctx.definition,
            domain=policy.domain,
            critical=policy.critical,
            error=error,
            phase=phase,
            propagate=policy.propagate,
            absorbed=absorbed,
        )
        failures.append(failure)
        return failure

    def _root_cause(self, error: BaseException) -> BaseException:
        """剥开嵌套派发的聚合异常，递归找到最初原因。"""
        seen: set[int] = set()
        while isinstance(error, SignalDispatchError):
            if error.primary is None or id(error) in seen:
                break
            seen.add(id(error))
            error = error.primary
        return error

    def _aggregate(
        self,
        event: str,
        failures: list[SignalFailure],
        *,
        fail_fast: bool = False,
    ) -> SignalDispatchError:
        def _outward(f: SignalFailure) -> bool:
            if f.resolved or f.absorbed or not f.propagate:
                return False
            # ISOLATE/COMPENSATE 在帧末只让关键故障失败；
            # 显式 FAIL 的立即失败路径（fail_fast）则无论是否关键都抛出。
            return f.critical or (fail_fast and f.phase == "handler")

        propagating = [f for f in failures if _outward(f)]
        error = SignalDispatchError(event, propagating)
        if error.primary is not None:
            error.__cause__ = error.primary
        setattr(error, "__dispatched__", True)
        return error

    async def _invoke_policy_entry(
        self,
        entry: _ExecutionEntry,
        params: dict[str, Any],
        failures: list[SignalFailure],
        event: str,
    ) -> Any:
        """执行一个声明了策略的订阅者，并按其策略处置故障。

        ``CancelledError``（请求取消）不属于处理器故障：不补偿、不聚合，
        直接向上传播，本次派发的剩余计划不再执行。
        """
        policy = entry.policy
        assert policy is not None
        try:
            maybe_coroutine = entry.handler(**params)
            if isawaitable(maybe_coroutine):
                return await maybe_coroutine
            return maybe_coroutine
        except asyncio.CancelledError:
            raise
        except Exception as raised:  # noqa: BLE001
            # 嵌套内联派发抛出的聚合故障：内部记录作为细节保留，
            # 但由当前订阅者的策略统一处置，最初原因沿异常链向上传递。
            inner: tuple[SignalFailure, ...] = ()
            if isinstance(raised, SignalDispatchError):
                inner = raised.failures
                cause = self._root_cause(raised)
            else:
                cause = raised

            for inner_failure in inner:
                inner_failure.absorbed = True
                failures.append(inner_failure)

            failure = self._record_failure(failures, entry, cause)
            error_logger.warning(
                "Signal subscriber %r on %r failed in domain %r "
                "(critical=%s, mode=%s): %r",
                entry.name,
                event,
                policy.domain,
                policy.critical,
                policy.failure.value,
                cause,
            )
            # 聚合异常内的原始故障已在各自的帧内上报过，不重复上报；
            # 嵌套帧内旧式处理器抛出并冒泡上来的异常也已在该帧上报过。
            already_reported = getattr(cause, "__dispatched__", False)
            if not inner and not already_reported:
                setattr(cause, "__dispatched__", True)
                await self._report_failure(cause)

            if policy.failure is FailureMode.COMPENSATE:
                compensation_error = await run_compensation(
                    policy, params, cause
                )
                if compensation_error is None:
                    # 补偿成功：故障视为已解决，派发继续；
                    # 被吸收的嵌套故障一并视为解决。
                    failure.resolved = True
                    return None
                # 补偿失败：单独记录为附加故障，继承订阅者的关键级别，
                # 但永远不会成为最初原因。
                comp_failure = self._record_failure(
                    failures,
                    entry,
                    compensation_error,
                    phase="compensation",
                )
                error_logger.error(
                    "Compensation %r for signal subscriber %r failed; "
                    "the original failure is preserved as the cause",
                    comp_failure.handler_name,
                    entry.name,
                    exc_info=compensation_error,
                )
                setattr(compensation_error, "__dispatched__", True)
                await self._report_failure(compensation_error)
                if policy.critical:
                    # 关键订阅者补偿失败：无法保证安全，立即失败，
                    # 异常链上的原因仍是最初异常。
                    raise self._aggregate(event, failures) from cause
                # 非关键订阅者补偿失败：隔离后继续。
                return None

            if policy.failure is FailureMode.FAIL:
                # 显式选择“立即失败”：中断本次派发的剩余计划，
                # 无论是否关键都向外抛出（关键级别体现在故障记录中）。
                raise self._aggregate(
                    event, failures, fail_fast=True
                ) from cause

            # ISOLATE：隔离后继续执行同伴订阅者；若故障是关键的，
            # 不会被吞掉——帧末仍会聚合抛出，只是让同伴先执行完。
            return None

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
            group, _handlers, params = self.get(event, condition=condition)
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

        # 嵌套内联派发复用最外层派发的视图；后台任务派发不继承调用方视图。
        view = _current_view.get()
        token = None
        if view is None:
            view = self._DispatchView()
            token = _current_view.set(view)
        try:
            plan, all_signals = self._build_plan(
                view, group, event, condition, reverse
            )

            # 该事件上没有任何订阅者声明策略：逐字走既有派发路径，
            # 异常类型、上报次数与短路行为全部保持不变。
            if not any(entry.policy is not None for entry in plan):
                return await self._dispatch_legacy(
                    event, group, params, condition, reverse=reverse
                )

            await self._notify_waiters(all_signals, event, condition, params)

            failures: list[SignalFailure] = []
            retval: Any = None
            try:
                for entry in plan:
                    result: Any
                    if entry.policy is None:
                        # 旧式订阅者：抛出即中断，异常原样传播，
                        # 由下面的帧级兜底复刻旧有上报行为。
                        maybe_coroutine: Any = entry.handler(**params)
                        if isawaitable(maybe_coroutine):
                            result = await maybe_coroutine
                        else:
                            result = maybe_coroutine
                    else:
                        result = await self._invoke_policy_entry(
                            entry, params, failures, event
                        )
                    if result:
                        # 与旧行为一致：真值返回值短路后续订阅者。
                        retval = result
                        break
            except Exception as e:
                if self.ctx.app.debug and self.ctx.app.state.verbosity >= 1:
                    error_logger.exception(e)
                reported = getattr(e, "__dispatched__", False)
                if (
                    event != Event.SERVER_EXCEPTION_REPORT.value
                    and not reported
                    and not isinstance(e, SignalDispatchError)
                ):
                    await self._report_failure(e)
                    setattr(e, "__dispatched__", True)
                raise

            # 帧末聚合：关键 + ISOLATE 的故障允许同伴订阅者继续执行，
            # 但在派发（或嵌套派发帧）结束时仍使派发失败；
            # propagate=False 的故障被限制在其故障域/派发帧内，只留痕。
            unresolved = [
                f
                for f in failures
                if not f.resolved
                and not f.absorbed
                and f.propagate
                and f.critical
            ]
            if unresolved:
                raise self._aggregate(event, failures)
            return retval
        finally:
            if token is not None:
                _reset_view(token)

    async def _dispatch_legacy(
        self,
        event: str,
        group: SignalGroup,
        params: dict[str, Any],
        condition: dict[str, str] | None,
        reverse: bool = False,
    ) -> Any:
        """未声明策略时的原始派发路径，逐字保留既有行为。"""
        signals = group.routes
        if not reverse:
            signals = signals[::-1]
        try:
            for signal in signals:
                for waiter in signal.ctx.waiters:
                    if waiter.matches(event, condition):
                        waiter.future.set_result(dict(params))

            for signal in signals:
                requirements = signal.extra.requirements
                if (
                    (condition is None and signal.ctx.exclusive is False)
                    or (condition is None and not requirements)
                    or (condition == requirements)
                ) and (signal.ctx.trigger or event == signal.ctx.definition):
                    maybe_coroutine = signal.handler(**params)
                    if isawaitable(maybe_coroutine):
                        retval = await maybe_coroutine
                        if retval:
                            return retval
                    elif maybe_coroutine:
                        return maybe_coroutine
            return None
        except Exception as e:
            if self.ctx.app.debug and self.ctx.app.state.verbosity >= 1:
                error_logger.exception(e)

            if event != Event.SERVER_EXCEPTION_REPORT.value:
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
        logger.debug(f"Dispatching signal: {event}", extra={"verbosity": 1})

        if inline:
            return await self._dispatch(
                event,
                context=context,
                condition=condition,
                fail_not_found=fail_not_found and inline,
                reverse=reverse,
            )

        # create_task() 会复制创建处的 context。显式置空派发视图，
        # 使该派发及其内联嵌套派发拥有自己的一致视图，不挂在调用方派发上。
        async def dispatch_detached() -> Any:
            token = _current_view.set(None)
            try:
                return await self._dispatch(
                    event,
                    context=context,
                    condition=condition,
                    fail_not_found=fail_not_found and inline,
                    reverse=reverse,
                )
            finally:
                _reset_view(token)

        task = asyncio.get_running_loop().create_task(dispatch_detached())
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
        policy: SignalPolicy | None = None,
    ) -> Signal:
        event_definition = self.format_event(event)
        name, trigger, event_string = self._get_event_parts(event_definition)

        signal = super().add(
            event_string,
            handler,
            name=name,
            append=True,
            priority=priority,
        )  # type: ignore

        signal.ctx.exclusive = exclusive
        signal.ctx.trigger = trigger
        signal.ctx.definition = event_definition
        # 订阅者声明的派发策略（顺序/关键级别/故障域/故障处置/补偿）。
        # None 表示旧式处理器：派发时走既有路径，维持当前行为。
        signal.ctx.policy = policy
        signal.extra.requirements = condition

        return cast(Signal, signal)

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
