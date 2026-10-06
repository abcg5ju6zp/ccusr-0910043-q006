import asyncio

from itertools import count

import pytest

from sanic import Blueprint, Sanic
from sanic.exceptions import InvalidSignal
from sanic.signals import (
    Event,
    SignalCriticality,
    SignalFailure,
    SignalFailureMode,
    SignalPolicy,
)


@pytest.mark.asyncio
async def test_non_critical_observer_failure_does_not_block_authorization(
    app: Sanic,
):
    """风控场景:非关键指标观察器抛错不应阻断授权记录写入。"""
    calls = []
    reported = []

    @app.signal("risk.request.received", order=1)
    async def risk_scorer():
        calls.append("score")

    @app.signal(
        "risk.request.received",
        order=2,
        criticality="non_critical",
        domain="metrics",
    )
    async def metrics_observer():
        calls.append("metrics")
        raise RuntimeError("metrics backend unavailable")

    @app.signal("risk.request.received", order=3)
    async def authorization_record_writer():
        calls.append("auth_record")

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report(exception):
        reported.append(exception)

    app.signal_router.finalize()
    await app.dispatch("risk.request.received", inline=True)
    for _ in range(5):
        await asyncio.sleep(0)

    assert calls == ["score", "metrics", "auth_record"]
    assert [str(e) for e in reported] == ["metrics backend unavailable"]


@pytest.mark.asyncio
async def test_order_declaration_controls_execution_sequence(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", order=3)
    async def third():
        calls.append("third")

    @app.signal("foo.bar.baz", order=1)
    async def first():
        calls.append("first")

    @app.signal("foo.bar.baz", order=2)
    async def second():
        calls.append("second")

    @app.signal("foo.bar.baz")
    async def legacy():
        calls.append("legacy")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    # order 越小越先执行;旧式 priority=0 先于所有 order > 0 的声明
    assert calls == ["legacy", "first", "second", "third"]


@pytest.mark.asyncio
async def test_order_composes_with_legacy_priority(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", priority=2)
    async def high_priority():
        calls.append("high_priority")

    @app.signal("foo.bar.baz", order=1)
    async def ordered():
        calls.append("ordered")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    # order = N 等价于 priority = -N,priority 大者先执行
    assert calls == ["high_priority", "ordered"]


@pytest.mark.asyncio
async def test_critical_subscriber_fails_immediately(app: Sanic):
    calls = []
    reported = []

    @app.signal("foo.bar.baz", order=1, criticality="critical")
    async def critical_handler():
        calls.append("critical")
        raise RuntimeError("critical boom")

    @app.signal("foo.bar.baz", order=2)
    async def after():
        calls.append("after")

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report(exception):
        reported.append(exception)

    app.signal_router.finalize()
    with pytest.raises(RuntimeError, match="critical boom"):
        await app.dispatch("foo.bar.baz", inline=True)
    for _ in range(5):
        await asyncio.sleep(0)

    assert calls == ["critical"]
    assert [str(e) for e in reported] == ["critical boom"]


@pytest.mark.asyncio
async def test_non_critical_defaults_to_isolate(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", order=1, criticality="non_critical")
    async def metrics():
        calls.append("metrics")
        raise RuntimeError("boom")

    @app.signal("foo.bar.baz", order=2)
    async def writer():
        calls.append("writer")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    assert calls == ["metrics", "writer"]


@pytest.mark.asyncio
async def test_failure_domain_isolation_skips_same_domain_only(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", order=1, on_failure="isolate", domain="metrics")
    async def metrics_one():
        calls.append("metrics_one")
        raise RuntimeError("boom")

    @app.signal("foo.bar.baz", order=2, domain="metrics")
    async def metrics_two():
        calls.append("metrics_two")

    @app.signal("foo.bar.baz", order=3, domain="auth")
    async def auth_handler():
        calls.append("auth_handler")

    @app.signal("foo.bar.baz", order=4)
    async def no_domain_handler():
        calls.append("no_domain_handler")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    # 同域订阅者随故障域一起被隔离,其他域与无域订阅者不受影响
    assert calls == ["metrics_one", "auth_handler", "no_domain_handler"]


@pytest.mark.asyncio
async def test_failed_domain_does_not_leak_between_dispatches(app: Sanic):
    calls = []
    should_fail = True

    @app.signal("foo.bar.baz", order=1, on_failure="isolate", domain="metrics")
    async def metrics_one():
        calls.append("metrics_one")
        if should_fail:
            raise RuntimeError("boom")

    @app.signal("foo.bar.baz", order=2, domain="metrics")
    async def metrics_two():
        calls.append("metrics_two")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)
    assert calls == ["metrics_one"]

    # 故障域状态只存在于单次派发,下一次派发重新开始
    should_fail = False
    await app.dispatch("foo.bar.baz", inline=True)
    assert calls == ["metrics_one", "metrics_one", "metrics_two"]


@pytest.mark.asyncio
async def test_compensate_runs_registered_compensation(app: Sanic):
    calls = []
    failures = []

    @app.signal("risk.auth.record", order=1, on_failure="compensate")
    async def auth_writer(amount):
        calls.append("auth_writer")
        raise RuntimeError("primary store down")

    @app.signal("risk.auth.record", order=2)
    async def audit(amount):
        calls.append("audit")

    @app.signal_compensation("risk.auth.record")
    async def fallback_store(failure):
        failures.append(failure)
        calls.append("compensate")

    app.signal_router.finalize()
    await app.dispatch("risk.auth.record", context={"amount": 9}, inline=True)

    assert calls == ["auth_writer", "compensate", "audit"]
    (failure,) = failures
    assert isinstance(failure, SignalFailure)
    assert failure.event == "risk.auth.record"
    assert failure.definition == "risk.auth.record"
    assert failure.criticality is SignalCriticality.CRITICAL
    assert failure.params["amount"] == 9
    assert str(failure.exception) == "primary store down"


@pytest.mark.asyncio
async def test_compensated_failure_does_not_close_domain(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", order=1, on_failure="compensate", domain="auth")
    async def writer():
        calls.append("writer")
        raise RuntimeError("boom")

    @app.signal("foo.bar.baz", order=2, domain="auth")
    async def same_domain_after():
        calls.append("same_domain_after")

    @app.signal_compensation("foo.bar.baz", domain="auth")
    async def compensation(failure):
        calls.append("compensated")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    # 补偿成功意味着失败已被处理,故障域不会像 ISOLATE 那样关闭
    assert calls == ["writer", "compensated", "same_domain_after"]


@pytest.mark.asyncio
async def test_compensation_domain_matching(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", order=1, on_failure="compensate", domain="auth")
    async def auth_writer():
        raise RuntimeError("auth boom")

    @app.signal_compensation("foo.bar.baz", domain="metrics")
    async def metrics_compensation(failure):
        calls.append("metrics_compensation")

    @app.signal_compensation("foo.bar.baz")
    async def event_compensation(failure):
        calls.append("event_compensation")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    # 域级补偿只匹配同域失败,事件级补偿总是匹配
    assert calls == ["event_compensation"]


@pytest.mark.asyncio
async def test_compensations_run_in_registration_order(app: Sanic):
    calls = []

    @app.signal("foo.bar.baz", on_failure="compensate")
    async def writer():
        raise RuntimeError("boom")

    @app.signal_compensation("foo.bar.baz")
    async def first_compensation(failure):
        calls.append("first")

    @app.signal_compensation("foo.bar.baz")
    async def second_compensation(failure):
        calls.append("second")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    assert calls == ["first", "second"]


@pytest.mark.asyncio
async def test_compensate_without_registration_propagates(app: Sanic):
    @app.signal("foo.bar.baz", on_failure="compensate")
    async def writer():
        raise RuntimeError("boom")

    app.signal_router.finalize()
    with pytest.raises(RuntimeError, match="boom"):
        await app.dispatch("foo.bar.baz", inline=True)


@pytest.mark.asyncio
async def test_compensation_failure_does_not_mask_original_cause(app: Sanic):
    reported = []

    @app.signal("foo.bar.baz", on_failure="compensate")
    async def writer():
        raise ValueError("original cause")

    @app.signal_compensation("foo.bar.baz")
    async def bad_compensation(failure):
        raise RuntimeError("compensation exploded")

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report(exception):
        reported.append(exception)

    app.signal_router.finalize()
    with pytest.raises(ValueError, match="original cause") as excinfo:
        await app.dispatch("foo.bar.baz", inline=True)
    for _ in range(5):
        await asyncio.sleep(0)

    error = excinfo.value
    assert [str(e) for e in error.__compensation_errors__] == [
        "compensation exploded"
    ]
    # 上报链路看到的仍是最初的异常,而不是补偿异常
    assert [str(e) for e in reported] == ["original cause"]


@pytest.mark.asyncio
async def test_legacy_handler_behavior_unchanged(app: Sanic):
    calls = []
    reported = []

    @app.signal("foo.bar.baz")
    async def first():
        calls.append("first")
        raise RuntimeError("boom")

    @app.signal("foo.bar.baz")
    async def second():
        calls.append("second")

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report(exception):
        reported.append(exception)

    app.signal_router.finalize()
    with pytest.raises(RuntimeError, match="boom"):
        await app.dispatch("foo.bar.baz", inline=True)
    for _ in range(5):
        await asyncio.sleep(0)

    # 未声明策略:立即失败、后续订阅者中断、异常照常上报
    assert calls == ["first"]
    assert [str(e) for e in reported] == ["boom"]
    signal = app.signal_router.name_index["foo.bar.baz"]
    assert signal.ctx.policy is None


@pytest.mark.asyncio
async def test_conditional_dispatch_consistent_view(app: Sanic):
    calls = []

    @app.signal(
        "foo.bar.baz",
        condition={"tenant": "a"},
        order=1,
        criticality="non_critical",
        domain="metrics",
    )
    async def tenant_a_metrics():
        calls.append("a_metrics")
        raise RuntimeError("boom")

    @app.signal("foo.bar.baz", condition={"tenant": "a"}, order=2)
    async def tenant_a_writer():
        calls.append("a_writer")

    @app.signal("foo.bar.baz", condition={"tenant": "b"}, order=1)
    async def tenant_b_metrics():
        calls.append("b_metrics")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", condition={"tenant": "a"}, inline=True)
    assert calls == ["a_metrics", "a_writer"]

    # 条件过滤在派发开始时完成,租户 b 的视图不受租户 a 失败影响
    await app.dispatch("foo.bar.baz", condition={"tenant": "b"}, inline=True)
    assert calls == ["a_metrics", "a_writer", "b_metrics"]


@pytest.mark.asyncio
async def test_nested_dispatch_frames_are_independent(app: Sanic):
    calls = []

    @app.signal(
        "inner.event.here", order=1, on_failure="isolate", domain="shared"
    )
    async def inner_one():
        calls.append("inner_one")
        raise RuntimeError("inner boom")

    @app.signal("inner.event.here", order=2, domain="shared")
    async def inner_two():
        calls.append("inner_two")

    @app.signal("outer.event.here", order=1)
    async def outer_one():
        calls.append("outer_one")
        await app.dispatch("inner.event.here", inline=True)

    @app.signal("outer.event.here", order=2, domain="shared")
    async def outer_two():
        calls.append("outer_two")

    app.signal_router.finalize()
    await app.dispatch("outer.event.here", inline=True)

    # 内层派发的故障域状态不会泄漏到外层帧
    assert calls == ["outer_one", "inner_one", "outer_two"]


@pytest.mark.asyncio
async def test_nested_propagated_failure_uses_outer_policy(app: Sanic):
    calls = []

    @app.signal("inner.event.here")
    async def inner_legacy():
        raise RuntimeError("inner legacy boom")

    @app.signal("outer.event.here", order=1, criticality="non_critical")
    async def outer_one():
        calls.append("outer_one")
        await app.dispatch("inner.event.here", inline=True)

    @app.signal("outer.event.here", order=2)
    async def outer_two():
        calls.append("outer_two")

    app.signal_router.finalize()
    await app.dispatch("outer.event.here", inline=True)

    # 内层旧式失败向外传播后,按外层订阅者的策略被隔离
    assert calls == ["outer_one", "outer_two"]


@pytest.mark.asyncio
async def test_nested_dispatch_reports_exception_once(app: Sanic):
    reported = []

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report(exception):
        reported.append(exception)

    @app.signal("inner.event.here")
    async def inner_legacy():
        raise RuntimeError("boom")

    @app.signal("outer.event.here")
    async def outer_legacy():
        await app.dispatch("inner.event.here", inline=True)

    app.signal_router.finalize()
    with pytest.raises(RuntimeError, match="boom"):
        await app.dispatch("outer.event.here", inline=True)
    for _ in range(5):
        await asyncio.sleep(0)

    # 同一个异常在嵌套派发中只上报一次
    assert len(reported) == 1


@pytest.mark.asyncio
async def test_dynamic_listener_replacement_keeps_dispatch_snapshot(
    app: Sanic,
):
    calls = []
    replaced = False

    async def replacement():
        calls.append("replacement")

    @app.signal("foo.bar.baz", order=1)
    async def first():
        nonlocal replaced
        calls.append("first")
        if not replaced:
            replaced = True
            # 派发中途动态替换监听器(框架的 reset/add/finalize 流程)
            app.signal_router.reset()
            app.add_signal(replacement, "foo.bar.baz")
            app.signal_router.finalize()

    @app.signal("foo.bar.baz", order=2)
    async def second():
        calls.append("second")

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    # 进行中的派发仍持有派发开始时的订阅者快照
    assert calls == ["first", "second"]

    # 下一次派发看到替换后的集合(replacement 以默认 priority 最先执行)
    await app.dispatch("foo.bar.baz", inline=True)
    assert calls == ["first", "second", "replacement", "first", "second"]


@pytest.mark.asyncio
async def test_compensation_registered_mid_dispatch_not_visible(app: Sanic):
    calls = []
    registered = False

    @app.signal("foo.bar.baz", order=1, on_failure="compensate")
    async def failing():
        nonlocal registered
        calls.append("failing")
        if not registered:
            registered = True
            app.add_signal_compensation(late_compensation, "foo.bar.baz")
        raise RuntimeError("boom")

    @app.signal("foo.bar.baz", order=2)
    async def after():
        calls.append("after")

    async def late_compensation(failure):
        calls.append("compensated")

    app.signal_router.finalize()

    # 本次派发的快照中没有补偿,声明了 COMPENSATE 但未登记 → 立即失败
    with pytest.raises(RuntimeError, match="boom"):
        await app.dispatch("foo.bar.baz", inline=True)
    assert calls == ["failing"]

    # 后续派发可以看到新登记的补偿
    await app.dispatch("foo.bar.baz", inline=True)
    assert calls == ["failing", "failing", "compensated", "after"]


@pytest.mark.asyncio
async def test_cancellation_skips_compensation_and_keeps_view(app: Sanic):
    calls = []
    started = asyncio.Event()
    first_run = True

    @app.signal("foo.bar.baz", order=1, on_failure="compensate")
    async def slow():
        nonlocal first_run
        if first_run:
            started.set()
            await asyncio.sleep(10)
        calls.append("slow")

    @app.signal("foo.bar.baz", order=2)
    async def after():
        calls.append("after")

    @app.signal_compensation("foo.bar.baz")
    async def compensation(failure):
        calls.append("compensated")

    app.signal_router.finalize()
    task = await app.dispatch("foo.bar.baz")
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 取消不是订阅者失败:不触发隔离,也不执行补偿
    assert calls == []

    # 被取消的派发不留下任何帧状态,后续派发行为一致
    first_run = False
    await app.dispatch("foo.bar.baz", inline=True)
    assert calls == ["slow", "after"]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_break_dispatch(app: Sanic):
    counter = count()

    @app.signal("foo.bar.baz")
    def sync_signal(*_):
        next(counter)

    app.signal_router.finalize()

    # 模拟请求取消后 future 已取消、但 waiter 尚未移出队列的瞬间
    waiter = app.signal_router.get_waiter("foo.bar.baz")
    loop = asyncio.get_running_loop()
    waiter.future = loop.create_future()
    signal = app.signal_router.name_index["foo.bar.baz"]
    signal.ctx.waiters.append(waiter)
    waiter.future.cancel()

    await app.dispatch("foo.bar.baz", inline=True)
    assert next(counter) == 1


@pytest.mark.asyncio
async def test_report_event_failure_is_not_reported_recursively(app: Sanic):
    calls = []

    @app.signal(Event.SERVER_EXCEPTION_REPORT, criticality="non_critical")
    async def report_one(exception):
        calls.append("one")
        raise RuntimeError("report handler boom")

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report_two(exception):
        calls.append("two")

    app.signal_router.finalize()
    await app.dispatch(
        Event.SERVER_EXCEPTION_REPORT.value,
        context={"exception": RuntimeError("x")},
        inline=True,
    )

    # 上报事件自身的隔离失败只记录日志,不会递归触发上报
    assert calls == ["one", "two"]


@pytest.mark.asyncio
async def test_isolated_failure_carries_failure_metadata(app: Sanic):
    seen = []

    @app.signal(
        "foo.bar.baz",
        criticality="non_critical",
        domain="metrics",
    )
    async def metrics():
        raise RuntimeError("boom")

    @app.signal(Event.SERVER_EXCEPTION_REPORT)
    async def report(exception):
        seen.append(exception)

    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)
    for _ in range(5):
        await asyncio.sleep(0)

    (error,) = seen
    failure = error.__signal_failure__
    assert failure.domain == "metrics"
    assert failure.criticality is SignalCriticality.NON_CRITICAL
    assert failure.definition == "foo.bar.baz"


@pytest.mark.asyncio
async def test_add_signal_with_policy_kwargs(app: Sanic):
    calls = []

    def metrics():
        calls.append("metrics")
        raise RuntimeError("boom")

    def writer():
        calls.append("writer")

    app.add_signal(metrics, "foo.bar.baz", order=1, criticality="non_critical")
    app.add_signal(writer, "foo.bar.baz", order=2)
    app.signal_router.finalize()
    await app.dispatch("foo.bar.baz", inline=True)

    assert calls == ["metrics", "writer"]


@pytest.mark.asyncio
async def test_blueprint_signal_policy_passthrough(app: Sanic):
    bp = Blueprint("bp")
    calls = []

    @bp.signal("foo.bar.baz", order=1, criticality="non_critical")
    async def bp_metrics():
        calls.append("bp_metrics")
        raise RuntimeError("boom")

    @bp.signal("foo.bar.baz", order=2)
    async def bp_writer():
        calls.append("bp_writer")

    app.blueprint(bp)
    app.signal_router.finalize()
    await bp.dispatch("foo.bar.baz")

    assert calls == ["bp_metrics", "bp_writer"]


@pytest.mark.asyncio
async def test_blueprint_signal_compensation(app: Sanic):
    bp = Blueprint("bp")
    calls = []

    @bp.signal("foo.bar.baz", on_failure="compensate")
    async def bp_writer():
        calls.append("bp_writer")
        raise RuntimeError("boom")

    @bp.signal_compensation("foo.bar.baz")
    async def bp_compensation(failure):
        calls.append("compensated")

    app.blueprint(bp)
    app.signal_router.finalize()
    await bp.dispatch("foo.bar.baz")

    assert calls == ["bp_writer", "compensated"]


def test_policy_string_coercion(app: Sanic):
    @app.signal(
        "foo.bar.baz", criticality="non_critical", on_failure="isolate"
    )
    def sync_signal(*_): ...

    signal = app.signal_router.name_index["foo.bar.baz"]
    policy = signal.ctx.policy
    assert policy is not None
    assert policy.criticality is SignalCriticality.NON_CRITICAL
    assert policy.on_failure is SignalFailureMode.ISOLATE


def test_invalid_policy_values(app: Sanic):
    with pytest.raises(InvalidSignal, match="Invalid signal criticality"):
        app.signal("foo.bar.baz", criticality="sometimes")

    with pytest.raises(InvalidSignal, match="Invalid signal failure mode"):
        app.signal("foo.bar.baz", on_failure="explode")


def test_policy_object_cannot_mix_with_kwargs(app: Sanic):
    policy = SignalPolicy(order=1)
    with pytest.raises(InvalidSignal, match="Cannot combine policy="):
        app.signal_router.add(
            lambda: ..., "foo.bar.baz", policy=policy, order=2
        )


def test_policy_object_applied_directly(app: Sanic):
    policy = SignalPolicy(
        order=2,
        criticality=SignalCriticality.NON_CRITICAL,
        domain="metrics",
        on_failure=SignalFailureMode.ISOLATE,
    )
    signal = app.signal_router.add(lambda: ..., "foo.bar.baz", policy=policy)
    assert signal.ctx.policy is policy
    assert signal.priority == -2
