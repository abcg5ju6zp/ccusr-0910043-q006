"""运行时信号订阅者派发策略（顺序/关键级别/故障域/补偿）的回归测试。

覆盖的核心保证：

* 非关键观察器故障可以被隔离，不再阻断关键的授权记录写入；
* 同一事件可以选择立即失败、隔离后继续或执行已登记的补偿；
* 条件路由、嵌套派发、监听器动态替换、请求取消时保持一次派发的一致视图；
* 补偿失败不掩盖最初原因；
* 没有声明策略的旧式处理器维持当前行为。
"""

from __future__ import annotations

import asyncio

import pytest

from sanic import Blueprint, Sanic, empty
from sanic.exceptions import InvalidSignal
from sanic.signal_policy import (
    FailureMode,
    SignalDispatchError,
    SignalPolicy,
)


RISK_EVENT = "risk.eval.done"
AUX_EVENT = "risk.aux.done"


def _finalize(app: Sanic) -> None:
    app.signal_router.finalize()


# --------------------------------------------------------------------------- #
# 风控主场景
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_noncritical_isolate_does_not_block_critical_writer(app):
    """非关键指标观察器抛错后，授权记录写入与其余观察者照常完成。"""
    ran = []

    @app.signal(RISK_EVENT, order=0)
    async def authz_writer(**_):
        # 关键的授权记录写入
        ran.append("authz")

    @app.signal(
        RISK_EVENT,
        critical=False,
        failure=FailureMode.ISOLATE,
    )
    async def metrics_observer(**_):
        raise RuntimeError("metrics backend down")

    @app.signal(RISK_EVENT, order=2)
    async def audit_observer(**_):
        ran.append("audit")

    _finalize(app)
    await app.dispatch(RISK_EVENT, inline=True)

    assert ran == ["authz", "audit"]


def test_risk_scenario_end_to_end(app: Sanic):
    """真实 HTTP 请求中，指标观察器故障不影响授权写入与响应。"""
    authorized = []

    @app.signal(
        "risk.signal.observed",
        critical=False,
        failure=FailureMode.ISOLATE,
    )
    async def metrics(**_):
        raise RuntimeError("metrics backend down")

    @app.signal(
        "risk.signal.observed",
        critical=True,
        failure=FailureMode.FAIL,
    )
    async def write_authorization(request, **_):
        authorized.append(request.path)

    @app.post("/authorize")
    async def authorize(request):
        await app.dispatch(
            "risk.signal.observed",
            inline=True,
            context={"request": request},
        )
        return empty()

    _, response = app.test_client.post("/authorize")
    assert response.status == 204
    assert authorized == ["/authorize"]


# --------------------------------------------------------------------------- #
# 三种故障模式
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_critical_fail_fast_aborts_remaining_subscribers(app):
    ran = []

    @app.signal(RISK_EVENT, critical=True, failure=FailureMode.FAIL)
    async def blocker(**_):
        ran.append("blocker")
        raise ValueError("blocked")

    @app.signal(RISK_EVENT, order=1)
    async def after(**_):
        ran.append("after")

    _finalize(app)
    with pytest.raises(SignalDispatchError) as excinfo:
        await app.dispatch(RISK_EVENT, inline=True)

    assert ran == ["blocker"]
    error = excinfo.value
    assert isinstance(error.primary, ValueError)
    assert str(error.primary) == "blocked"
    # 最初原因同时挂在异常链上
    assert isinstance(error.__cause__, ValueError)


@pytest.mark.asyncio
async def test_noncritical_fail_still_aborts_dispatch(app):
    """显式选择 FAIL 时，即使非关键也立即失败，但同伴不再执行。"""
    ran = []

    @app.signal(RISK_EVENT, critical=False, failure=FailureMode.FAIL)
    async def blocker(**_):
        raise LookupError("missing")

    @app.signal(RISK_EVENT, order=1)
    async def after(**_):
        ran.append("after")

    _finalize(app)
    with pytest.raises(SignalDispatchError) as excinfo:
        await app.dispatch(RISK_EVENT, inline=True)

    assert ran == []
    assert isinstance(excinfo.value.primary, LookupError)


@pytest.mark.asyncio
async def test_critical_isolate_runs_peers_then_fails(app):
    ran = []

    @app.signal(RISK_EVENT, critical=True, failure=FailureMode.ISOLATE)
    async def isolated(**_):
        ran.append("isolated")
        raise RuntimeError("contained but critical")

    @app.signal(RISK_EVENT, order=1)
    async def peer(**_):
        ran.append("peer")

    _finalize(app)
    with pytest.raises(SignalDispatchError) as excinfo:
        await app.dispatch(RISK_EVENT, inline=True)

    # 同伴先执行完，派发结束时仍因关键故障而失败
    assert ran == ["isolated", "peer"]
    assert isinstance(excinfo.value.primary, RuntimeError)


@pytest.mark.asyncio
async def test_successful_compensation_resolves_failure(app):
    ran = []

    async def rollback(error, **_):
        ran.append(("compensated", type(error).__name__))

    @app.signal(
        RISK_EVENT,
        critical=True,
        failure=FailureMode.COMPENSATE,
        compensation=rollback,
    )
    async def flaky(**_):
        raise KeyError("boom")

    @app.signal(RISK_EVENT, order=1)
    async def tail(**_):
        ran.append("tail")

    _finalize(app)
    await app.dispatch(RISK_EVENT, inline=True)

    assert ran == [("compensated", "KeyError"), "tail"]


@pytest.mark.asyncio
async def test_compensation_failure_preserves_original_cause(app):
    async def bad_compensation(error, **_):
        raise RuntimeError("compensation broken")

    @app.signal(
        RISK_EVENT,
        critical=True,
        failure=FailureMode.COMPENSATE,
        compensation=bad_compensation,
    )
    async def original(**_):
        raise ValueError("original failure")

    _finalize(app)
    with pytest.raises(SignalDispatchError) as excinfo:
        await app.dispatch(RISK_EVENT, inline=True)

    error = excinfo.value
    # 最初原因始终是原始处理器异常，而不是补偿异常
    assert isinstance(error.primary, ValueError)
    assert str(error.primary) == "original failure"
    assert len(error.compensation_failures) == 1
    assert isinstance(error.compensation_failures[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_compensation_receives_only_declared_context(app):
    seen = {}

    async def compensation(error):
        # 只声明 error：多余的派发上下文不应导致 TypeError
        seen["error"] = error

    @app.signal(
        RISK_EVENT,
        critical=True,
        failure=FailureMode.COMPENSATE,
        compensation=compensation,
    )
    async def handler(**_):
        raise RuntimeError("x")

    _finalize(app)
    await app.dispatch(
        RISK_EVENT, inline=True, context={"amount": 9, "request": None}
    )
    assert isinstance(seen["error"], RuntimeError)


# --------------------------------------------------------------------------- #
# 旧式处理器保持当前行为
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_legacy_handler_preserves_raw_exception(app):
    @app.signal(RISK_EVENT)
    async def legacy(**_):
        raise RuntimeError("raw")

    @app.signal(RISK_EVENT, order=1)
    async def after(**_):
        raise AssertionError("must not run after legacy failure")

    _finalize(app)
    # 不是聚合异常，而是原始异常直接冒泡
    with pytest.raises(RuntimeError, match="raw"):
        await app.dispatch(RISK_EVENT, inline=True)


@pytest.mark.asyncio
async def test_failure_is_reported_once(app):
    reports = []

    @app.report_exception
    async def catch(app, exception):
        reports.append(exception)

    @app.signal(
        RISK_EVENT,
        critical=False,
        failure=FailureMode.ISOLATE,
    )
    async def isolated(**_):
        raise RuntimeError("reported")

    _finalize(app)
    await app.dispatch(RISK_EVENT, inline=True)

    assert len(reports) == 1
    assert isinstance(reports[0], RuntimeError)


# --------------------------------------------------------------------------- #
# 声明顺序
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_explicit_order_overrides_registration_order(app):
    ran = []

    @app.signal(RISK_EVENT, order=10)
    async def late(**_):
        ran.append("late")

    @app.signal(RISK_EVENT, order=-10)
    async def early(**_):
        ran.append("early")

    @app.signal(RISK_EVENT, order=0)
    async def middle(**_):
        ran.append("middle")

    _finalize(app)
    await app.dispatch(RISK_EVENT, inline=True)
    assert ran == ["early", "middle", "late"]


@pytest.mark.asyncio
async def test_mixed_legacy_and_policy_ordering(app):
    ran = []

    @app.signal(RISK_EVENT)
    async def legacy(**_):
        ran.append("legacy")

    @app.signal(RISK_EVENT, order=-1)
    async def ordered(**_):
        ran.append("ordered")

    _finalize(app)
    await app.dispatch(RISK_EVENT, inline=True)
    assert ran == ["ordered", "legacy"]


# --------------------------------------------------------------------------- #
# 条件路由
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_policy_respects_condition_routing(app):
    ran = []

    @app.signal(
        RISK_EVENT,
        condition={"kind": "authz"},
        critical=False,
        failure=FailureMode.ISOLATE,
    )
    async def authz_observer(**_):
        ran.append("authz")
        raise RuntimeError("ignored")

    @app.signal(RISK_EVENT, condition={"kind": "other"})
    async def other_observer(**_):
        ran.append("other")

    _finalize(app)
    await app.dispatch(RISK_EVENT, inline=True, condition={"kind": "other"})
    assert ran == ["other"]

    ran.clear()
    await app.dispatch(RISK_EVENT, inline=True, condition={"kind": "authz"})
    assert ran == ["authz"]


# --------------------------------------------------------------------------- #
# 嵌套派发与故障域
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nested_failure_absorbed_by_outer_policy(app):
    ran = []

    @app.signal(AUX_EVENT, critical=True, failure=FailureMode.FAIL)
    async def inner_fail(**_):
        raise ValueError("inner boom")

    @app.signal(
        RISK_EVENT,
        order=0,
        critical=True,
        failure=FailureMode.ISOLATE,
    )
    async def outer_caller(**_):
        ran.append("before-inner")
        await app.dispatch(AUX_EVENT, inline=True)

    @app.signal(RISK_EVENT, order=1)
    async def outer_tail(**_):
        ran.append("tail")

    _finalize(app)
    with pytest.raises(SignalDispatchError) as excinfo:
        await app.dispatch(RISK_EVENT, inline=True)

    assert ran == ["before-inner", "tail"]
    # 最初原因是内层异常，沿嵌套帧一路保留
    assert isinstance(excinfo.value.primary, ValueError)
    assert str(excinfo.value.primary) == "inner boom"


@pytest.mark.asyncio
async def test_propagate_false_contains_failure_in_domain(app):
    ran = []

    @app.signal(
        AUX_EVENT,
        critical=True,
        failure=FailureMode.ISOLATE,
        domain="aux-domain",
        propagate=False,
    )
    async def contained(**_):
        ran.append("contained")
        raise RuntimeError("stay inside")

    @app.signal(RISK_EVENT, order=0)
    async def caller(**_):
        await app.dispatch(AUX_EVENT, inline=True)
        ran.append("continued")

    @app.signal(RISK_EVENT, order=1)
    async def tail(**_):
        ran.append("tail")

    _finalize(app)
    # 故障被限制在 aux 派发帧内，外层派发成功
    await app.dispatch(RISK_EVENT, inline=True)
    assert ran == ["contained", "continued", "tail"]


@pytest.mark.asyncio
async def test_nested_legacy_failure_still_propagates(app):
    @app.signal(AUX_EVENT)
    async def inner_legacy(**_):
        raise RuntimeError("legacy inner")

    @app.signal(RISK_EVENT)
    async def caller(**_):
        await app.dispatch(AUX_EVENT, inline=True)

    _finalize(app)
    with pytest.raises(RuntimeError, match="legacy inner"):
        await app.dispatch(RISK_EVENT, inline=True)


# --------------------------------------------------------------------------- #
# 一致视图：动态替换与请求取消
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_replacement_uses_snapshot_for_inflight_dispatch(app):
    log = []

    async def replacement(**_):
        log.append("new")

    @app.signal(
        RISK_EVENT,
        critical=False,
        failure=FailureMode.ISOLATE,
    )
    async def dynamic(**_):
        log.append("old")
        # 派发途中替换自己
        app.replace_signal(dynamic, replacement)

    @app.signal(RISK_EVENT, order=1)
    async def peer(**_):
        log.append("peer")

    _finalize(app)
    # 第一次派发：快照仍指向旧处理器
    await app.dispatch(RISK_EVENT, inline=True)
    assert log == ["old", "peer"]

    # 第二次派发：使用替换后的处理器
    await app.dispatch(RISK_EVENT, inline=True)
    assert log == ["old", "peer", "new", "peer"]


@pytest.mark.asyncio
async def test_blueprint_policy_is_applied_and_isolated(app):
    bp = Blueprint("risk_bp")
    ran = []

    @bp.signal(
        RISK_EVENT,
        order=1,
        critical=False,
        failure=FailureMode.ISOLATE,
    )
    async def metrics(**_):
        ran.append("metrics")
        raise RuntimeError("bp metrics down")

    @bp.signal(RISK_EVENT, order=0)
    async def writer(**_):
        ran.append("writer")

    app.blueprint(bp)
    _finalize(app)
    await app.dispatch(
        RISK_EVENT,
        inline=True,
        condition={"__blueprint__": "risk_bp"},
    )
    assert ran == ["writer", "metrics"]


@pytest.mark.asyncio
async def test_cancellation_is_not_compensated_or_aggregated(app):
    ran = []
    gate = asyncio.Event()

    async def compensation(error, **_):
        ran.append("compensated")

    @app.signal(
        RISK_EVENT,
        critical=True,
        failure=FailureMode.COMPENSATE,
        compensation=compensation,
    )
    async def slow(**_):
        ran.append("started")
        gate.set()
        await asyncio.sleep(10)

    @app.signal(RISK_EVENT, order=1)
    async def after(**_):
        ran.append("after")

    _finalize(app)
    task = asyncio.create_task(app.dispatch(RISK_EVENT, inline=True))
    await gate.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert ran == ["started"]


@pytest.mark.asyncio
async def test_detached_task_dispatch_does_not_share_view(app):
    """后台任务派发拥有独立视图，两次任务派发互不影响嵌套快照。"""
    ran = []

    @app.signal(RISK_EVENT, order=0)
    async def first(**_):
        ran.append("first")

    @app.signal(RISK_EVENT, order=1)
    async def second(**_):
        ran.append("second")

    _finalize(app)
    task1 = await app.dispatch(RISK_EVENT)
    task2 = await app.dispatch(RISK_EVENT)
    await asyncio.gather(task1, task2)
    assert ran == ["first", "second", "first", "second"]


# --------------------------------------------------------------------------- #
# 策略声明校验
# --------------------------------------------------------------------------- #


def test_compensate_requires_compensation(app):
    with pytest.raises(InvalidSignal, match="requires a compensation"):

        @app.signal(
            RISK_EVENT,
            critical=True,
            failure=FailureMode.COMPENSATE,
        )
        async def handler(**_): ...


def test_compensation_requires_compensate_mode(app):
    async def comp(**_): ...

    with pytest.raises(InvalidSignal, match="only be registered"):

        @app.signal(
            RISK_EVENT,
            failure=FailureMode.ISOLATE,
            compensation=comp,
        )
        async def handler(**_): ...


def test_unknown_failure_mode(app):
    with pytest.raises(InvalidSignal, match="Unknown signal failure mode"):
        SignalPolicy(failure="explode")


def test_policy_object_and_kwargs_are_mutually_exclusive(app):
    policy = SignalPolicy(critical=True, failure=FailureMode.ISOLATE)
    with pytest.raises(Exception):
        app.signal(RISK_EVENT, policy=policy, critical=False)


@pytest.mark.asyncio
async def test_policy_object_registration(app):
    policy = SignalPolicy(
        critical=True,
        failure=FailureMode.ISOLATE,
        domain="risk",
        order=5,
    )

    @app.signal(RISK_EVENT, policy=policy)
    async def handler(**_): ...

    _finalize(app)
    routes = [
        route
        for route in app.signal_router.routes
        if getattr(route.ctx, "definition", "") == RISK_EVENT
    ]
    assert routes[0].ctx.policy is policy
