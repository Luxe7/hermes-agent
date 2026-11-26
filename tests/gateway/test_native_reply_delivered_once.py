"""A native reply reaches the chat exactly once even when an already-running drain claims and
runs the admission before ``admit_message`` resumes to register its delivery waiter
(andrexibiza-6, andrexibiza-5480238075-1). Every native reply goes through one of two senders:
the drain's ``deliver_settled`` (no live waiter) or the waiting adapter lifecycle."""
import asyncio

import pytest

from gateway.session_contract import Submission


@pytest.mark.asyncio
async def test_native_waiter_registered_after_the_claim_still_gets_the_only_delivery(tmp_path, monkeypatch):
    from gateway import session_ingress
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionActor, SessionRef
    from tests.gateway.test_prompt_attachments import _authority as _store_authority

    turn_started = asyncio.Event()

    async def answer(event):
        turn_started.set()
        return 'model reply'
    authority = await _store_authority(tmp_path, monkeypatch, answer)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c')
    authority.sessions['s'] = LiveSession(source, 's')
    authority.runner._adapter_for_source = lambda source: object()
    drain_sent = []

    async def deliver(adapter, event, session_key, response):
        drain_sent.append(response)
    monkeypatch.setattr(session_ingress, 'deliver_response', deliver)
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')

    async def admit_native(event):
        receipt = await authority.submit(actor, Submission('m1', SessionRef('p', 's'), {'text': 'hi'}, 'queue'))
        # The admitting caller resumes only after the drain already claimed and started the turn.
        await turn_started.wait()
        return receipt
    monkeypatch.setattr(authority, 'admit_native', admit_native)
    adapter_sent = await asyncio.wait_for(
        session_ingress.admit_message(authority, MessageEvent(text='hi', source=source)), 10)
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    deliveries = drain_sent + ([adapter_sent] if adapter_sent else [])
    assert deliveries == ['model reply'], f'the reply was sent {len(deliveries)} times: {deliveries}'
    assert not authority.native_waiters and not authority.pending_deliveries


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_concurrent_native_retries_share_one_final_delivery(tmp_path, monkeypatch, cancel_first):
    from gateway import session_ingress
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionActor, SessionRef
    from tests.gateway.test_prompt_attachments import _authority

    started, release = asyncio.Event(), asyncio.Event()
    executions = []

    async def answer(event):
        executions.append(event.text)
        started.set()
        await release.wait()
        return 'model reply\nMEDIA: /tmp/example.png'

    authority = await _authority(tmp_path, monkeypatch, answer)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c')
    authority.sessions['s'] = LiveSession(source, 's')
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')
    admitted = asyncio.Event()
    calls = 0

    async def admit_native(event):
        nonlocal calls
        receipt = await authority.submit(actor, Submission('m1', SessionRef('p', 's'), {'text': 'hi'}, 'queue'))
        calls += 1
        if calls == 2:
            admitted.set()
        return receipt

    monkeypatch.setattr(authority, 'admit_native', admit_native)

    async def deliver():
        event = MessageEvent(text='hi', source=source)
        event._turn_marker_handoff = True
        return await session_ingress.admit_message(authority, event)

    async with asyncio.TaskGroup() as callers:
        first = callers.create_task(deliver())
        await asyncio.wait_for(started.wait(), 5)
        second = callers.create_task(deliver())
        await asyncio.wait_for(admitted.wait(), 5)
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        release.set()
    await asyncio.wait_for(authority.sessions['s'].task, 5)
    replies = [task.result() for task in (first, second) if not task.cancelled() and task.result()]
    assert executions == ['hi']
    assert replies == ['model reply\nMEDIA: /tmp/example.png']
