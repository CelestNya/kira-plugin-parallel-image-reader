"""
v2.0.0 生产级行为测试。stub 重依赖后加载真实 ParallelImageReader，零网络/DB 依赖。
覆盖：链替换、缓存、嵌套、并发、超时、异常隔离、环检测。

直接 `uv run python test_v2.py` 运行。
"""

import sys
import time
import types
import asyncio
import importlib.util
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# Core stubs
# ═══════════════════════════════════════════════════════════════

class _Logger:
    _buffer: list[str] = []

    def info(self, *a, **k):
        self._buffer.append(f"[INFO] {a[0] if a else ''}")
    def warning(self, *a, **k):
        self._buffer.append(f"[WARN] {a[0] if a else ''}")
    def error(self, *a, **k):
        self._buffer.append(f"[ERR]  {a[0] if a else ''}")
    def debug(self, *a, **k):
        pass


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# ── Realistic message element stubs ──

class _ImageBase:
    """Base for Image/Sticker stubs — realistic enough for isinstance checks
    and async method dispatch, zero I/O."""
    def __init__(self, md5: str = "aabbccdd11223344", desc: Optional[str] = None):
        self._md5 = md5
        self.caption = desc

    async def hash_image(self) -> str:
        return self._md5

    async def to_data_url(self) -> str:
        return "data:image/jpeg;base64,/9j/4AAQSkZJRg=="  # fake valid prefix


class Image(_ImageBase):
    def __init__(self, md5="aabbccdd11223344", desc=None):
        super().__init__(md5, desc)
        self.image = None


class Sticker(_ImageBase):
    pass


class Text:
    def __init__(self, content):
        self.content = content
        self.text = content  # .text used by message_format_to_text


class Reply:
    def __init__(self, chain=None):
        self.chain = chain


class Forward:
    def __init__(self, chains=None):
        self.chains = chains


class LLMRequest:
    def __init__(self, messages=None, system_prompt=None, user_prompt=None):
        self.messages = messages or []
        self.system_prompt = system_prompt or []
        self.user_prompt = user_prompt or []


# ── Fake DB with in-memory cache ──

class FakeDB:
    def __init__(self):
        self._cache: dict[str, str] = {}

    async def get_image_desc_cache(self, md5: str) -> Optional[dict]:
        desc = self._cache.get(md5)
        if desc is not None:
            return {"description": desc, "count": 1}
        return None

    async def add_image_desc_cache(self, md5: str, desc: str, **kw):
        self._cache[md5] = desc

    def seed(self, md5: str, desc: str):
        """Pre-populate cache for testing."""
        self._cache[md5] = desc


# ── Fake VLM provider ──

class FakeVLMResponse:
    def __init__(self, text: str):
        self.text_response = text


class FakeVLM:
    """VLM stub: returns configured descriptions, supports delays for timeout testing."""
    def __init__(self, description: str = "默认图片描述", delay: float = 0):
        self._description = description
        self._delay = delay
        self.call_count = 0

    async def chat(self, request) -> FakeVLMResponse:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return FakeVLMResponse(self._description)


class FakeProviderMgr:
    def __init__(self, vlm: FakeVLM):
        self._vlm = vlm

    def get_default_vlm(self):
        return self._vlm


# ── Fake plugin context ──

class FakeCtx:
    def __init__(self, db: FakeDB, vlm: FakeVLM):
        self.db = db
        self._vlm = vlm
        self._data_dir = str(Path(__file__).parent / "_test_data")

    def get_plugin_data_dir(self):
        return self._data_dir

    @property
    def provider_mgr(self):
        return FakeProviderMgr(self._vlm)


# ── Fake event objects ──

class FakeMessageChain(list):
    def is_empty(self):
        return len(self) == 0


class FakeMessage:
    def __init__(self, chain):
        self.chain = FakeMessageChain(chain)


class FakeSession:
    def __init__(self, sid: str = "test_session"):
        self.sid = sid


class FakeMessageEvent:
    """Mimics KiraMessageEvent — minimal fields plugin touches."""
    def __init__(self, chain, sid: str = "test_session"):
        self.session = FakeSession(sid)
        self.message = FakeMessage(chain)


class FakeMessageBatchEvent:
    def __init__(self, sid: str = "test_session"):
        self.session = FakeSession(sid)


# ═══════════════════════════════════════════════════════════════
# Plugin loader
# ═══════════════════════════════════════════════════════════════

_LOADED: Optional["PluginModule"] = None


def load_plugin():
    """Load ParallelImageReader module with core stubs in place.
    Returns (module, element_classes)."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    class _BasePlugin:
        def __init__(self, ctx, cfg):
            self.ctx = ctx
            self.plugin_cfg = cfg

    class _Priority:
        SYS_HIGH = 100

    class _on:
        @staticmethod
        def im_message(**k):
            def deco(f): return f
            return deco

        @staticmethod
        def llm_request(**k):
            def deco(f): return f
            return deco

    # Register core stubs BEFORE import
    _stub("core.logging_manager", get_logger=lambda *a, **k: _Logger())
    _stub("core.plugin", BasePlugin=_BasePlugin, PluginContext=object,
          register_tool=lambda *a, **k: (lambda f: f), on=_on, Priority=_Priority,
          logger=_Logger())
    _stub("core.chat.message_elements", Image=Image, Text=Text, Sticker=Sticker,
          Reply=Reply, Forward=Forward)
    _stub("core.chat.message_utils", KiraMessageEvent=FakeMessageEvent,
          KiraMessageBatchEvent=FakeMessageBatchEvent)
    # desc_img stub that delegates to VLM.chat for the native path
    async def _fake_desc_img(client, image, prompt):
        request = LLMRequest(messages=[])
        resp = await client.chat(request)
        return resp.text_response or ""

    _stub("core.utils.common_utils", desc_img=_fake_desc_img)
    _stub("core.provider", LLMRequest=LLMRequest)

    # Stub PIL — realistic enough for the quality path (convert, resize, save)
    class _FakePIL:
        mode = "RGB"
        size = (224, 224)
        def resize(self, *a, **k): return self
        def convert(self, *a, **k): return self
        def save(self, *a, **k): pass
    if "PIL" not in sys.modules:
        pil = _stub("PIL")
        _stub("PIL.Image", new=lambda *a, **k: _FakePIL(),
              open=lambda *a, **k: _FakePIL())
        pil.Image = sys.modules["PIL.Image"]

    main_path = Path(__file__).parent / "main.py"
    spec = importlib.util.spec_from_file_location("pir_v2_test", str(main_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _LOADED = (mod, {"Image": Image, "Sticker": Sticker, "Text": Text,
                     "Reply": Reply, "Forward": Forward})
    return _LOADED


# ═══════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════

async def _make_plugin(db: FakeDB, vlm: FakeVLM, cfg: Optional[dict] = None):
    """Create a configured ParallelImageReader instance with fake dependencies."""
    mod, _ = load_plugin()
    ctx = FakeCtx(db, vlm)
    plug = mod.ParallelImageReader(ctx, cfg or {})
    await plug.initialize()
    return plug, mod


def _chain_texts(chain) -> list[str]:
    """Extract content from Text elements in a chain (flat, no recursion)."""
    return [ele.text if hasattr(ele, "text") else str(ele) for ele in chain]


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0
_SKIP = 0


def _test(name: str):
    """Decorator: run async test, count pass/fail."""
    global _PASS, _FAIL, _SKIP
    def deco(func):
        async def wrapper():
            global _PASS, _FAIL
            try:
                await func()
                _PASS += 1
                print(f"  [OK] {name}")
            except Exception as e:
                _FAIL += 1
                print(f"  [FAIL] {name}: {e}")
        return wrapper
    return deco


def _check(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


# ── T1: Basic single image ──

@_test("T1: single image → described & replaced")
async def _t1():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("一只猫"))
    img = Image(md5="t1_md5")
    ev = FakeMessageEvent([img, Text("hello")])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check(len(texts) == 2, f"expected 2 elements, got {len(texts)}")
    _check("[Image: 一只猫" in texts[0], f"unexpected: {texts[0]}")
    _check(texts[1] == "hello")


# ── T2: Multiple images ──

@_test("T2: multiple images → all described in parallel")
async def _t2():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc", delay=0.05))
    imgs = [Image(md5=f"t2_{i}") for i in range(3)]
    ev = FakeMessageEvent(imgs)

    t0 = time.monotonic()
    await plug.on_im_message(ev)
    elapsed = time.monotonic() - t0

    texts = _chain_texts(ev.message.chain)
    _check(len(texts) == 3, f"expected 3, got {len(texts)}")
    for t in texts:
        _check("[Image: desc" in t, f"unexpected: {t}")
    # With concurrency=3, 3 images each 50ms → ~50ms, not 150ms
    _check(elapsed < 0.12, f"took {elapsed:.3f}s, expected < 0.12s (parallel)")


# ── T3: Cache hit ──

@_test("T3: cache hit → skip VLM")
async def _t3():
    db = FakeDB()
    db.seed("t3_md5", "cached description")
    vlm = FakeVLM("fresh desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5="t3_md5")
    ev = FakeMessageEvent([img])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check(len(texts) == 1)
    _check("[Image: cached description" in texts[0],
           f"got: {texts[0]}")
    _check(vlm.call_count == 0, f"VLM called {vlm.call_count} times (expected 0)")


# ── T4: Cache miss → VLM → cached ──

@_test("T4: cache miss → VLM called → desc cached")
async def _t4():
    db = FakeDB()
    vlm = FakeVLM("fresh desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5="t4_md5")
    ev = FakeMessageEvent([img])

    await plug.on_im_message(ev)

    cached = await db.get_image_desc_cache("t4_md5")
    _check(cached is not None, "not cached")
    _check(cached["description"] == "fresh desc")
    _check(vlm.call_count == 1)


# ── T5: Mixed cache ──

@_test("T5: mixed cache hit/miss")
async def _t5():
    db = FakeDB()
    db.seed("hit_md5", "cached")
    vlm = FakeVLM("fresh")
    plug, mod = await _make_plugin(db, vlm)
    img_hit = Image(md5="hit_md5")
    img_miss = Image(md5="miss_md5")
    ev = FakeMessageEvent([img_hit, img_miss])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check(len(texts) == 2)
    _check("[Image: cached" in texts[0], f"got: {texts[0]}")
    _check("[Image: fresh" in texts[1], f"got: {texts[1]}")
    _check(vlm.call_count == 1, f"VLM called {vlm.call_count} times (expected 1)")


# ── T6: Image inside Reply ──

@_test("T6: image inside Reply.chain")
async def _t6():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("猫"))
    reply_img = Image(md5="reply_md5")
    reply = Reply(chain=[Text("quote"), reply_img])
    ev = FakeMessageEvent([Text("forward"), reply])

    await plug.on_im_message(ev)

    # Flat chain: Text("forward"), Reply(chain=[Text, Text])
    # After replacement: Reply.chain[1] should be Text("[Image: 猫]")
    _check(isinstance(ev.message.chain[0], Text), "first should be Text")
    _check(isinstance(ev.message.chain[1], Reply), "second should be Reply")
    _check(isinstance(ev.message.chain[1].chain[0], Text), "reply[0] Text")
    _check(isinstance(ev.message.chain[1].chain[1], Text), "reply[1] Text")
    _check("[Image: 猫" in ev.message.chain[1].chain[1].text,
           f"got: {ev.message.chain[1].chain[1].text}")


# ── T7: Images inside Forward ──

@_test("T7: images inside Forward.chains")
async def _t7():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("图"))
    fwd = Forward(chains=[
        [Image(md5="f1"), Text("a")],
        [Text("b"), Image(md5="f2")],
    ])
    ev = FakeMessageEvent([fwd])

    await plug.on_im_message(ev)

    fwd = ev.message.chain[0]
    _check(isinstance(fwd, Forward))
    c0 = fwd.chains[0]
    c1 = fwd.chains[1]
    _check("[Image: 图" in c0[0].text, f"c0[0]: {c0[0].text}")
    _check(c0[1].text == "a")
    _check(c1[0].text == "b")
    _check("[Image: 图" in c1[1].text, f"c1[1]: {c1[1].text}")


# ── T8: No images → 0 ──

@_test("T8: no images → no op")
async def _t8():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    ev = FakeMessageEvent([Text("a"), Text("b")])
    await plug.on_im_message(ev)
    texts = _chain_texts(ev.message.chain)
    _check(texts == ["a", "b"], f"got: {texts}")


# ── T9: VLM returns empty → fallback ──

@_test("T9: VLM fails → (description unavailable)")
async def _t9():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM(""))
    img = Image(md5="t9_md5")
    ev = FakeMessageEvent([img])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check("(description unavailable)" in texts[0],
           f"got: {texts[0]}")


# ── T10: Timeout → fallback ──

@_test("T10: VLM timeout → (description unavailable)")
async def _t10():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("too slow", delay=99))
    mod.VLM_TIMEOUT = 0.05  # force short timeout
    img = Image(md5="t10_md5")
    ev = FakeMessageEvent([img])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check("(description unavailable)" in texts[0],
           f"got: {texts[0]}")


# ── T11: Cycle detection ──

@_test("T11: Reply cycle → no RecursionError")
async def _t11():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    r = Reply()
    r2 = Reply(chain=[r])
    r.chain = [r2]  # r → r2 → r (cycle!)
    ev = FakeMessageEvent([r])

    try:
        await plug.on_im_message(ev)
    except RecursionError:
        raise AssertionError("RecursionError not prevented by cycle detection")


# ── T12: Event handler exception safety ──

@_test("T12: bad event → caught, no crash")
async def _t12():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    ev = FakeMessageEvent("not_a_chain")  # intentionally broken
    try:
        await plug.on_im_message(ev)
    except Exception:
        raise AssertionError("exception escaped on_im_message")


# ── T13: Concurrent concurrency control ──

@_test("T13: concurrency=1 → sequential (total ~N*delay)")
async def _t13():
    db = FakeDB()
    vlm = FakeVLM("slow", delay=0.03)
    plug, mod = await _make_plugin(db, vlm, cfg={"max_concurrent": 1})
    imgs = [Image(md5=f"t13_{i}") for i in range(3)]
    ev = FakeMessageEvent(imgs)

    t0 = time.monotonic()
    await plug.on_im_message(ev)
    elapsed = time.monotonic() - t0

    _check(elapsed >= 0.07, f"too fast ({elapsed:.3f}s), expected >= 90ms (sequential)")
    texts = _chain_texts(ev.message.chain)
    _check(len(texts) == 3, f"got {len(texts)} elements")


# ── T14: Quality path ──

@_test("T14: quality_enabled path works")
async def _t14():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("quality desc"),
                             cfg={"quality_enabled": True, "quality_value": 50})
    img = Image(md5="t14_md5")
    ev = FakeMessageEvent([img])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check("[Image: quality desc" in texts[0],
           f"got: {texts[0]}")


# ── T15: Sticker (same as Image) ──

@_test("T15: Sticker element → described")
async def _t15():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("sticker desc"))
    st = Sticker(md5="t15_md5")
    ev = FakeMessageEvent([st])

    await plug.on_im_message(ev)

    texts = _chain_texts(ev.message.chain)
    _check("[Image: sticker desc" in texts[0],
           f"got: {texts[0]}")


# ── T16: Empty chain → no errors ──

@_test("T16: empty chain → no error, 0 images")
async def _t16():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    ev = FakeMessageEvent([])
    await plug.on_im_message(ev)
    _check(isinstance(ev.message.chain, FakeMessageChain))
    _check(len(ev.message.chain) == 0)


# ── T17: on_llm_request system hint injection ──

@_test("T17: on_llm_request injects system hint")
async def _t17():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    fake_batch_ev = FakeMessageBatchEvent("test")
    fake_prompt = types.SimpleNamespace(name="chat_env", content="")
    req = LLMRequest(system_prompt=[fake_prompt])

    await plug.on_llm_request(fake_batch_ev, req)

    _check("当消息中包含 [Image: 描述内容]" in fake_prompt.content,
           f"hint not injected: {fake_prompt.content}")


# ── T18: Sticker in Forward ──

@_test("T18: Sticker inside Forward.chains")
async def _t18():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("图"))
    fwd = Forward(chains=[
        [Sticker(md5="s1"), Text("a")],
    ])
    ev = FakeMessageEvent([fwd])
    await plug.on_im_message(ev)
    c0 = ev.message.chain[0].chains[0]
    _check("[Image: 图" in c0[0].text, f"got: {c0[0].text}")
    _check(c0[1].text == "a")


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

_TESTS = [
    _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10,
    _t11, _t12, _t13, _t14, _t15, _t16, _t17, _t18,
]


def main():
    global _PASS, _FAIL, _SKIP
    print(f"\nParallel Image Reader v2.0.0 — 生产级行为测试\n")
    print(f"共 {len(_TESTS)} 个测试\n")

    asyncio.run(_run_all())

    total = _PASS + _FAIL + _SKIP
    print(f"\n── 结果: {_PASS}/{total} 通过", end="")
    if _FAIL:
        print(f", {_FAIL} 失败", end="")
    if _SKIP:
        print(f", {_SKIP} 跳过", end="")
    print()
    return 1 if _FAIL else 0


async def _run_all():
    for t in _TESTS:
        await t()


if __name__ == "__main__":
    sys.exit(main())
