"""
v2.1.0 两阶段架构测试。stub 重依赖后加载真实 ParallelImageReader，零网络/DB 依赖。
覆盖：阶段1占位替换+暂存、阶段2并行VLM+message_str替换、缓存、嵌套、并发、超时、
      异常隔离、环检测、discard 零 VLM。

直接 `python test_v2.py` 运行。
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
    """模拟 KiraIMMessage — 有 chain 和 message_str，可挂自定义属性。"""
    def __init__(self, chain, message_str=None):
        self.chain = FakeMessageChain(chain)
        # _pir_pending 默认 None（模拟本体行为，插件会设置它）
        self._pir_pending = None
        # message_str 模拟本体 message_format_to_text 的输出
        if message_str is not None:
            self.message_str = message_str
        else:
            self.message_str = _simulate_format_to_text(chain)


def _simulate_format_to_text(chain, _visited=None) -> str:
    """模拟本体 message_format_to_text 的输出格式。

    本体对 Image: caption is None → 调 VLM；caption is not None → 跳过 VLM，写缓存
    输出格式: [Image {caption}, file_path: data/temp/xxx.jpg]
    对 Sticker: [Sticker {caption}]
    对 Text: {text}

    带环检测：用 id(chain) 标记，避免 Reply 互引导致无限递归。
    注意：阶段1已把 Image 替换为 Text(占位符)，所以本函数看到的是 Text。
    """
    if _visited is None:
        _visited = set()
    cid = id(chain)
    if cid in _visited:
        return "[cycle]"
    _visited.add(cid)

    parts = []
    for ele in chain:
        if isinstance(ele, Text):
            parts.append(ele.text)
        elif isinstance(ele, (Image, Sticker)):
            # 阶段1未处理的情况（理论上不应出现，因为插件先于本体执行）
            cap = getattr(ele, "caption", None)
            if cap is None:
                cap = "(vlm_desc)"
            if isinstance(ele, Image):
                parts.append(f"[Image {cap}, file_path: data/temp/fake.jpg]")
            else:
                parts.append(f"[Sticker {cap}]")
        elif isinstance(ele, Reply):
            if ele.chain:
                inner = _simulate_format_to_text(ele.chain, _visited)
                parts.append(f"[Reply ID: r1 content: {inner}]")
            else:
                parts.append("[Reply ID: r1]")
        elif isinstance(ele, Forward):
            if ele.chains:
                contents = ""
                for c in ele.chains:
                    contents += f"\n{_simulate_format_to_text(c, _visited)}\n"
                parts.append(f"[Forward {contents.strip()}]")
        else:
            parts.append(str(ele))
    return "".join(parts)


class FakeSession:
    def __init__(self, sid: str = "test_session"):
        self.sid = sid


class FakeMessageEvent:
    """Mimics KiraMessageEvent — minimal fields plugin touches."""
    def __init__(self, chain, sid: str = "test_session"):
        self.session = FakeSession(sid)
        self.message = FakeMessage(chain, message_str="")


class FakeBatchMessage:
    """模拟 batch 中的单条消息。chain 是已经过阶段1处理的（Image→Text占位）。"""
    def __init__(self, chain, message_str=None, _pir_pending=None):
        self.chain = FakeMessageChain(chain)
        self._pir_pending = _pir_pending
        self.message_str = message_str if message_str is not None else _simulate_format_to_text(chain)


class FakeMessageBatchEvent:
    """Mimics KiraMessageBatchEvent — minimal fields plugin touches."""
    def __init__(self, messages, sid: str = "test_session"):
        self.session = FakeSession(sid)
        self.messages = messages if isinstance(messages, list) else [messages]


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
        def im_batch_message(**k):
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


import re as _re
_PH_RE = _re.compile(r"\x00PIR_[^\x00]+\x00")


def _is_placeholder(text: str) -> bool:
    """判断文本是否为占位符。"""
    return bool(_PH_RE.match(text))


def _make_batch_from_event(ev: FakeMessageEvent) -> FakeBatchMessage:
    """从 FakeMessageEvent 构造 FakeBatchMessage，复用同一个 message 对象。

    模拟本体 flush_session_messages 的行为：batch 里的 message 就是
    ON_IM_MESSAGE 阶段的 event.message（同一个实例）。
    """
    msg = ev.message
    # 重新生成 message_str（模拟本体 message_format_to_text，此时 chain 已含占位 Text）
    msg.message_str = _simulate_format_to_text(msg.chain)
    return FakeBatchMessage(
        chain=msg.chain,
        message_str=msg.message_str,
        _pir_pending=getattr(msg, "_pir_pending", None),
    )


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


# ── 阶段 1: ON_IM_MESSAGE 占位替换测试 ──

@_test("T1: on_im_message 缓存未命中 → 替换为占位符 + 暂存")
async def _t1():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("一只猫"))
    img = Image(md5="t1_md5_0000000000000000000000ab")
    ev = FakeMessageEvent([img, Text("hello")])

    await plug.on_im_message(ev)

    # Image 应被替换为 Text(占位符)
    _check(isinstance(ev.message.chain[0], Text), "Image should be replaced by Text")
    _check(_is_placeholder(ev.message.chain[0].text), f"not placeholder: {ev.message.chain[0].text!r}")
    # 暂存到 _pir_pending
    pending = getattr(ev.message, "_pir_pending", None)
    _check(pending is not None and len(pending) == 1, f"pending={pending}")
    # VLM 不应被调用
    _check(plug.ctx.provider_mgr.get_default_vlm().call_count == 0, "VLM should not be called")


@_test("T2: on_im_message 缓存命中 → 直接替换为 [Image: 描述]")
async def _t2():
    db = FakeDB()
    db.seed("t2_md5_0000000000000000000000ab", "cached cat")
    plug, mod = await _make_plugin(db, FakeVLM("fresh"))
    img = Image(md5="t2_md5_0000000000000000000000ab")
    ev = FakeMessageEvent([img])

    await plug.on_im_message(ev)

    _check(isinstance(ev.message.chain[0], Text), "should be Text")
    _check("[Image: cached cat]" == ev.message.chain[0].text, f"got: {ev.message.chain[0].text}")
    _check(plug.ctx.provider_mgr.get_default_vlm().call_count == 0)
    # 不应有 pending
    _check(getattr(ev.message, "_pir_pending", None) is None, "should not have pending")


@_test("T3: on_im_message 混合缓存命中/未命中")
async def _t3():
    db = FakeDB()
    db.seed("hit_md5_000000000000000000000000", "cached")
    plug, mod = await _make_plugin(db, FakeVLM("fresh"))
    img_hit = Image(md5="hit_md5_000000000000000000000000")
    img_miss = Image(md5="miss_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img_hit, img_miss])

    await plug.on_im_message(ev)

    _check("[Image: cached]" == ev.message.chain[0].text, f"hit: {ev.message.chain[0].text}")
    _check(_is_placeholder(ev.message.chain[1].text), f"miss: {ev.message.chain[1].text}")
    pending = getattr(ev.message, "_pir_pending", None)
    _check(pending is not None and len(pending) == 1, f"pending={pending}")


@_test("T4: on_im_message 无图片 → 无操作")
async def _t4():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    ev = FakeMessageEvent([Text("a"), Text("b")])
    await plug.on_im_message(ev)
    _check(isinstance(ev.message.chain[0], Text))
    _check(isinstance(ev.message.chain[1], Text))
    _check(getattr(ev.message, "_pir_pending", None) is None)


@_test("T5: on_im_message 嵌套 Reply 中的图片被替换")
async def _t5():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    reply_img = Image(md5="reply_md5_0000000000000000000000a")
    reply = Reply(chain=[Text("quote"), reply_img])
    ev = FakeMessageEvent([Text("forward"), reply])

    await plug.on_im_message(ev)

    _check(isinstance(reply.chain[1], Text), "reply image should be Text")
    _check(_is_placeholder(reply.chain[1].text), f"not placeholder: {reply.chain[1].text!r}")


@_test("T6: on_im_message 嵌套 Forward 中的图片被替换")
async def _t6():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    f1 = Image(md5="f1_md5_0000000000000000000000000a")
    f2 = Image(md5="f2_md5_0000000000000000000000000a")
    fwd = Forward(chains=[[f1, Text("a")], [Text("b"), f2]])
    ev = FakeMessageEvent([fwd])

    await plug.on_im_message(ev)

    _check(_is_placeholder(fwd.chains[0][0].text), f"f1: {fwd.chains[0][0].text!r}")
    _check(_is_placeholder(fwd.chains[1][1].text), f"f2: {fwd.chains[1][1].text!r}")


@_test("T7: on_im_message 环检测不崩溃")
async def _t7():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    r = Reply()
    r2 = Reply(chain=[r])
    r.chain = [r2]  # r → r2 → r (cycle!)
    ev = FakeMessageEvent([r])

    try:
        await plug.on_im_message(ev)
    except RecursionError:
        raise AssertionError("RecursionError not prevented by cycle detection")


@_test("T8: on_im_message 异常事件不崩溃")
async def _t8():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    ev = FakeMessageEvent("not_a_chain")
    try:
        await plug.on_im_message(ev)
    except Exception:
        raise AssertionError("exception escaped on_im_message")


# ── 阶段 2: ON_IM_BATCH_MESSAGE 并行 VLM + 替换测试 ──

@_test("T9: batch 单图 → VLM 描述 + message_str 替换")
async def _t9():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("一只猫"))
    img = Image(md5="t9_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img, Text("hello")])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # message_str 中的占位符应被替换
    _check("[Image: 一只猫]" in batch_msg.message_str, f"message_str: {batch_msg.message_str}")
    _check("\x00PIR" not in batch_msg.message_str, f"placeholder leaked: {batch_msg.message_str!r}")
    # chain 中的占位 Text 也应被替换
    _check("[Image: 一只猫]" == batch_msg.chain[0].text, f"chain[0]: {batch_msg.chain[0].text}")


@_test("T10: batch 多图并行（3图50ms各 → ~50ms）")
async def _t10():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc", delay=0.05))
    imgs = [Image(md5=f"t10_{i}_" + "0" * 24) for i in range(3)]
    ev = FakeMessageEvent(imgs)
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])

    t0 = time.monotonic()
    await plug.on_im_batch_message(batch_ev)
    elapsed = time.monotonic() - t0

    # 3 个占位符都应被替换
    _check("\x00PIR" not in batch_msg.message_str, f"placeholder leaked: {batch_msg.message_str!r}")
    count = batch_msg.message_str.count("[Image: desc]")
    _check(count == 3, f"expected 3, got {count}")
    _check(elapsed < 0.12, f"took {elapsed:.3f}s, expected < 0.12s (parallel)")


@_test("T11: batch 全部缓存命中 → 无 pending，message_str 仍正确")
async def _t11():
    db = FakeDB()
    md5 = "t11_md5_00000000000000000000000a"
    db.seed(md5, "cached description")
    vlm = FakeVLM("fresh desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("[Image: cached description]" in batch_msg.message_str, f"got: {batch_msg.message_str}")
    _check(vlm.call_count == 0, f"VLM called {vlm.call_count} times (expected 0)")


@_test("T12: batch cache miss → VLM → 写入缓存")
async def _t12():
    db = FakeDB()
    md5 = "t12_md5_00000000000000000000000a"
    vlm = FakeVLM("fresh desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    cached = await db.get_image_desc_cache(md5)
    _check(cached is not None, "not cached")
    _check(cached["description"] == "fresh desc")
    _check(vlm.call_count == 1)


@_test("T13: batch 混合缓存命中/未命中")
async def _t13():
    db = FakeDB()
    db.seed("hit_md5_000000000000000000000000", "cached")
    vlm = FakeVLM("fresh")
    plug, mod = await _make_plugin(db, vlm)
    img_hit = Image(md5="hit_md5_000000000000000000000000")
    img_miss = Image(md5="miss_md5_0000000000000000000000a")
    ev = FakeMessageEvent([img_hit, img_miss])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("[Image: cached]" in batch_msg.message_str, f"got: {batch_msg.message_str}")
    _check("[Image: fresh]" in batch_msg.message_str, f"got: {batch_msg.message_str}")
    _check(vlm.call_count == 1, f"VLM called {vlm.call_count} times (expected 1)")


@_test("T14: batch VLM 返回空 → 降级")
async def _t14():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM(""))
    img = Image(md5="t14_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("(description unavailable)" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T15: batch VLM 超时 → 降级")
async def _t15():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("too slow", delay=99))
    mod.VLM_TIMEOUT = 0.05
    img = Image(md5="t15_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("(description unavailable)" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T16: batch concurrency=1 → 串行（~N*delay）")
async def _t16():
    db = FakeDB()
    vlm = FakeVLM("slow", delay=0.03)
    plug, mod = await _make_plugin(db, vlm, cfg={"max_concurrent": 1})
    imgs = [Image(md5=f"t16_{i}_" + "0" * 24) for i in range(3)]
    ev = FakeMessageEvent(imgs)
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])

    t0 = time.monotonic()
    await plug.on_im_batch_message(batch_ev)
    elapsed = time.monotonic() - t0

    _check(elapsed >= 0.07, f"too fast ({elapsed:.3f}s), expected >= 90ms (sequential)")
    count = batch_msg.message_str.count("[Image: slow]")
    _check(count == 3, f"expected 3, got {count}")


@_test("T17: batch quality_enabled 路径")
async def _t17():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("quality desc"),
                             cfg={"quality_enabled": True, "quality_value": 50})
    img = Image(md5="t17_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("[Image: quality desc]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T18: batch Sticker 元素 → 描述")
async def _t18():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("sticker desc"))
    st = Sticker(md5="t18_md5_00000000000000000000000a")
    ev = FakeMessageEvent([st])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("[Image: sticker desc]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T19: batch 空 chain → 无错误")
async def _t19():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    batch_msg = FakeBatchMessage([])
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)
    _check(len(batch_msg.chain) == 0)


@_test("T20: batch 嵌套 Reply 中的图片 → 描述+替换")
async def _t20():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("猫"))
    reply_img = Image(md5="reply_md5_0000000000000000000000a")
    reply = Reply(chain=[Text("quote"), reply_img])
    ev = FakeMessageEvent([Text("forward"), reply])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # Reply.chain[1] 应从 Text(占位符) 变为 Text("[Image: 猫]")
    _check("[Image: 猫]" == batch_msg.chain[1].chain[1].text,
           f"got: {batch_msg.chain[1].chain[1].text}")
    _check("[Image: 猫]" in batch_msg.message_str, f"message_str: {batch_msg.message_str}")


@_test("T21: batch 嵌套 Forward 中的图片 → 描述+替换")
async def _t21():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("图"))
    f1 = Image(md5="f1_md5_0000000000000000000000000a")
    f2 = Image(md5="f2_md5_0000000000000000000000000a")
    fwd = Forward(chains=[[f1, Text("a")], [Text("b"), f2]])
    ev = FakeMessageEvent([fwd])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    fwd = batch_msg.chain[0]
    _check("[Image: 图]" == fwd.chains[0][0].text, f"c0[0]: {fwd.chains[0][0].text}")
    _check("[Image: 图]" == fwd.chains[1][1].text, f"c1[1]: {fwd.chains[1][1].text}")
    _check(batch_msg.message_str.count("[Image: 图]") == 2, f"message_str: {batch_msg.message_str}")


@_test("T22: batch Sticker 嵌套 Forward → 描述+替换")
async def _t22():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("图"))
    st = Sticker(md5="s1_md5_0000000000000000000000000a")
    fwd = Forward(chains=[[st, Text("a")]])
    ev = FakeMessageEvent([fwd])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    c0 = batch_msg.chain[0].chains[0]
    _check("[Image: 图]" == c0[0].text, f"got: {c0[0].text}")


@_test("T23: batch 环检测不崩溃")
async def _t23():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    r = Reply()
    r2 = Reply(chain=[r])
    r.chain = [r2]
    ev = FakeMessageEvent([r])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    try:
        await plug.on_im_batch_message(batch_ev)
    except RecursionError:
        raise AssertionError("RecursionError not prevented in batch")


@_test("T24: batch 异常事件不崩溃")
async def _t24():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    batch_ev = FakeMessageBatchEvent("not_messages")
    try:
        await plug.on_im_batch_message(batch_ev)
    except Exception:
        raise AssertionError("exception escaped on_im_batch_message")


# ── 占位符不泄漏测试 ──

@_test("T25: 占位符不泄漏到 message_str（正常流程）")
async def _t25():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("猫"))
    img = Image(md5="t25_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img, Text("hello")])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("\x00PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")
    _check("[Image: 猫]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T26: 占位符不泄漏到 message_str（缓存命中）")
async def _t26():
    db = FakeDB()
    md5 = "t26_md5_00000000000000000000000a"
    db.seed(md5, "cached desc")
    plug, mod = await _make_plugin(db, FakeVLM("fresh"))
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("\x00PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")
    _check("[Image: cached desc]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T27: 占位符不泄漏到 message_str（异常兜底）")
async def _t27():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    img = Image(md5="t27_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    # 故意破坏 _pir_pending，触发异常路径
    batch_msg._pir_pending = "broken"
    batch_ev = FakeMessageBatchEvent([batch_msg])

    try:
        await plug.on_im_batch_message(batch_ev)
    except Exception:
        pass  # 异常应被捕获

    _check("\x00PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")


# ── discard 零 VLM 测试 ──

@_test("T28: discard 的消息不做 VLM（不进入 batch）")
async def _t28():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5="t28_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    # 阶段1替换为占位符
    await plug.on_im_message(ev)
    _check(vlm.call_count == 0, "VLM should not be called in stage 1")
    # 模拟消息被 discard：不进入 batch，不调用 on_im_batch_message
    _check(vlm.call_count == 0, f"VLM called {vlm.call_count} times for discarded msg")


@_test("T29: batch 多消息批量处理")
async def _t29():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc", delay=0.02))
    img1 = Image(md5="t29_1_md5_00000000000000000000000a")
    img2 = Image(md5="t29_2_md5_00000000000000000000000a")
    ev1 = FakeMessageEvent([img1, Text("msg1")])
    ev2 = FakeMessageEvent([img2, Text("msg2")])
    await plug.on_im_message(ev1)
    await plug.on_im_message(ev2)

    batch_msg1 = _make_batch_from_event(ev1)
    batch_msg2 = _make_batch_from_event(ev2)
    batch_ev = FakeMessageBatchEvent([batch_msg1, batch_msg2])

    t0 = time.monotonic()
    await plug.on_im_batch_message(batch_ev)
    elapsed = time.monotonic() - t0

    # 两图并行（concurrency=3），20ms 各 → ~20ms
    _check(elapsed < 0.06, f"took {elapsed:.3f}s, expected < 0.06s (parallel batch)")
    _check("[Image: desc]" in batch_msg1.message_str, f"msg1: {batch_msg1.message_str}")
    _check("[Image: desc]" in batch_msg2.message_str, f"msg2: {batch_msg2.message_str}")


# ── on_llm_request system hint 测试 ──

@_test("T30: on_llm_request 注入 system hint")
async def _t30():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    fake_batch_ev = FakeMessageBatchEvent([])
    fake_prompt = types.SimpleNamespace(name="chat_env", content="")
    req = LLMRequest(system_prompt=[fake_prompt])

    await plug.on_llm_request(fake_batch_ev, req)

    _check("当消息中包含 [Image: 描述内容]" in fake_prompt.content,
           f"hint not injected: {fake_prompt.content}")


# ── 缓存不被污染测试（回归 v2.1.0 caption 方案的 bug）──

@_test("T31: 缓存不被占位符污染")
async def _t31():
    db = FakeDB()
    md5 = "t31_md5_00000000000000000000000a"
    vlm = FakeVLM("real desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # 缓存中应是真实描述，不是占位符
    cached = await db.get_image_desc_cache(md5)
    _check(cached is not None, "not cached")
    _check(cached["description"] == "real desc", f"cached: {cached['description']!r}")
    _check("\x00PIR" not in cached["description"], f"cache polluted: {cached['description']!r}")


@_test("T32: 污染缓存（含\\x00）被忽略，走 VLM")
async def _t32():
    db = FakeDB()
    md5 = "t32_md5_00000000000000000000000a"
    # 模拟被旧版 caption 方案污染的缓存条目
    db.seed(md5, "\x00IMG_PENDING_t32_md5_00000000000000000000000a")
    vlm = FakeVLM("fresh real desc")
    plug, mod = await _make_plugin(db, vlm)
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    # 污染缓存应被忽略，图片走 pending（占位符），不是 [Image: 污染数据]
    _check(isinstance(ev.message.chain[0], Text), "should be Text")
    _check(_is_placeholder(ev.message.chain[0].text),
           f"polluted cache should be ignored, got: {ev.message.chain[0].text!r}")

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # VLM 应被调用
    _check(vlm.call_count == 1, f"VLM should be called, got {vlm.call_count}")
    # 最终描述应是真实描述
    _check("[Image: fresh real desc]" in batch_msg.message_str,
           f"got: {batch_msg.message_str}")
    # 缓存应被正确覆写
    cached = await db.get_image_desc_cache(md5)
    _check(cached["description"] == "fresh real desc", f"cached: {cached['description']!r}")


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

_TESTS = [
    # 阶段1
    _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8,
    # 阶段2
    _t9, _t10, _t11, _t12, _t13, _t14, _t15, _t16, _t17, _t18, _t19,
    _t20, _t21, _t22, _t23, _t24,
    # 占位符不泄漏
    _t25, _t26, _t27,
    # discard + batch
    _t28, _t29,
    # hint
    _t30,
    # 缓存不污染
    _t31,
    # 污染缓存忽略
    _t32,
]


def main():
    global _PASS, _FAIL, _SKIP
    print(f"\nParallel Image Reader v2.1.0 — 两阶段架构测试\n")
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
