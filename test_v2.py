"""
v2.2.0 两阶段架构 + 乐观加载测试。stub 重依赖后加载真实 ParallelImageReader，零网络/DB 依赖。
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
    """对齐真实 KiraAI Reply（message_id 必须存在——stub 缺属性会让
    插件新代码 AttributeError 被吞，测试假绿）。"""
    def __init__(self, chain=None, message_id="r1"):
        self.message_id = str(message_id)
        self.chain = chain


class Forward:
    def __init__(self, chains=None):
        self.chains = chains


class LLMRequest:
    def __init__(self, messages=None, system_prompt=None, user_prompt=None,
                 tool_set=None):
        self.messages = messages or []
        self.system_prompt = system_prompt or []
        self.user_prompt = user_prompt or []
        self.tool_set = tool_set


class FakeToolSet:
    """模拟 ToolSet — 支持 remove 检查。"""
    def __init__(self, names: list[str] = None):
        self.tools = names or []

    def remove(self, *names):
        self.tools = [t for t in self.tools if t not in names]


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
        self._data_dir = Path(__file__).parent / "_test_data"

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
    def __init__(self, chain, message_str=None, is_mentioned=False, group=None):
        self.chain = FakeMessageChain(chain)
        # _pir_images / _pir_optimistic 默认 None（模拟本体行为，插件会设置它）
        self._pir_images = None
        self._pir_optimistic = None
        # 环境属性（测试质量反思：此前缺这些导致私聊/提及场景从未被覆盖）
        self.is_mentioned = is_mentioned
        self.group = group
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
    def __init__(self, chain, sid: str = "test_session",
                 mentioned: bool = False, group: Optional[object] = None):
        self.session = FakeSession(sid)
        # group 非空 = 群聊；None = 私聊（对齐本体 is_group_message 判定）
        self.message = FakeMessage(chain, message_str="",
                                   is_mentioned=mentioned, group=group)


class FakeBatchMessage:
    """模拟 batch 中的单条消息。chain 是已经过阶段1处理的（Image→Text空标识符）。"""
    def __init__(self, chain, message_str=None,
                 _pir_optimistic=None, _pir_images=None,
                 is_mentioned=False, group=None):
        self.chain = FakeMessageChain(chain)
        self._pir_optimistic = _pir_optimistic
        self._pir_images = _pir_images
        self.is_mentioned = is_mentioned
        self.group = group
        self.message_str = message_str if message_str is not None else _simulate_format_to_text(chain)


class FakeMessageBatchEvent:
    """Mimics KiraMessageBatchEvent — minimal fields plugin touches."""
    def __init__(self, messages, sid: str = "test_session"):
        self.session = FakeSession(sid)
        self.messages = messages if isinstance(messages, list) else [messages]

    def is_group_message(self) -> bool:
        """对齐本体：batch 最后一条消息的 group 非空 = 群聊。"""
        if not self.messages:
            return False
        return getattr(self.messages[-1], "group", None) is not None


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

    class _register:
        """register.tool(...) 装饰器 stub。"""
        @staticmethod
        def tool(name, description, params):
            def deco(f): return f
            return deco

    # Register core stubs BEFORE import
    _stub("core.logging_manager", get_logger=lambda *a, **k: _Logger())
    _stub("core.plugin", BasePlugin=_BasePlugin, PluginContext=object,
          register_tool=lambda *a, **k: (lambda f: f), register=_register,
          on=_on, Priority=_Priority, logger=_Logger())
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

    class _Prompt:
        """Prompt stub — 与 core.prompt_manager.Prompt 接口对齐。"""
        def __init__(self, content, name=None, source=None, persist=True):
            self.content = content
            self.name = name
            self.source = source
            self.persist = persist

        def to_string(self):
            return self.content or ""

    _stub("core.prompt_manager", Prompt=_Prompt)

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


def _make_plugin_ctx(db: FakeDB, vlm: FakeVLM, data_dir: str):
    """Create a plugin with an isolated data dir (id_map 测试用)。"""
    mod, _ = load_plugin()
    ctx = FakeCtx(db, vlm)
    ctx._data_dir = data_dir
    plug = mod.ParallelImageReader(ctx, {})
    return plug, ctx


def _chain_texts(chain) -> list[str]:
    """Extract content from Text elements in a chain (flat, no recursion)."""
    return [ele.text if hasattr(ele, "text") else str(ele) for ele in chain]


import re as _re
# 空标识符：[Image #id: ]（待填充态）
_EMPTY_ID_RE = _re.compile(r"\[Image #[^\]]+: \]")


def _is_placeholder(text: str) -> bool:
    """判断文本是否为空标识符（待填充态）。"""
    return bool(_EMPTY_ID_RE.match(text))


def _make_batch_from_event(ev: FakeMessageEvent) -> FakeBatchMessage:
    """从 FakeMessageEvent 构造 FakeBatchMessage，复用同一个 message 对象。

    模拟本体 flush_session_messages 的行为：batch 里的 message 就是
    ON_IM_MESSAGE 阶段的 event.message（同一个实例）。
    附带传递 _pir_images / _pir_optimistic。
    """
    msg = ev.message
    # 重新生成 message_str（模拟本体 message_format_to_text，此时 chain 已含空标识符）
    msg.message_str = _simulate_format_to_text(msg.chain)
    return FakeBatchMessage(
        chain=msg.chain,
        message_str=msg.message_str,
        _pir_images=getattr(msg, "_pir_images", None),
        _pir_optimistic=getattr(msg, "_pir_optimistic", None),
        # 环境属性必须随消息传递（本体 batch 复用同一 message 实例）：
        # 缺失会导致 is_group_message 误判私聊 → 自动读取误触发
        is_mentioned=getattr(msg, "is_mentioned", False),
        group=getattr(msg, "group", None),
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

@_test("T1: on_im_message 缓存未命中 → 替换为空标识符 + 暂存 _pir_images")
async def _t1():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("一只猫"))
    img = Image(md5="t1_md5_0000000000000000000000ab")
    ev = FakeMessageEvent([img, Text("hello")])

    await plug.on_im_message(ev)

    # Image 应被替换为 Text(空标识符)
    _check(isinstance(ev.message.chain[0], Text), "Image should be replaced by Text")
    _check(_is_placeholder(ev.message.chain[0].text), f"not placeholder: {ev.message.chain[0].text!r}")
    # 暂存到 _pir_images
    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is not None and len(images_map) == 1, f"images_map={images_map}")
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
    _check("[Image #t2_md5_0: cached cat]" == ev.message.chain[0].text, f"got: {ev.message.chain[0].text}")
    _check(plug.ctx.provider_mgr.get_default_vlm().call_count == 0)
    # 缓存命中不应有 _pir_images
    _check(getattr(ev.message, "_pir_images", None) is None, "should not have images_map")


@_test("T3: on_im_message 混合缓存命中/未命中")
async def _t3():
    db = FakeDB()
    db.seed("hit_md5_000000000000000000000000", "cached")
    plug, mod = await _make_plugin(db, FakeVLM("fresh"))
    img_hit = Image(md5="hit_md5_000000000000000000000000")
    img_miss = Image(md5="miss_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img_hit, img_miss])

    await plug.on_im_message(ev)

    _check("[Image #hit_md5_: cached]" == ev.message.chain[0].text, f"hit: {ev.message.chain[0].text}")
    _check(_is_placeholder(ev.message.chain[1].text), f"miss: {ev.message.chain[1].text}")
    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is not None and len(images_map) == 1, f"images_map={images_map}")


@_test("T4: on_im_message 无图片 → 无操作")
async def _t4():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    ev = FakeMessageEvent([Text("a"), Text("b")])
    await plug.on_im_message(ev)
    _check(isinstance(ev.message.chain[0], Text))
    _check(isinstance(ev.message.chain[1], Text))
    _check(getattr(ev.message, "_pir_images", None) is None)


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
    _check("[Image #t9_md5_0: 一只猫]" in batch_msg.message_str, f"message_str: {batch_msg.message_str}")
    _check("<!--PIR" not in batch_msg.message_str, f"placeholder leaked: {batch_msg.message_str!r}")
    # chain 中的占位 Text 也应被替换
    _check("[Image #t9_md5_0: 一只猫]" == batch_msg.chain[0].text, f"chain[0]: {batch_msg.chain[0].text}")


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

    # 3 个占位符都应被替换为 [Image #id: desc]（id 各不相同）
    _check("<!--PIR" not in batch_msg.message_str, f"placeholder leaked: {batch_msg.message_str!r}")
    count = batch_msg.message_str.count(": desc]")
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

    _check("[Image #t11_md5_: cached description]" in batch_msg.message_str, f"got: {batch_msg.message_str}")
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

    _check("cached" in batch_msg.message_str, f"got: {batch_msg.message_str}")
    _check("fresh" in batch_msg.message_str, f"got: {batch_msg.message_str}")
    _check(": ]" not in batch_msg.message_str, f"empty id left: {batch_msg.message_str}")
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
    count = batch_msg.message_str.count(": slow]")
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

    _check("[Image #t17_md5_: quality desc]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T18: batch Sticker 元素 → 描述")
async def _t18():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("sticker desc"))
    st = Sticker(md5="t18_md5_00000000000000000000000a")
    ev = FakeMessageEvent([st])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("[Image #t18_md5_: sticker desc]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


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

    # Reply.chain[1] 应从 Text(占位符) 变为 Text("[Image #reply_md: 猫]")
    _check("[Image #reply_md: 猫]" == batch_msg.chain[1].chain[1].text,
           f"got: {batch_msg.chain[1].chain[1].text}")
    _check("[Image #reply_md: 猫]" in batch_msg.message_str, f"message_str: {batch_msg.message_str}")


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
    _check("[Image #f1_md5_0: 图]" == fwd.chains[0][0].text, f"c0[0]: {fwd.chains[0][0].text}")
    _check("[Image #f2_md5_0: 图]" == fwd.chains[1][1].text, f"c1[1]: {fwd.chains[1][1].text}")
    _check(batch_msg.message_str.count(": 图]") == 2, f"message_str: {batch_msg.message_str}")


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
    _check("[Image #s1_md5_0: 图]" == c0[0].text, f"got: {c0[0].text}")


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

    _check("<!--PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")
    _check("[Image #t25_md5_: 猫]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


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

    _check("<!--PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")
    _check("[Image #t26_md5_: cached desc]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


@_test("T27: 异常路径 → 空标识符残留合法，不崩溃")
async def _t27():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    img = Image(md5="t27_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    # 故意破坏 _pir_images，触发异常路径（非 dict）
    batch_msg._pir_images = "broken"
    batch_ev = FakeMessageBatchEvent([batch_msg])

    try:
        await plug.on_im_batch_message(batch_ev)
    except Exception:
        pass  # 异常应被捕获

    # 空标识符残留是合法状态（与 llm_select 常态一致），无旧占位符
    _check("<!--PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")


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
    _check("[Image #t29_1_md: desc]" in batch_msg1.message_str, f"msg1: {batch_msg1.message_str}")
    _check("[Image #t29_2_md: desc]" in batch_msg2.message_str, f"msg2: {batch_msg2.message_str}")


# ── on_llm_request system hint 测试 ──

@_test("T30: on_llm_request 注入 system hint")
async def _t30():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    fake_batch_ev = FakeMessageBatchEvent([])
    fake_prompt = types.SimpleNamespace(name="chat_env", content="")
    req = LLMRequest(system_prompt=[fake_prompt])

    await plug.on_llm_request(fake_batch_ev, req)

    _check("当消息中包含 [Image #xxxx: 描述内容]" in fake_prompt.content,
           f"hint not injected: {fake_prompt.content}")


# ── 缓存不被污染测试（回归：占位符不应写入缓存）──

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
    _check("<!--PIR" not in cached["description"], f"cache polluted: {cached['description']!r}")


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

    # 污染缓存应被忽略，图片走 pending（占位符），不是 "[Image #t32_md5_: 污染数据]"
    _check(isinstance(ev.message.chain[0], Text), "should be Text")
    _check(_is_placeholder(ev.message.chain[0].text),
           f"polluted cache should be ignored, got: {ev.message.chain[0].text!r}")

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # VLM 应被调用
    _check(vlm.call_count == 1, f"VLM should be called, got {vlm.call_count}")
    # 最终描述应是真实描述
    _check("[Image #t32_md5_: fresh real desc]" in batch_msg.message_str,
           f"got: {batch_msg.message_str}")
    # 缓存应被正确覆写
    cached = await db.get_image_desc_cache(md5)
    _check(cached["description"] == "fresh real desc", f"cached: {cached['description']!r}")


# ═══════════════════════════════════════════════════════════════
# 边界条件测试 (T33-T40)
# ═══════════════════════════════════════════════════════════════

# ── T33: hash_image 阶段1失败 → 仍设占位符，阶段2降级 ──

class _HashFailImage(Image):
    """hash_image 始终抛异常的图片元素。"""
    def __init__(self):
        super().__init__(md5="unused")

    async def hash_image(self):
        raise RuntimeError("image data corrupted")


@_test("T33: hash_image 失败 → 阶段1空标识符（noid_），阶段2降级描述")
async def _t33():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    img = _HashFailImage()
    ev = FakeMessageEvent([img, Text("hello")])
    await plug.on_im_message(ev)

    # 阶段1应设空标识符（用 noid_ 前缀），不崩溃
    _check(isinstance(ev.message.chain[0], Text), "should be Text placeholder")
    _check(_is_placeholder(ev.message.chain[0].text), f"not placeholder: {ev.message.chain[0].text!r}")
    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is not None and len(images_map) == 1, "should have 1 images_map")

    # 阶段2：hash 再次失败，_describe_one 的 try/except 捕获，返回空描述
    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # 空标识符应被替换为降级文本
    _check(": ]" not in batch_msg.message_str, f"empty id leaked: {batch_msg.message_str!r}")
    _check("(description unavailable)" in batch_msg.message_str, f"got: {batch_msg.message_str}")


# ── T34: VLM 未配置 (get_default_vlm 返回 None) ──

class _NoneVLMProviderMgr:
    """get_default_vlm 返回 None 的 ProviderManager。"""
    def get_default_vlm(self):
        return None


@_test("T34: VLM 未配置 → 降级描述，不崩溃")
async def _t34():
    mod, _ = load_plugin()

    class _NoneVLMCtx(FakeCtx):
        def __init__(self):
            super().__init__(FakeDB(), FakeVLM(""))  # vlm 不会被用到
        @property
        def provider_mgr(self):
            return _NoneVLMProviderMgr()

    ctx = _NoneVLMCtx()
    plug = mod.ParallelImageReader(ctx, {})
    await plug.initialize()

    img = Image(md5="t34_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # 不崩溃，占位符被替换为降级文本
    _check("<!--PIR" not in batch_msg.message_str, f"placeholder leaked: {batch_msg.message_str!r}")
    _check("(description unavailable)" in batch_msg.message_str, f"got: {batch_msg.message_str}")


# ── T35: VLM chat 抛异常（非超时） ──

class _CrashVLM:
    """chat 始终抛异常的 VLM。"""
    def __init__(self):
        self.call_count = 0
    async def chat(self, request):
        self.call_count += 1
        raise ConnectionError("VLM API unreachable")


def _make_crash_ctx(db, crash_vlm):
    """构造 VLM 始终崩溃的 context。"""
    class _CrashProviderMgr:
        def get_default_vlm(self):
            return crash_vlm
    mod, _ = load_plugin()

    class _CrashCtx(FakeCtx):
        @property
        def provider_mgr(self):
            return _CrashProviderMgr()
    ctx = _CrashCtx(db, crash_vlm)
    return mod, ctx


@_test("T35: VLM chat 抛异常 → 降级描述，不影响其他图片")
async def _t35():
    db = FakeDB()
    crash_vlm = _CrashVLM()
    mod, ctx = _make_crash_ctx(db, crash_vlm)
    plug = mod.ParallelImageReader(ctx, {})
    await plug.initialize()

    imgs = [Image(md5=f"t35_{i}_0000000000000000000000000a") for i in range(3)]
    ev = FakeMessageEvent(imgs)
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # 3 张图都降级
    _check(crash_vlm.call_count == 3, f"VLM called {crash_vlm.call_count} times (expected 3)")
    _check(batch_msg.message_str.count("(description unavailable)") == 3,
           f"got: {batch_msg.message_str}")
    _check("<!--PIR" not in batch_msg.message_str, "placeholder leaked")


# ── T36: to_data_url 失败（quality 模式） ──

class _DataUrlFailImage(Image):
    """to_data_url 抛异常的图片。"""
    def __init__(self):
        super().__init__(md5="t36_md5_0000000000000000000000a")

    async def to_data_url(self):
        raise IOError("failed to download image")


@_test("T36: quality 模式 to_data_url 失败 → 降级描述")
async def _t36():
    plug, mod = await _make_plugin(
        FakeDB(), FakeVLM("desc"),
        cfg={"quality_enabled": True, "quality_value": 50}
    )
    img = _DataUrlFailImage()
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("(description unavailable)" in batch_msg.message_str, f"got: {batch_msg.message_str}")
    _check("<!--PIR" not in batch_msg.message_str, "placeholder leaked")


# ── T37: _pir_images 丢失（message 实例未复用） ──

@_test("T37: _pir_images 丢失 → 空标识符残留为合法状态，不崩溃")
async def _t37():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    img = Image(md5="t37_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    # 构造一个没有 _pir_images 的 batch_msg（模拟实例未复用 / 属性丢失）
    chain = ev.message.chain  # chain 里有空标识符 Text
    batch_msg = FakeBatchMessage(chain, message_str=f"[Image #t37_md5_: ]")
    batch_msg._pir_images = None  # 模拟丢失
    batch_ev = FakeMessageBatchEvent([batch_msg])

    await plug.on_im_batch_message(batch_ev)

    # 空标识符残留是合法状态（与 llm_select 常态一致），不崩溃、无旧占位符
    _check("<!--PIR" not in batch_msg.message_str, f"leaked: {batch_msg.message_str!r}")
    _check("[Image #t37_md5_: ]" in batch_msg.message_str, f"got: {batch_msg.message_str}")


# ── T38: 同一图片重复出现（同 md5，多消息） ──

@_test("T38: 同一图片在多消息中重复 → 各自替换正确")
async def _t38():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("same desc"))
    md5 = "t38_md5_00000000000000000000000a"
    img1 = Image(md5=md5)
    img2 = Image(md5=md5)  # 同 md5
    ev1 = FakeMessageEvent([img1, Text("msg1")])
    ev2 = FakeMessageEvent([img2, Text("msg2")])
    await plug.on_im_message(ev1)
    await plug.on_im_message(ev2)

    batch_msg1 = _make_batch_from_event(ev1)
    batch_msg2 = _make_batch_from_event(ev2)
    batch_ev = FakeMessageBatchEvent([batch_msg1, batch_msg2])
    await plug.on_im_batch_message(batch_ev)

    # 两条消息的占位符都应被替换
    _check("[Image #t38_md5_: same desc]" in batch_msg1.message_str, f"msg1: {batch_msg1.message_str}")
    _check("[Image #t38_md5_: same desc]" in batch_msg2.message_str, f"msg2: {batch_msg2.message_str}")
    _check("<!--PIR" not in batch_msg1.message_str, "msg1 leaked")
    _check("<!--PIR" not in batch_msg2.message_str, "msg2 leaked")


# ── T39: message_str 为 None ──

@_test("T39: message_str 为 None → 不崩溃，chain 仍替换")
async def _t39():
    plug, mod = await _make_plugin(FakeDB(), FakeVLM("desc"))
    img = Image(md5="t39_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_msg.message_str = None  # 模拟 None
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # chain 中的占位符应被替换
    _check("[Image #t39_md5_: desc]" == batch_msg.chain[0].text, f"chain: {batch_msg.chain[0].text}")


# ── T40: 混合场景（hash成功+失败+缓存命中+VLM失败） ──

@_test("T40: 混合场景 — hash成功/失败/缓存命中/VLM失败 共存")
async def _t40():
    db = FakeDB()
    db.seed("cached_md5_000000000000000000000000", "cached desc")
    crash_vlm = _CrashVLM()
    mod, ctx = _make_crash_ctx(db, crash_vlm)
    plug = mod.ParallelImageReader(ctx, {})
    await plug.initialize()

    # 4 张图：缓存命中、hash失败、VLM会失败(hash成功)、正常(hash成功但VLM失败)
    img_cached = Image(md5="cached_md5_000000000000000000000000")
    img_hashfail = _HashFailImage()
    img_vlmfail1 = Image(md5="vlmfail1_00000000000000000000000a")
    img_vlmfail2 = Image(md5="vlmfail2_00000000000000000000000a")
    ev = FakeMessageEvent([img_cached, img_hashfail, img_vlmfail1, img_vlmfail2])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    # 缓存命中的有描述
    _check("[Image #cached_m: cached desc]" in batch_msg.message_str, f"cached: {batch_msg.message_str}")
    # 其余降级
    _check(batch_msg.message_str.count("(description unavailable)") == 3,
           f"expected 3 unavailable, got: {batch_msg.message_str}")
    # 无占位符泄漏
    _check("<!--PIR" not in batch_msg.message_str, "placeholder leaked")


# ═══════════════════════════════════════════════════════════════
# 乐观加载测试（eager_loading，v2.2.0）
# ═══════════════════════════════════════════════════════════════

@_test("T41: 乐观加载开启 → on_im_message 不阻塞，启动后台 VLM task")
async def _t41():
    """VLM delay=0.1s，on_im_message 应在 0.05s 内返回（非阻塞）。"""
    vlm = FakeVLM("desc", delay=0.1)
    plug, mod = await _make_plugin(
        FakeDB(), vlm, {"eager_loading": True}
    )
    img = Image(md5="t41_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    t0 = time.monotonic()
    await plug.on_im_message(ev)
    elapsed = time.monotonic() - t0
    _check(elapsed < 0.05,
           f"on_im_message blocked {elapsed:.3f}s (VLM delay=0.1s)")
    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is not None and len(images_map) == 1, "should have 1 images_map")
    task = getattr(ev.message, "_pir_optimistic", None)
    _check(task is not None, "should have optimistic task")
    await asyncio.sleep(0.15)  # 等待 task 完成
    _check(vlm.call_count == 1, f"VLM called {vlm.call_count}, expected 1")


@_test("T42: 乐观加载开启 → batch 阶段复用 task，VLM 不重复调用")
async def _t42():
    db = FakeDB()
    vlm = FakeVLM("eager_desc", delay=0.02)
    plug, mod = await _make_plugin(db, vlm, {"eager_loading": True})
    md5 = "t42_md5_00000000000000000000000a"
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check(vlm.call_count == 1,
           f"VLM called {vlm.call_count}, expected 1 (reuse)")
    _check("[Image #t42_md5_: eager_desc]" in batch_msg.message_str,
           f"got: {batch_msg.message_str}")
    _check("<!--PIR" not in batch_msg.message_str, "placeholder leaked")
    cached = await db.get_image_desc_cache(md5)
    _check(cached and cached["description"] == "eager_desc",
           f"cache: {cached['description'] if cached else 'None'}")


@_test("T43: 乐观加载开启 + 全缓存命中 → 不启动 task")
async def _t43():
    db = FakeDB()
    db.seed("t43_md5_00000000000000000000000a", "cached desc")
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"eager_loading": True})
    img = Image(md5="t43_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is None or len(images_map) == 0,
           "should have no images_map (cache hit)")
    task = getattr(ev.message, "_pir_optimistic", None)
    _check(task is None, "should have no optimistic task (cache hit)")
    _check(vlm.call_count == 0,
           f"VLM called {vlm.call_count}, expected 0")


@_test("T44: 乐观加载开启 + task 异常 → 降级文本，不崩溃")
async def _t44():
    """手动设置一个崩溃的 _pir_optimistic task，验证异常隔离。"""
    plug, mod = await _make_plugin(
        FakeDB(), FakeVLM(""), {"eager_loading": True}
    )
    md5 = "t44_md5_00000000000000000000000a"
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    # 覆盖为会崩溃的 task（加 sleep(0) 让事件循环先调度 gather，
    # 避免 "Task exception was never retrieved" 警告）
    async def _crash():
        await asyncio.sleep(0)  # 让 gather 有机会捕获异常
        raise RuntimeError("sim crash for T44")
    crash_task = asyncio.create_task(_crash())
    ev.message._pir_optimistic = crash_task

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)

    _check("<!--PIR" not in batch_msg.message_str,
           f"placeholder leaked: {batch_msg.message_str!r}")
    _check("(description unavailable)" in batch_msg.message_str,
           f"got: {batch_msg.message_str}")


@_test("T45: 乐观加载关闭 → 行为不变（回归）")
async def _t45():
    """eager_loading=False（默认）时 _pir_optimistic 不被设置。"""
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(FakeDB(), vlm)
    img = Image(md5="t45_md5_00000000000000000000000a")
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is not None, "should have images_map")
    task = getattr(ev.message, "_pir_optimistic", None)
    _check(task is None, "should NOT have optimistic task when eager=off")

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)
    _check("[Image #t45_md5_: desc]" in batch_msg.message_str,
           f"got: {batch_msg.message_str}")
    _check(vlm.call_count == 1,
           f"VLM called {vlm.call_count}, expected 1 (fresh)")


@_test("T46: 乐观加载开启 + 多消息 batch → 全部正确处理")
async def _t46():
    """两条消息，都有 pending，batch 阶段合并处理。"""
    db = FakeDB()
    vlm = FakeVLM("desc46", delay=0.01)
    plug, mod = await _make_plugin(db, vlm, {"eager_loading": True})
    md5_1 = "t46_1_md5_00000000000000000000000a"
    md5_2 = "t46_2_md5_00000000000000000000000b"
    img1 = Image(md5=md5_1)
    img2 = Image(md5=md5_2)
    ev1 = FakeMessageEvent([img1, Text("msg1")])
    ev2 = FakeMessageEvent([img2, Text("msg2")])
    await plug.on_im_message(ev1)
    await plug.on_im_message(ev2)

    batch_msg1 = _make_batch_from_event(ev1)
    batch_msg2 = _make_batch_from_event(ev2)
    batch_ev = FakeMessageBatchEvent([batch_msg1, batch_msg2])
    await plug.on_im_batch_message(batch_ev)

    # VLM 仅被每条消息的 on_im_message 各调一次
    _check(vlm.call_count == 2,
           f"VLM called {vlm.call_count}, expected 2")
    _check(f"[Image #{md5_1[:8]}: desc46]" in batch_msg1.message_str,
           f"msg1: {batch_msg1.message_str}")
    _check(f"[Image #{md5_2[:8]}: desc46]" in batch_msg2.message_str,
           f"msg2: {batch_msg2.message_str}")
    _check("<!--PIR" not in batch_msg1.message_str, "msg1 placeholder leaked")
    _check("<!--PIR" not in batch_msg2.message_str, "msg2 placeholder leaked")


# ═══════════════════════════════════════════════════════════════
# llm_select 模式测试（v2.3.0）
# ═══════════════════════════════════════════════════════════════

def _mk_tool_req(messages=None, system_prompt=None):
    """构造带 FakeToolSet 的 LLMRequest。"""
    tool_set = FakeToolSet(["describe_image"])
    return LLMRequest(messages=messages or [], system_prompt=system_prompt or [],
                      tool_set=tool_set), tool_set


@_test("T47: llm_select 阶段1 → 空标识符 + _pir_images + id_map")
async def _t47():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t47_md5_00000000000000000000000a"
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    # 空标识符 [Image #id: ]，不触发 VLM
    _check(f"[Image #{md5[:8]}: ]" == ev.message.chain[0].text,
           f"got: {ev.message.chain[0].text!r}")
    _check(vlm.call_count == 0, f"VLM called {vlm.call_count}, expected 0")
    # _pir_images 挂原图
    images_map = getattr(ev.message, "_pir_images", None)
    _check(images_map is not None and md5[:8] in images_map,
           f"no _pir_images: {images_map}")
    # id_map 写入
    _check(plug._id_map.get(md5[:8]) == md5, f"id_map: {plug._id_map}")


@_test("T48: llm_select 阶段2 不 VLM，空标识符进历史")
async def _t48():
    db = FakeDB()
    vlm = FakeVLM("desc", delay=0.05)
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t48_md5_00000000000000000000000a"
    img = Image(md5=md5)
    # 群聊未提及（避免私聊单图/被@自动读取干扰本测试意图）
    ev = FakeMessageEvent([img], group=object(), mentioned=False)
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    t0 = time.monotonic()
    await plug.on_im_batch_message(batch_ev)
    elapsed = time.monotonic() - t0

    # 零 VLM、零等待
    _check(vlm.call_count == 0, f"VLM called {vlm.call_count}, expected 0")
    _check(elapsed < 0.02, f"batch took {elapsed:.3f}s, expected ~0")
    # (未识别) 系统状态标记进历史（括号区分于图片内容）
    _check(f"[Image #{md5[:8]}: (未识别)]" in batch_msg.message_str,
           f"got: {batch_msg.message_str}")
    _check("<!--PIR" not in batch_msg.message_str, "placeholder leaked")


@_test("T49: describe_image 当前回合 → VLM 描述")
async def _t49():
    db = FakeDB()
    vlm = FakeVLM("tool desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t49_md5_00000000000000000000000a"
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])

    desc = await plug.describe_image(batch_ev, md5[:8])
    _check(desc == f"[Image #{md5[:8]}: tool desc]", f"got: {desc!r}")
    _check(vlm.call_count == 1, f"VLM called {vlm.call_count}, expected 1")
    # 描述写缓存
    cached = await db.get_image_desc_cache(md5)
    _check(cached and cached["description"] == "tool desc",
           f"cache: {cached['description'] if cached else 'None'}")


@_test("T50: describe_image 历史回合 → id_map 反查缓存")
async def _t50():
    db = FakeDB()
    md5 = "t50_md5_00000000000000000000000a"
    db.seed(md5, "cached hist desc")
    plug, mod = await _make_plugin(db, FakeVLM("vlm"),
                                   {"load_mode": "llm_select"})
    # 模拟历史回合：id_map 已持久化，当前 batch 无 _pir_images
    plug._id_map[md5[:8]] = md5
    batch_ev = FakeMessageBatchEvent([])

    desc = await plug.describe_image(batch_ev, md5[:8])
    _check(desc == f"[Image #{md5[:8]}: cached hist desc]", f"got: {desc!r}")


@_test("T51: describe_image 不可追溯 → 已过期")
async def _t51():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("vlm"),
                                   {"load_mode": "llm_select"})
    batch_ev = FakeMessageBatchEvent([])
    desc = await plug.describe_image(batch_ev, "deadbeef")
    _check("已过期" in desc, f"got: {desc!r}")


@_test("T52: ON_LLM_REQUEST 扫描替换（缓存命中）")
async def _t52():
    db = FakeDB()
    md5 = "t52_md5_00000000000000000000000a"
    db.seed(md5, "scanned desc")
    plug, mod = await _make_plugin(db, FakeVLM("vlm"),
                                   {"load_mode": "llm_select"})
    plug._id_map[md5[:8]] = md5

    # 历史消息里有一个空标识符
    hist_msg = types.SimpleNamespace(
        content=f"[Image #{md5[:8]}: ]",
        role="user",
    )
    req, tool_set = _mk_tool_req(messages=[hist_msg])
    fake_prompt = types.SimpleNamespace(name="chat_env", content="")
    req.system_prompt.append(fake_prompt)

    fake_batch_ev = FakeMessageBatchEvent([])
    await plug.on_llm_request(fake_batch_ev, req)

    _check(f"[Image #{md5[:8]}: scanned desc]" in hist_msg.content,
           f"got: {hist_msg.content!r}")
    # llm_select 无图消息 → describe_image 常驻（工具前缀跨消息稳定）
    _check("describe_image" in tool_set.tools, f"tools: {tool_set.tools}")


@_test("T53: ON_LLM_REQUEST 扫描未命中 → VLM 填充（lazy 换态）")
async def _t53():
    db = FakeDB()
    vlm = FakeVLM("fill desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "lazy"})
    md5 = "t53_md5_00000000000000000000000a"
    plug._id_map[md5[:8]] = md5

    # 历史消息空标识符 + 当前回合有原图（可 VLM）
    hist_msg = types.SimpleNamespace(
        content=f"[Image #{md5[:8]}: ]",
        role="user",
    )
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)  # lazy 模式：设占位符，无 _pir_images

    # 模拟换态：把原图挂到 batch 消息的 _pir_images（llm_select 阶段1 留下的）
    batch_msg = _make_batch_from_event(ev)
    batch_msg._pir_images = {md5[:8]: img}

    batch_ev = FakeMessageBatchEvent([batch_msg])
    req, tool_set = _mk_tool_req(messages=[hist_msg])
    await plug.on_llm_request(batch_ev, req)

    _check(f"[Image #{md5[:8]}: fill desc]" in hist_msg.content,
           f"got: {hist_msg.content!r}")
    _check(vlm.call_count == 1, f"VLM called {vlm.call_count}, expected 1")


@_test("T54: 换态工具增删（load_mode 切换 → tool_set）")
async def _t54():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"))

    # lazy 模式：describe_image 被移除
    req, tool_set = _mk_tool_req()
    fake_batch_ev = FakeMessageBatchEvent([])
    await plug.on_llm_request(fake_batch_ev, req)
    _check("describe_image" not in tool_set.tools, f"lazy tools: {tool_set.tools}")

    # llm_select 模式 + 有图消息：工具保留
    plug.load_mode = "llm_select"
    md5 = "t54_md5_00000000000000000000000a"
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    req2, tool_set2 = _mk_tool_req()
    await plug.on_llm_request(batch_ev, req2)
    _check("describe_image" in tool_set2.tools, f"llm_select tools: {tool_set2.tools}")


@_test("T55: 旧配置 eager_loading=true 迁移 → load_mode=eager")
async def _t55():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                   {"eager_loading": True})
    _check(plug.load_mode == "eager", f"load_mode: {plug.load_mode}")

    plug2, mod2 = await _make_plugin(db, FakeVLM("desc"),
                                     {"eager_loading": False})
    _check(plug2.load_mode == "lazy", f"load_mode: {plug2.load_mode}")

    plug3, mod3 = await _make_plugin(db, FakeVLM("desc"),
                                     {"load_mode": "llm_select",
                                      "eager_loading": True})
    _check(plug3.load_mode == "llm_select",
           f"load_mode should prefer explicit: {plug3.load_mode}")

    # 顶层 id_map_limit 读取（schema 简化：配置项都放外面）
    plug4, mod4 = await _make_plugin(db, FakeVLM("desc"),
                                     {"load_mode": "llm_select",
                                      "id_map_limit": 42})
    _check(plug4.id_map_limit == 42, f"id_map_limit: {plug4.id_map_limit}")

    # 兼容旧位置：llm_select_config section 内的 id_map_limit
    plug5, mod5 = await _make_plugin(db, FakeVLM("desc"),
                                     {"load_mode": "llm_select",
                                      "llm_select_config": {"id_map_limit": 7}})
    _check(plug5.id_map_limit == 7, f"legacy id_map_limit: {plug5.id_map_limit}")

    # 顶层优先于旧位置
    plug6, mod6 = await _make_plugin(db, FakeVLM("desc"),
                                     {"load_mode": "llm_select",
                                      "id_map_limit": 42,
                                      "llm_select_config": {"id_map_limit": 7}})
    _check(plug6.id_map_limit == 42, f"top-level should win: {plug6.id_map_limit}")


@_test("T56: id_map 1000 条 FIFO 淘汰")
async def _t56():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                   {"load_mode": "llm_select"})
    # 清空共享 data_dir 加载的旧映射，隔离测试
    plug._id_map = {}
    # 用 3 条小上限验证 FIFO（不同 md5 → 不同 short_id）
    plug.id_map_limit = 3
    for i in range(5):
        full = f"{i:08x}" + "0" * 24  # 前 8 位各不相同（00000000~00000004）
        plug._id_map_add(full[:8], full)
    _check(len(plug._id_map) == 3, f"size: {len(plug._id_map)}")
    # 最早写入的 0、1 被淘汰，最后写入的 3、4 保留
    _check("00000000" not in plug._id_map, f"map: {plug._id_map}")
    _check("00000001" not in plug._id_map, f"map: {plug._id_map}")
    _check("00000003" in plug._id_map, f"map: {plug._id_map}")
    _check("00000004" in plug._id_map, f"map: {plug._id_map}")


# ═══════════════════════════════════════════════════════════════
# 多场景 / 换态矩阵测试（v2.3.0）
# ═══════════════════════════════════════════════════════════════

@_test("T57: llm_select 多图 → 多个空标识符 + 逐张工具加载")
async def _t57():
    db = FakeDB()
    vlm = FakeVLM("multi desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5s = [f"t57_{i}_" + "0" * 24 for i in range(3)]
    imgs = [Image(md5=m) for m in md5s]
    ev = FakeMessageEvent(imgs)
    await plug.on_im_message(ev)

    # 3 个空标识符，各带独立 id
    _check(vlm.call_count == 0, f"VLM called {vlm.call_count}, expected 0")
    texts = _chain_texts(ev.message.chain)
    _check(all(f"[Image #{m[:8]}: ]" in t for t, m in zip(texts, md5s)),
           f"chain: {texts}")

    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])

    # 逐张加载：每次工具调用只 VLM 一次（返回带 id 格式）
    for i, m in enumerate(md5s):
        desc = await plug.describe_image(batch_ev, m[:8])
        _check(desc == f"[Image #{m[:8]}: multi desc]", f"#{i} got: {desc!r}")
        _check(vlm.call_count == i + 1,
               f"VLM count after #{i}: {vlm.call_count}")


@_test("T58: llm_select 缓存命中 → 直接带描述标识符，不进 _pir_images")
async def _t58():
    db = FakeDB()
    md5 = "t58_md5_00000000000000000000000a"
    db.seed(md5, "cached desc")
    plug, mod = await _make_plugin(db, FakeVLM("vlm"),
                                   {"load_mode": "llm_select"})
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)

    _check(f"[Image #{md5[:8]}: cached desc]" == ev.message.chain[0].text,
           f"got: {ev.message.chain[0].text!r}")
    _check(getattr(ev.message, "_pir_images", None) is None,
           "cache hit should not stash _pir_images")


@_test("T59: 运行时换态 eager→llm_select→lazy 全流程")
async def _t59():
    """同一插件实例运行时切换 load_mode，行为应随之切换。"""
    db = FakeDB()
    vlm = FakeVLM("mode desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "lazy"})

    # 1. lazy：阶段1 占位符，阶段2 VLM
    md5a = "t59_a_" + "0" * 24
    eva = FakeMessageEvent([Image(md5=md5a)])
    await plug.on_im_message(eva)
    _check(_is_placeholder(eva.message.chain[0].text), f"lazy: {eva.message.chain[0].text!r}")
    batch_a = _make_batch_from_event(eva)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_a]))
    _check(f"[Image #{md5a[:8]}: mode desc]" in batch_a.message_str,
           f"lazy batch: {batch_a.message_str}")

    # 2. 运行时切 llm_select：空标识符 + 不 VLM（群聊，避免私聊自动读取）
    plug.load_mode = "llm_select"
    md5b = "t59_b_" + "0" * 24
    evb = FakeMessageEvent([Image(md5=md5b)], group=object())
    await plug.on_im_message(evb)
    _check(f"[Image #{md5b[:8]}: ]" == evb.message.chain[0].text,
           f"llm_select: {evb.message.chain[0].text!r}")
    count_before = vlm.call_count
    batch_b = _make_batch_from_event(evb)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_b]))
    _check(vlm.call_count == count_before, "llm_select should not VLM in stage2")

    # 3. 切回 lazy：新消息走占位符+VLM
    plug.load_mode = "lazy"
    md5c = "t59_c_" + "0" * 24
    evc = FakeMessageEvent([Image(md5=md5c)])
    await plug.on_im_message(evc)
    _check(_is_placeholder(evc.message.chain[0].text), f"back to lazy: {evc.message.chain[0].text!r}")


@_test("T60: llm_select 历史标识符在 eager 模式下被扫描 VLM 填充")
async def _t60():
    """换态到 eager：历史空标识符 + 当前回合原图 → 扫描触发 VLM。"""
    db = FakeDB()
    vlm = FakeVLM("eager fill")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "eager"})
    md5 = "t60_md5_00000000000000000000000a"
    plug._id_map[md5[:8]] = md5

    # 历史消息里的空标识符（llm_select 时代留下）
    hist_msg = types.SimpleNamespace(content=f"[Image #{md5[:8]}: ]", role="user")

    # 当前回合消息带原图（模拟换态后新消息中的同一图）
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    # 模拟 llm_select 阶段1 的 _pir_images 残留（原图仍可追溯）
    batch_msg._pir_images = {md5[:8]: img}
    batch_ev = FakeMessageBatchEvent([batch_msg])

    req, _ = _mk_tool_req(messages=[hist_msg])
    await plug.on_llm_request(batch_ev, req)

    _check(f"[Image #{md5[:8]}: eager fill]" in hist_msg.content,
           f"got: {hist_msg.content!r}")
    _check(vlm.call_count == 1, f"VLM called {vlm.call_count}, expected 1")


@_test("T61: VLM 返回污染描述 → 降级不写缓存")
async def _t61():
    db = FakeDB()
    md5 = "t61_md5_00000000000000000000000a"
    img = Image(md5=md5)
    ev = FakeMessageEvent([img])

    # 含 \x00 的污染描述（混沌测试发现：旧实现会写缓存）
    vlm1 = FakeVLM("污染描述\x00测试")
    plug1, mod1 = await _make_plugin(db, vlm1, {"load_mode": "lazy"})
    await plug1.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug1.on_im_batch_message(batch_ev)
    _check(db._cache.get(md5) is None, f"cache polluted: {db._cache}")
    _check("(description unavailable)" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")

    # 含旧占位符的污染描述
    db2 = FakeDB()
    vlm2 = FakeVLM("<!--PIR:deadbeef-->")
    plug2, mod2 = await _make_plugin(db2, vlm2, {"load_mode": "lazy"})
    ev2 = FakeMessageEvent([Image(md5=md5)])
    await plug2.on_im_message(ev2)
    batch_msg2 = _make_batch_from_event(ev2)
    await plug2.on_im_batch_message(FakeMessageBatchEvent([batch_msg2]))
    _check(db2._cache.get(md5) is None, f"cache polluted: {db2._cache}")
    _check("<!--PIR" not in batch_msg2.message_str,
           f"placeholder leaked: {batch_msg2.message_str!r}")
    _check("(description unavailable)" in batch_msg2.message_str,
           f"got: {batch_msg2.message_str!r}")


@_test("T62: VLM 返回嵌套标识符描述 → 降级不写缓存")
async def _t62():
    db = FakeDB()
    md5 = "t62_md5_00000000000000000000000a"
    # VLM 描述里模仿标识符格式（图片内容恰含 "[Image #...]" 文字）：
    # 若写缓存会被嵌套进外层标识符，扫描器误匹配内层空标识符并改写描述
    vlm = FakeVLM("图中有文字 [Image #deadbeef: ]")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "lazy"})
    ev = FakeMessageEvent([Image(md5=md5)])
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(db._cache.get(md5) is None, f"cache polluted: {db._cache}")
    _check("(description unavailable)" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")
    # 无嵌套标识符泄漏
    _check(batch_msg.message_str.count("[Image #") == 1,
           f"nested identifier leaked: {batch_msg.message_str!r}")


@_test("T63: 真实路径 id_map FIFO 上限（on_im_message 驱动）")
async def _t63():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                   {"load_mode": "llm_select"})
    plug._id_map = {}
    plug.id_map_limit = 3
    # 5 张不同图未命中 → 全部走 _id_map_add，FIFO 淘汰最早两条
    for i in range(5):
        md5 = f"t63_{i}_" + "0" * 24
        ev = FakeMessageEvent([Image(md5=md5)])
        await plug.on_im_message(ev)
    _check(len(plug._id_map) == 3, f"size: {len(plug._id_map)}")
    _check("t63_0_00" not in plug._id_map, f"map: {plug._id_map}")
    _check("t63_1_00" not in plug._id_map, f"map: {plug._id_map}")
    _check("t63_2_00" in plug._id_map, f"map: {plug._id_map}")
    _check("t63_4_00" in plug._id_map, f"map: {plug._id_map}")

    # 缓存命中路径也写 id_map 且遵守上限（审查发现：旧实现直接写 dict 绕过 FIFO）
    db.seed("t63_9_" + "0" * 24, "缓存描述")
    ev = FakeMessageEvent([Image(md5="t63_9_" + "0" * 24)])
    await plug.on_im_message(ev)
    _check("t63_9_00" in plug._id_map, f"命中路径未写 id_map: {plug._id_map}")
    _check(len(plug._id_map) == 3, f"上限被突破: {len(plug._id_map)}")


@_test("T64: 嵌套 Forward 图片被识别（拍平修复）")
async def _t64():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("fwd desc"),
                                   {"load_mode": "lazy",
                                    "forward_max_depth": 64})
    md5 = "t64_md5_00000000000000000000000a"
    img = Image(md5=md5)
    # 两层嵌套：外层 Forward → 中层（含 Forward）→ 内层含图
    inner = FakeMessageChain([img, Text("深层")])
    mid = FakeMessageChain([Text("中层"), Forward(chains=[inner])])
    outer = FakeMessageChain([Forward(chains=[mid])])
    ev = FakeMessageEvent(outer)
    await plug.on_im_message(ev)

    # 拍平后图片进 message_str（用户可观察行为）：内容不丢 + 标识符存在
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check("[Image #" in batch_msg.message_str and "深层" in batch_msg.message_str,
           f"msg_str: {batch_msg.message_str!r}")


@_test("T65: Forward 成环不崩溃（恶意输入）")
async def _t65():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                   {"load_mode": "lazy"})
    md5 = "t65_md5_00000000000000000000000a"
    c1 = FakeMessageChain([Image(md5=md5), Text("环1")])
    c2 = FakeMessageChain([Text("环2")])
    c1.append(Forward(chains=[c2]))
    c2.append(Forward(chains=[c1]))  # c1 → c2 → c1 成环
    ev = FakeMessageEvent(FakeMessageChain([Forward(chains=[c1])]))
    await plug.on_im_message(ev)  # 不崩溃、不无限递归即通过

    # 环内至少一张图被识别进 message_str（核心过滤兜底安全降级）
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check("[Image #" in batch_msg.message_str,
           f"msg_str: {batch_msg.message_str!r}")


@_test("T66: Reply 内嵌套 Forward 图片被识别（审查补漏）")
async def _t66():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                   {"load_mode": "lazy",
                                    "forward_max_depth": 64})
    md5 = "t66_md5_00000000000000000000000a"
    # Reply.chain → Forward → 嵌套 Forward → 图片（issue 待办边界）
    inner = FakeMessageChain([Image(md5=md5), Text("深层转发")])
    fwd = Forward(chains=[FakeMessageChain([Text("层1"),
                                            Forward(chains=[inner])])])
    reply_chain = FakeMessageChain([Text("引用"), fwd])
    outer = FakeMessageChain([Reply(chain=reply_chain)])
    ev = FakeMessageEvent(outer)
    await plug.on_im_message(ev)

    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check("[Image #" in batch_msg.message_str
           and "深层转发" in batch_msg.message_str,
           f"msg_str: {batch_msg.message_str!r}")


@_test("T67: 超深 Forward 嵌套插件不崩溃（恶意，审查补漏）")
async def _t67():
    db = FakeDB()
    plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                   {"load_mode": "lazy"})
    # 第 1 层（顶层 Forward 的子链）放图 + 1500 层深链：
    # 浅层图必须被识别（阶段1 不得整体中止）
    # 修复前：递归无守卫 → RecursionError 被吞 → 阶段1 中止 → 图全丢
    md5 = "t67_md5_00000000000000000000000a"
    deep = FakeMessageChain([Image(md5=md5)])
    for _ in range(1499):
        deep = FakeMessageChain([Forward(chains=[deep])])
    top_chain = FakeMessageChain([Image(md5=md5), Forward(chains=[deep])])
    ev = FakeMessageEvent(FakeMessageChain([Forward(chains=[top_chain])]))
    await plug.on_im_message(ev)

    # 顶层 Forward 保壳，第 1 层 Forward 展开 → 浅层图被替换为 Text
    # （深层 64+ 保留 Forward 壳，无痕省略——安全降级而非整体中止）
    top = ev.message.chain[0].chains[0]
    top_types = [type(e).__name__ for e in top]
    _check("Text" in top_types, f"shallow image should be identified: {top_types}")
    _check("Image" not in top_types, f"shallow image not replaced: {top_types}")


@_test("T68: forward_max_depth 配置生效（隔断层数）")
async def _t68():
    db = FakeDB()
    md5 = "t68_md5_00000000000000000000000a"

    async def run_with(cfg):
        plug, mod = await _make_plugin(db, FakeVLM("desc"),
                                       {"load_mode": "lazy", **cfg})
        l3 = FakeMessageChain([Image(md5=md5), Text("三层")])
        l2 = FakeMessageChain([Text("二层"), Forward(chains=[l3])])
        l1 = FakeMessageChain([Text("一层"), Forward(chains=[l2])])
        ev = FakeMessageEvent(FakeMessageChain([Forward(chains=[l1])]))
        await plug.on_im_message(ev)
        # 拍平作用在 event.message.chain（包装对象），取其 Forward.chains[0]
        flattened = ev.message.chain[0].chains[0]
        return plug, [type(e).__name__ for e in flattened]

    # depth=2：L2 内容展开进 L1，L3 保留 Forward 壳（真实核心渲染时无痕过滤）
    plug1, types1 = await run_with({"forward_max_depth": 2})
    _check(plug1.forward_max_depth == 2, f"depth: {plug1.forward_max_depth}")
    _check(types1 == ["Text", "Text", "Forward"],
           f"depth=2 should keep L3 shell: {types1}")

    # 默认 1：只读第一层（跟从 KiraAI 原生）——L1 不展开，L2/L3 全保留壳
    plug2, types2 = await run_with({})
    _check(plug2.forward_max_depth == 1,
           f"default depth should be 1 (native-aligned): {plug2.forward_max_depth}")
    _check(types2 == ["Text", "Forward"],
           f"default=1 should keep L2 shell: {types2}")

    # 显式 64：三层全展开（Image 已被替换为 Text 标识符）
    plug3, types3 = await run_with({"forward_max_depth": 64})
    _check(types3 == ["Text", "Text", "Text", "Text"],
           f"depth=64 should expand all: {types3}")


@_test("T69: llm_select 当前回合空标识符不被误标已过期")
async def _t69():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t69_md5_00000000000000000000000a"
    # 群聊（避免私聊单图自动读取干扰本测试意图）
    img = Image(md5=md5)
    ev = FakeMessageEvent([img], group=object())
    await plug.on_im_message(ev)              # 空标识符 + _pir_images 挂原图
    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)  # llm_select：改写为 (未识别)

    # 当前回合消息进 user_prompt → ON_LLM_REQUEST 扫描：
    # 原图可追溯（describe_image 可用）→ 必须保持 (未识别)，不得误标"已过期"
    # 注意：user_prompt 扫描用 isinstance(p, Prompt)，必须用真实 Prompt 实例
    mod, _ = load_plugin()
    req, tool_set = _mk_tool_req()
    fake_prompt = mod.Prompt(content=batch_msg.message_str, name="message")
    req.user_prompt.append(fake_prompt)
    await plug.on_llm_request(batch_ev, req)
    _check(f"[Image #{md5[:8]}: (未识别)]" in fake_prompt.content,
           f"should stay unidentified: {fake_prompt.content!r}")
    _check("已过期" not in fake_prompt.content,
           f"must not be marked expired: {fake_prompt.content!r}")
    _check(vlm.call_count == 0, f"llm_select must not VLM: {vlm.call_count}")


@_test("T70: llm_select 历史空标识符（不可追溯）→ 已过期（回归）")
async def _t70():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    # 历史消息里的空标识符：无原图（不在 _pir_images）、无 id_map、无缓存
    hist_msg = types.SimpleNamespace(
        content="[Image #deadbeef: ]",
        role="user",
    )
    req, tool_set = _mk_tool_req(messages=[hist_msg])
    fake_prompt = types.SimpleNamespace(name="chat_env", content="")
    req.system_prompt.append(fake_prompt)
    batch_ev = FakeMessageBatchEvent([])
    await plug.on_llm_request(batch_ev, req)
    _check("[Image #deadbeef: (已过期)]" in hist_msg.content,
           f"history should be expired: {hist_msg.content!r}")
    _check(vlm.call_count == 0, f"must not VLM: {vlm.call_count}")


@_test("T71: llm_select hash 失败图（noid_）当前回合保持空")
async def _t71():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})

    class _BrokenImage(Image):
        async def hash_image(self):
            raise RuntimeError("chaos hash")

    img = _BrokenImage(md5="t71_md5_00000000000000000000000a")
    # 群聊（避免私聊单图自动读取干扰本测试意图）
    ev = FakeMessageEvent([img], group=object())
    await plug.on_im_message(ev)  # noid_ 空标识符 + _pir_images 挂原图
    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])
    await plug.on_im_batch_message(batch_ev)  # llm_select：改写为 (未识别)

    req, tool_set = _mk_tool_req()
    mod2, _ = load_plugin()
    fake_prompt = mod2.Prompt(content=batch_msg.message_str, name="message")
    req.user_prompt.append(fake_prompt)
    await plug.on_llm_request(batch_ev, req)
    _check("[Image #noid_" in fake_prompt.content and "(未识别)" in fake_prompt.content,
           f"noid_ should stay unidentified: {fake_prompt.content!r}")
    _check("已过期" not in fake_prompt.content,
           f"must not VLM: {vlm.call_count}")


@_test("T72: describe_image 批量调用（多 id 并行）")
async def _t72():
    db = FakeDB()
    vlm = FakeVLM("批量描述")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5s = [f"t72_{i}_" + "0" * 24 for i in range(3)]
    imgs = [Image(md5=m) for m in md5s]
    ev = FakeMessageEvent(imgs)
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    batch_ev = FakeMessageBatchEvent([batch_msg])

    # 一次调用传 3 个 id（逗号分隔）→ 并行 VLM，返回逐图描述
    desc = await plug.describe_image(
        batch_ev, ",".join(m[:8] for m in md5s))
    _check(vlm.call_count == 3, f"VLM called {vlm.call_count}, expected 3")
    for m in md5s:
        _check(f"[Image #{m[:8]}: 批量描述]" in desc, f"missing {m[:8]}: {desc!r}")

    # 混合：当前回合 2 个 + 历史 1 个（缓存命中）+ 1 个不可追溯
    db.seed(md5s[2], "历史缓存描述")
    desc2 = await plug.describe_image(
        batch_ev, f"{md5s[0][:8]},{md5s[2][:8]},deadbeef")
    _check(f"[Image #{md5s[0][:8]}: 批量描述]" in desc2, f"got: {desc2!r}")
    _check(f"[Image #{md5s[2][:8]}: 历史缓存描述]" in desc2, f"got: {desc2!r}")
    _check("[Image #deadbeef: 图片已过期或不可追溯]" in desc2, f"got: {desc2!r}")

    # 空参数/纯逗号 → 不可追溯
    _check("图片已过期或不可追溯" in await plug.describe_image(batch_ev, ""),
           "empty arg")
    _check("图片已过期或不可追溯" in await plug.describe_image(batch_ev, " , "),
           "blank arg")


@_test("T73: 私聊单图 + 默认开关 → 阶段2 自动 VLM 填充")
async def _t73():
    db = FakeDB()
    vlm = FakeVLM("私聊描述")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    _check(plug.private_single_auto_read, "default on")
    md5 = "t73_md5_00000000000000000000000a"
    ev = FakeMessageEvent([Image(md5=md5)], group=None)  # 私聊
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(vlm.call_count == 1, f"VLM: {vlm.call_count}")
    _check(f"[Image #{md5[:8]}: 私聊描述]" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")
    _check("(未识别)" not in batch_msg.message_str, "should not be unidentified")


@_test("T74: 私聊单图 + 开关关 → (未识别) 零 VLM")
async def _t74():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select",
                                             "auto_read_config": {
                                                 "private_single_auto_read": False}})
    _check(not plug.private_single_auto_read, "switch off")
    md5 = "t74_md5_00000000000000000000000a"
    ev = FakeMessageEvent([Image(md5=md5)], group=None)  # 私聊
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(vlm.call_count == 0, f"VLM: {vlm.call_count}")
    _check(f"[Image #{md5[:8]}: (未识别)]" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")


@_test("T75: 群聊单图（未@）→ (未识别) 不误读")
async def _t75():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t75_md5_00000000000000000000000a"
    ev = FakeMessageEvent([Image(md5=md5)], group=object(), mentioned=False)
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(vlm.call_count == 0, f"VLM: {vlm.call_count}")
    _check(f"[Image #{md5[:8]}: (未识别)]" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")


@_test("T76: 被@提及 → 自动 VLM；群聊未@ → (未识别)")
async def _t76():
    db = FakeDB()
    vlm = FakeVLM("提及描述")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t76_md5_00000000000000000000000a"
    # 被@：群聊 + mentioned=True → 自动读取
    ev = FakeMessageEvent([Image(md5=md5)], group=object(), mentioned=True)
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(vlm.call_count == 1, f"VLM: {vlm.call_count}")
    _check(f"[Image #{md5[:8]}: 提及描述]" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")
    # 开关关 → (未识别)（用无缓存的新图，避免段1 缓存命中干扰）
    plug2, mod2 = await _make_plugin(db, FakeVLM("d"),
                                     {"load_mode": "llm_select",
                                      "auto_read_config": {
                                          "mention_reply_auto_read": False}})
    md5b = "t76b_md5_" + "0" * 24
    ev2 = FakeMessageEvent([Image(md5=md5b)], group=object(), mentioned=True)
    await plug2.on_im_message(ev2)
    b2 = _make_batch_from_event(ev2)
    await plug2.on_im_batch_message(FakeMessageBatchEvent([b2]))
    _check(f"[Image #{md5b[:8]}: (未识别)]" in b2.message_str,
           f"switch off: {b2.message_str!r}")


@_test("T77: 引用（Reply）含图消息 → 自动 VLM")
async def _t77():
    db = FakeDB()
    vlm = FakeVLM("引用描述")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t77_md5_00000000000000000000000a"
    # 消息含 Reply（引用），Reply.chain 里有图 → 自动读取
    reply_chain = FakeMessageChain([Image(md5=md5), Text("被引用的图")])
    chain = FakeMessageChain([Text("引用了一条消息"), Reply(chain=reply_chain)])
    ev = FakeMessageEvent(chain, group=object(), mentioned=False)
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(vlm.call_count == 1, f"VLM: {vlm.call_count}")
    _check(f"[Image #{md5[:8]}: 引用描述]" in batch_msg.message_str,
           f"got: {batch_msg.message_str!r}")
    # 无引用的普通消息 → (未识别)（用无缓存的新图）
    md5b = "t77b_md5_" + "0" * 24
    ev2 = FakeMessageEvent([Image(md5=md5b)], group=object(), mentioned=False)
    await plug.on_im_message(ev2)
    b2 = _make_batch_from_event(ev2)
    await plug.on_im_batch_message(FakeMessageBatchEvent([b2]))
    _check(f"[Image #{md5b[:8]}: (未识别)]" in b2.message_str,
           f"no reply: {b2.message_str!r}")

    # 引用内容多图（grill 细化：仅引用单图时默认读取）→ (未识别)
    md5c = "t77c_md5_" + "0" * 24
    md5d = "t77d_md5_" + "0" * 24
    reply_chain2 = FakeMessageChain([Image(md5=md5c), Image(md5=md5d)])
    chain2 = FakeMessageChain([Reply(chain=reply_chain2)])
    ev3 = FakeMessageEvent(chain2, group=object(), mentioned=False)
    await plug.on_im_message(ev3)
    b3 = _make_batch_from_event(ev3)
    await plug.on_im_batch_message(FakeMessageBatchEvent([b3]))
    _check(f"[Image #{md5c[:8]}: (未识别)]" in b3.message_str
           and f"[Image #{md5d[:8]}: (未识别)]" in b3.message_str,
           f"reply multi-image should stay unidentified: {b3.message_str!r}")


@_test("T78: 私聊多图 → 不自动读取（非单图）")
async def _t78():
    db = FakeDB()
    vlm = FakeVLM("desc")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5s = [f"t78_{i}_" + "0" * 24 for i in range(2)]
    ev = FakeMessageEvent([Image(md5=m) for m in md5s], group=None)  # 私聊但 2 图
    await plug.on_im_message(ev)
    batch_msg = _make_batch_from_event(ev)
    await plug.on_im_batch_message(FakeMessageBatchEvent([batch_msg]))
    _check(vlm.call_count == 0, f"VLM: {vlm.call_count}")
    for m in md5s:
        _check(f"[Image #{m[:8]}: (未识别)]" in batch_msg.message_str,
               f"got: {batch_msg.message_str!r}")


@_test("T79: 私聊批次多消息单图 → 自动读取；批次两图 → 不读")
async def _t79():
    db = FakeDB()
    vlm = FakeVLM("批次描述")
    plug, mod = await _make_plugin(db, vlm, {"load_mode": "llm_select"})
    md5 = "t79_md5_00000000000000000000000a"

    # 私聊批次：文字消息 + 图消息（flush 聚合）→ 批次 1 图 → 自动读取
    ev_t = FakeMessageEvent([Text("文字消息")], group=None)
    await plug.on_im_message(ev_t)
    ev_i = FakeMessageEvent([Image(md5=md5)], group=None)
    await plug.on_im_message(ev_i)
    b_t = _make_batch_from_event(ev_t)
    b_i = _make_batch_from_event(ev_i)
    batch_ev = FakeMessageBatchEvent([b_t, b_i])
    await plug.on_im_batch_message(batch_ev)
    _check(vlm.call_count == 1, f"batch single image should VLM: {vlm.call_count}")
    _check(f"[Image #{md5[:8]}: 批次描述]" in b_i.message_str,
           f"got: {b_i.message_str!r}")

    # 私聊批次：两条图消息（全新 md5，避免第一段缓存干扰）
    # → 批次 2 图 → 不自动读取
    md5b = "t79b_md5_" + "0" * 24
    md5c = "t79c_md5_" + "0" * 24
    plug2, mod2 = await _make_plugin(db, FakeVLM("d"),
                                     {"load_mode": "llm_select"})
    ev1 = FakeMessageEvent([Image(md5=md5b)], group=None)
    await plug2.on_im_message(ev1)
    ev2 = FakeMessageEvent([Image(md5=md5c)], group=None)
    await plug2.on_im_message(ev2)
    c1 = _make_batch_from_event(ev1)
    c2 = _make_batch_from_event(ev2)
    await plug2.on_im_batch_message(FakeMessageBatchEvent([c1, c2]))
    _check(f"[Image #{md5b[:8]}: (未识别)]" in c1.message_str
           and f"[Image #{md5c[:8]}: (未识别)]" in c2.message_str,
           f"batch 2 images should stay unidentified: "
           f"{c1.message_str!r} | {c2.message_str!r}")


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
    # 边界条件
    _t33, _t34, _t35, _t36, _t37, _t38, _t39, _t40,
    # 乐观加载（v2.2.0）
    _t41, _t42, _t43, _t44, _t45, _t46,
    # llm_select 模式（v2.3.0）
    _t47, _t48, _t49, _t50, _t51, _t52, _t53, _t54, _t55, _t56,
    # 多场景 / 换态矩阵（v2.3.0）
    _t57, _t58, _t59, _t60,
    # 混沌测试发现的防御修复（v2.4.0）
    _t61,
    # 标识符注入防御（v2.4.0）
    _t62,
    # 审查修复：真实路径 FIFO（v2.4.0）
    _t63,
    # Forward 嵌套/成环（issue #1）
    _t64, _t65,
    # 审查补漏：Reply 内 Forward / 超深嵌套（issue #1）
    _t66, _t67,
    # 转发展开层数配置（v2.4.2）
    _t68,
    # llm_select 误标已过期（v2.4.3）
    _t69,
    # v2.4.3 review 补测：历史已过期回归 / noid_ 保持空
    _t70, _t71,
    # 工具批量调用（v2.4.4）
    _t72,
    # 自动读取矩阵（v2.4.5）：私聊单图/被@/引用 × 开关 × 多图
    _t73, _t74, _t75, _t76, _t77, _t78,
    # grill 细化（v2.4.6）：私聊批次单图
    _t79,
]


def main():
    global _PASS, _FAIL, _SKIP
    print(f"\nParallel Image Reader v2.3.0 — 三模式架构 + 换态矩阵测试\n")
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
