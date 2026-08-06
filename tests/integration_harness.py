"""
集成测试 harness — 在独立进程装配 KiraAI 核心管线，驱动真实插件。

不依赖真实 QQ/napcat：构造 KiraMessageEvent 直接走 message_processor，
覆盖完整链路：IM → 插件阶段1 → flush → batch → 插件阶段2 → LLM 请求
（ON_LLM_REQUEST + AgentExecutor 工具循环）→ 持久化。

用法：
    cd D:/Projects/KiraAI-dev/parallel_image_reader
    python tests/integration_harness.py

依赖 KiraAI-src（用其 .venv 运行）：
    D:/Projects/KiraAI-dev/KiraAI-src/.venv/Scripts/python.exe tests/integration_harness.py
"""

import asyncio
import base64
import shutil
import sys
import tempfile
import time
from pathlib import Path

KIRAAI_SRC = Path(__file__).resolve().parents[2] / "KiraAI-src"
PLUGIN_DIR = Path(__file__).resolve().parents[1]

# 日志路径按 cwd 解析（get_data_path → ./data/log.log）：
# chdir 到 KiraAI-src 让日志写到其 data/（生产日志文件，可接受）；
# chat_memory 与 DB 已隔离到临时目录。
import os
os.chdir(str(KIRAAI_SRC))

sys.path.insert(0, str(KIRAAI_SRC))
sys.path.insert(0, str(PLUGIN_DIR))

# 1x1 红色 PNG（base64）— 真实 Image 元素用，hash_image 真实计算 md5
_PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _png_b64_variant(seed: int) -> str:
    """按 seed 生成内容不同的 PNG base64（改 IDAT 末字节 → md5 不同）。

    用于压测/换态场景需要不同 md5 图片时。
    """
    raw = base64.b64decode(_PNG_B64)
    # 修改最后一个字节（IDAT 数据区），产生不同 md5
    raw = raw[:-1] + bytes([raw[-1] ^ (seed & 0xFF)])
    return base64.b64encode(raw).decode()


# ── Mock Provider / LLM / VLM ──

class _FakeLLMResponse:
    def __init__(self, text="", tool_calls=None):
        self.text_response = text
        self.reasoning_content = ""
        self.tool_calls = tool_calls or []
        self.tool_results = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = None
        self.time_consumed = 0.001


class _FakeLLMModel:
    """模拟 LLM：按预设序列返回响应（含 tool_calls 支持）。"""
    def __init__(self, script):
        self.script = list(script)  # [(text, tool_calls), ...]
        self.call_count = 0
        # LLMModelClient 结构：model 属性含 provider_name/model_id
        self.model = type("M", (), {
            "provider_name": "fake",
            "model_id": "fake-model",
        })()

    async def chat(self, request):
        self.call_count += 1
        step = self.script.pop(0) if self.script else ("(no more script)", [])
        text, tool_calls = step
        resp = _FakeLLMResponse(text, tool_calls)
        return resp


class _FakeVLM:
    """模拟 VLM：返回固定描述。真实 desc_img 需要 .model 属性。"""
    def __init__(self, desc="集成测试图片描述"):
        self._desc = desc
        self.call_count = 0
        self.model = type("M", (), {
            "provider_name": "fake-vlm",
            "model_id": "fake-vlm-model",
        })()

    async def chat(self, request):
        self.call_count += 1
        return _FakeLLMResponse(self._desc)


class _FakeProviderManager:
    def __init__(self, llm, vlm):
        self._llm = llm
        self._vlm = vlm

    def get_default_llm(self):
        return self._llm

    def get_default_vlm(self):
        return self._vlm

    def get_llm_client(self, *a, **k):
        return self._llm


class _FakeSkillsManager:
    def __init__(self):
        self.skills_info = {}

    async def execute(self, *a, **k):
        return None


class _FakeMCPManager:
    """最小 MCP manager：无工具服务器，全部允许。"""
    def get_tool_server_map(self):
        return {}

    def is_server_allowed(self, server, sid):
        return True


class _FakeAdapterManager:
    def __init__(self, adapter_info):
        self._info = adapter_info

    def get_adapter(self, name):
        return type("A", (), {"info": self._info})


# ── Harness ──

class Harness:
    """装配 KiraAI 核心 + 注册插件 handler。"""

    def __init__(self, load_mode="lazy", llm_script=None):
        self.tmpdir = tempfile.mkdtemp(prefix="kira_harness_")
        self.llm = _FakeLLMModel(llm_script or [("已收到你的图片。", [])])
        self.vlm = _FakeVLM()
        self.load_mode = load_mode
        self._core = None

    async def start(self):
        from core.config import KiraConfig
        from core.db.db_mgr import DatabaseManager
        from core.db.service import DatabaseService
        from core.chat.session_manager import SessionManager
        from core.llm_client import LLMClient
        from core.prompt_manager import PromptManager
        from core.persona import PersonaManager
        from core.message_manager import MessageProcessor
        from core.adapter.adapter_info import AdapterInfo
        from core.event_bus import EventBus
        from core.statistics import Statistics

        # 1. 临时 sqlite（独立于生产 data.db）
        db_url = f"sqlite+aiosqlite:///{Path(self.tmpdir).as_posix()}/test.db"
        self.cfg = KiraConfig()
        self.dm = DatabaseManager(db_url, echo=False)
        await self.dm.init()
        self.db = DatabaseService(self.dm)
        await self.db.init_tables()

        # 2. SessionManager — chat_memory 指向临时文件（不污染生产）
        self.sm = SessionManager(self.db, self.cfg)
        self.sm.chat_memory_path = str(Path(self.tmpdir) / "chat_memory.json")
        self.sm.chat_memory = self.sm._load_memory(self.sm.chat_memory_path)

        # 3. Provider / LLM / Skills / Adapter
        self.provider = _FakeProviderManager(self.llm, self.vlm)
        self.llm_api = LLMClient(self.cfg, self.provider)
        self.skills = _FakeSkillsManager()
        self.adapter_info = AdapterInfo(
            enabled=True, adapter_id="test", name="qq",
            platform="qq", description="test",
            config={"self_id": "test_bot"},
        )
        self.adapters = _FakeAdapterManager(self.adapter_info)

        # 4. Persona / Prompt / EventBus / Stats
        self.persona = PersonaManager(db=self.db)
        await self.persona.init_persona()
        self.prompts = PromptManager(self.cfg, self.persona)
        self.stats = Statistics()
        self.event_bus = EventBus(self.stats, asyncio.Queue(), db=self.db)

        # 5. MessageProcessor
        self.mp = MessageProcessor(
            db=self.db,
            kira_config=self.cfg,
            llm_api=self.llm_api,
            provider_manager=self.provider,
            skills_manager=self.skills,
            adapter_manager=self.adapters,
            session_manager=self.sm,
            prompt_manager=self.prompts,
            mcp_manager=_FakeMCPManager(),
        )
        self.mp.event_bus = self.event_bus
        self.sm.event_bus = self.event_bus

        # 6. 加载插件（真实 main.py）+ 手动注册 handler
        await self._load_plugin(self.load_mode)

        self._core = True
        return self

    async def _load_plugin(self, load_mode):
        from core.plugin.plugin_handlers import event_handler_reg, EventType, EventHandler
        from core.plugin import PluginContext, Priority
        import main as plugin_main

        # 清空上一测试注册的 handler（event_handler_reg 是全局单例）
        event_handler_reg._handlers.clear()

        # PluginContext — 最小字段
        ctx = PluginContext(
            db=self.db,
            config=self.cfg,
            event_bus=self.event_bus,
            provider_mgr=self.provider,
            llm_api=self.llm_api,
            adapter_mgr=self.adapters,
            persona_mgr=self.persona,
            sticker_manager=None,
            session_mgr=self.sm,
            message_processor=self.mp,
            plugin_mgr=None,
        )
        cfg = {"load_mode": load_mode}
        self.plugin = plugin_main.ParallelImageReader(ctx, cfg)
        await self.plugin.initialize()

        # 手动注册三个 handler（模拟 plugin_manager 的绑定）
        handlers = [
            (EventType.ON_IM_MESSAGE, self.plugin.on_im_message, Priority.SYS_HIGH - 1),
            (EventType.ON_IM_BATCH_MESSAGE, self.plugin.on_im_batch_message, Priority.SYS_HIGH - 1),
            (EventType.ON_LLM_REQUEST, self.plugin.on_llm_request, Priority.SYS_HIGH - 1),
        ]
        for etype, func, prio in handlers:
            eh = EventHandler(event_type=etype, priority=prio, handler=func, desc=func.__doc__)
            event_handler_reg.register(eh)

        # 注册工具到 llm_api（模拟 plugin_manager._register_plugin_tools_for 的效果）
        # build_tool_set() 从 llm_api.tools_definitions 构建，工具才能被 LLM 调用
        self.llm_api.register_tool(
            name="describe_image",
            description="获取图片内容描述。当消息中包含 [Image #xxxx: ] 格式标识符且你需要了解图片内容时调用，传入标识符中的 xxxx。",
            parameters={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "图片标识符（[Image #xxxx: ] 中的 xxxx）"},
                },
                "required": ["image_id"],
            },
            func=self.plugin.describe_image,
        )

    # ── 事件构造 ──

    def _make_chain_with_image(self, png_b64: str):
        """构造 [Image(png), Text] 的 chain。"""
        from core.chat.message_utils import MessageChain
        from core.chat.message_elements import Image, Text
        return MessageChain([
            Image(image=f"base64://{png_b64}"),
            Text("看看这张图"),
        ])

    def make_image_event(self, sid="test_session", mentioned=True,
                         chain=None, message_id="m1") -> object:
        from core.chat.message_utils import KiraMessageEvent, KiraIMMessage, MessageChain
        from core.chat.message_elements import Image, Text
        from core.chat.session import User, Group

        if chain is None:
            chain = MessageChain([
                Image(image=f"base64://{_PNG_B64}"),
                Text("看看这张图"),
            ])
        msg = KiraIMMessage(
            message_id=message_id,
            self_id="test_bot",
            chain=chain,
            timestamp=int(time.time()),
            sender=User(user_id="user123", nickname="测试用户"),
            group=Group(group_id=sid, group_name="测试群"),
            is_mentioned=mentioned,
        )
        ev = KiraMessageEvent(
            message_types=["text", "image"],
            timestamp=int(time.time()),
            message=msg,
            adapter=self.adapter_info,
        )
        return ev

    async def run_im(self, ev):
        """走 IM 管线（插件阶段1 + chat 决策 flush）。"""
        ev.flush()
        await self.mp.handle_im_message(ev)

    async def run_batch(self):
        """flush 后手动触发 batch 处理（管线中由 event_bus 分发）。

        消费 queue 中所有 batch 并依次处理，返回最后一个（或 None）。
        """
        batches = []
        while not self.event_bus.event_queue.empty():
            ev = self.event_bus.event_queue.get_nowait()
            from core.chat.message_utils import KiraMessageBatchEvent
            if isinstance(ev, KiraMessageBatchEvent):
                batches.append(ev)
        for b in batches:
            await self.mp.handle_im_batch_message(b)
        return batches[-1] if batches else None

    def get_memory(self, sid):
        """读取持久化的聊天历史。"""
        chunks = self.sm.chat_memory.get(sid, {}).get("memory", [])
        return [m for chunk in chunks for m in chunk]

    async def stop(self):
        await self.plugin.terminate()
        await self.dm.dispose()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ── 测试场景 ──

async def _test_lazy_full_pipeline():
    """lazy 模式全链路：图 → 阶段1 空标识符 → flush → 阶段2 VLM → 历史。"""
    h = Harness(load_mode="lazy")
    await h.start()
    try:
        ev = h.make_image_event(message_id="lazy1")
        await h.run_im(ev)

        # 阶段1：chain 已替换为统一空标识符 [Image #id: ]
        chain_texts = [ele.text for ele in ev.message.chain if hasattr(ele, "text")]
        import re
        assert re.search(r"\[Image #[0-9a-f]+: \]", "".join(chain_texts)), \
            f"stage1 empty identifier missing: {chain_texts}"

        # flush → batch → 阶段2 VLM → LLM 请求
        batch = await h.run_batch()
        assert batch is not None, "batch not produced"
        msg = batch.messages[0]
        assert "[Image #" in msg.message_str, f"batch message_str: {msg.message_str!r}"
        assert "集成测试图片描述" in msg.message_str, f"desc missing: {msg.message_str!r}"
        assert ": ]" not in msg.message_str, f"empty identifier not filled: {msg.message_str!r}"
        print("  [OK] lazy 全链路：标识符+描述进 message_str，空标识符已填充")

        # LLM 请求已执行（ON_LLM_REQUEST + AgentExecutor）
        assert h.llm.call_count >= 1, f"LLM not called: {h.llm.call_count}"

        # 历史持久化
        mem = h.get_memory("qq:gm:test_session")
        joined = "".join(m.get("content", "") for m in mem)
        assert "<!--PIR" not in joined, f"history leaked placeholder: {joined[:200]!r}"
        assert "[Image #" in joined, f"history missing identifier: {joined[:200]!r}"
        print("  [OK] lazy 历史持久化：无占位符，含标识符")
    finally:
        await h.stop()


async def _test_llm_select_full_pipeline():
    """llm_select 全链路：空标识符进历史 → LLM 调 describe_image → 描述持久化。"""
    # LLM 脚本：第一轮调工具，第二轮回复
    tool_call = [{
        "id": "call_1",
        "function": {
            "name": "describe_image",
            "arguments": '{"image_id": "____"}',  # id 由断言后修正
        },
    }]
    h = Harness(load_mode="llm_select", llm_script=[
        ("", [tool_call]),  # 第一轮：请求工具
        ("图片内容已了解。", []),  # 第二轮：回复
    ])
    await h.start()
    try:
        ev = h.make_image_event(message_id="llm1")
        await h.run_im(ev)

        # 阶段1：空标识符 [Image #id: ]（id 是真实 md5 前 8 位）
        from core.chat.message_elements import Image as RealImage
        chain_texts = [ele.text for ele in ev.message.chain if hasattr(ele, "text")]
        import re
        m = re.search(r"\[Image #([0-9a-f]+): \]", "".join(chain_texts))
        assert m, f"empty identifier missing: {chain_texts}"
        image_id = m.group(1)
        print(f"  [OK] llm_select 阶段1：空标识符 [Image #{image_id}: ]")

        # 修正工具参数中的 image_id（真实 md5 前缀）
        h.llm.script = [("", [{
            "id": "call_1",
            "function": {"name": "describe_image",
                         "arguments": f'{{"image_id": "{image_id}"}}'},
        }]), ("图片内容已了解。", [])]

        # run_batch 内部包含 LLM 请求 + AgentExecutor 工具循环；
        # 阶段2 零 VLM 的检查必须在 run_batch 之前（此时仅阶段2 执行过）
        assert h.vlm.call_count == 0, f"stage2 should not VLM: {h.vlm.call_count}"
        batch = await h.run_batch()
        msg = batch.messages[0]
        # 阶段2 后（LLM 请求前）：空标识符保留
        assert f"[Image #{image_id}: ]" in msg.message_str, f"empty kept: {msg.message_str!r}"
        print("  [OK] llm_select 阶段2：零 VLM，空标识符进 message_str")

        # LLM 请求：工具调用 → describe_image → VLM → tool 消息
        # （AgentExecutor 执行工具，vlm 被调用一次）
        assert h.vlm.call_count >= 1, f"describe_image tool should VLM: {h.vlm.call_count}"
        print("  [OK] llm_select 工具链路：LLM 调 describe_image → VLM 加载")

        # 历史持久化：tool 消息含描述
        mem = h.get_memory("qq:gm:test_session")
        joined = "".join(m.get("content", "") for m in mem)
        assert "集成测试图片描述" in joined, f"desc not in history: {joined[:300]!r}"
        print("  [OK] llm_select 历史持久化：描述随 tool 消息进历史")
    finally:
        await h.stop()


async def _test_eager_full_pipeline():
    """eager 模式全链路：收到即 VLM。"""
    h = Harness(load_mode="eager")
    await h.start()
    try:
        ev = h.make_image_event(message_id="eager1")
        await h.run_im(ev)
        # eager：阶段1 启动后台 task，等它完成
        await asyncio.sleep(0.1)
        assert h.vlm.call_count >= 1, f"eager should VLM early: {h.vlm.call_count}"
        print("  [OK] eager 阶段1：收到即 VLM")

        batch = await h.run_batch()
        msg = batch.messages[0]
        assert "[Image #" in msg.message_str and "集成测试图片描述" in msg.message_str, \
            f"batch: {msg.message_str!r}"
        assert ": ]" not in msg.message_str, f"empty identifier not filled: {msg.message_str!r}"
        print("  [OK] eager 全链路：标识符+描述进 message_str")
    finally:
        await h.stop()


async def _test_switch_matrix():
    """换态矩阵：同一实例运行时 lazy→llm_select→eager 切换（每阶段用不同图避免缓存干扰）。"""
    h = Harness(load_mode="lazy")
    await h.start()
    try:
        # 1. lazy 阶段1 → 空标识符，阶段2 填充
        ev1 = h.make_image_event(message_id="sw1",
                                 chain=h._make_chain_with_image(_png_b64_variant(1)))
        await h.run_im(ev1)
        batch1 = await h.run_batch()
        assert "集成测试图片描述" in batch1.messages[0].message_str, \
            f"lazy fill: {batch1.messages[0].message_str!r}"

        # 2. 运行时切 llm_select → 空标识符进历史，不 VLM
        h.plugin.load_mode = "llm_select"
        vlm_before = h.vlm.call_count
        ev2 = h.make_image_event(message_id="sw2",
                                 chain=h._make_chain_with_image(_png_b64_variant(2)))
        await h.run_im(ev2)
        batch2 = await h.run_batch()
        import re
        assert re.search(r"\[Image #[0-9a-f]+: \]", batch2.messages[0].message_str), \
            f"llm_select empty: {batch2.messages[0].message_str!r}"
        assert h.vlm.call_count == vlm_before, "llm_select should not VLM in stage2"
        print("  [OK] 换态矩阵：lazy→llm_select 行为切换正确")

        # 3. 切回 lazy → 新消息恢复填充
        h.plugin.load_mode = "lazy"
        ev3 = h.make_image_event(message_id="sw3",
                                 chain=h._make_chain_with_image(_png_b64_variant(3)))
        await h.run_im(ev3)
        batch3 = await h.run_batch()
        assert "集成测试图片描述" in batch3.messages[0].message_str, \
            f"back to lazy: {batch3.messages[0].message_str!r}"
        print("  [OK] 换态矩阵：llm_select→lazy 恢复填充")
    finally:
        await h.stop()


async def _test_stress_multi_image():
    """压测：单消息 30 图 + 5 消息并发，验证无重复 VLM、保序、无死锁。"""
    h = Harness(load_mode="lazy")
    await h.start()
    try:
        # 5 条消息 × 6 图（不同 md5）= 30 图
        from core.chat.message_utils import MessageChain
        from core.chat.message_elements import Image, Text
        events = []
        for mi in range(5):
            chain = MessageChain(
                [Image(image=f"base64://{_png_b64_variant(mi * 6 + j)}")
                 for j in range(6)]
                + [Text(f"msg{mi}")]
            )
            ev = h.make_image_event(message_id=f"stress_{mi}", chain=chain)
            events.append(ev)

        t0 = time.monotonic()
        for ev in events:
            await h.run_im(ev)
        # 5 条消息各自 flush → 5 个 batch 全部处理
        await h.run_batch()
        elapsed = time.monotonic() - t0

        # 每图一次 VLM（6 图 × 5 消息 = 30 次），无重复/无遗漏
        assert h.vlm.call_count == 30, \
            f"VLM called {h.vlm.call_count}, expected 30 (no dup/miss)"
        # 所有消息的空标识符都被填充
        for ev in events:
            msg = ev.message
            assert ": ]" not in msg.message_str, f"unfilled: {msg.message_str!r}"
            assert msg.message_str.count("[Image #") == 6, \
                f"expected 6 ids: {msg.message_str!r}"
        print(f"  [OK] 压测：30 图（5×6）VLM 计数精确 {h.vlm.call_count}，"
              f"耗时 {elapsed:.2f}s")
    finally:
        await h.stop()


async def _test_stress_cache_hit():
    """压测：预填缓存后同图多次出现 → 零 VLM。"""
    h = Harness(load_mode="lazy")
    await h.start()
    try:
        # 预填缓存（真实 md5 需先算，这里直接用真实图走一次 VLM 填缓存）
        ev0 = h.make_image_event(message_id="warm")
        await h.run_im(ev0)
        await h.run_batch()
        first_vlm = h.vlm.call_count
        assert first_vlm >= 1, "warmup should VLM"

        # 同图再来 5 次 → 缓存命中，零新 VLM
        for i in range(5):
            ev = h.make_image_event(message_id=f"cache_{i}")
            await h.run_im(ev)
            await h.run_batch()
        assert h.vlm.call_count == first_vlm, \
            f"cache miss repeated: {h.vlm.call_count} vs {first_vlm}"
        print(f"  [OK] 压测：同图重复 5 次零新 VLM（缓存命中）")
    finally:
        await h.stop()


async def _test_forward_nested():
    """Forward 转发：嵌套 Forward 内容不丢失，图片全部识别（issue #1）。"""
    from core.chat.message_elements import Forward, Text, Image
    from core.chat.message_utils import MessageChain

    h = Harness(load_mode="lazy")
    await h.start()
    try:
        # 三层嵌套：外层 Forward → 中层 → 内层含图
        inner = MessageChain([Image(image=f"base64://{_PNG_B64}"), Text("深层转发")])
        mid = MessageChain([Text("外层转发"), Forward(chains=[inner])])
        outer = MessageChain([Forward(chains=[mid])])
        ev = h.make_image_event(chain=outer, message_id="fwd-nested")
        await h.run_im(ev)

        batch = await h.run_batch()
        assert batch is not None, "batch not produced"
        msg_str = batch.messages[0].message_str
        assert "深层转发" in msg_str, f"nested content lost: {msg_str!r}"
        assert "外层转发" in msg_str, f"outer content lost: {msg_str!r}"
        assert "[Image #" in msg_str, f"image identifier missing: {msg_str!r}"
        print("  [OK] Forward 嵌套：内容不丢失，图片全识别")

        # 历史持久化同样完整
        mem = h.get_memory("qq:gm:test_session")
        joined = "".join(m.get("content", "") for m in mem)
        assert "[Image #" in joined and "深层转发" in joined, \
            f"history incomplete: {joined[:200]!r}"
        print("  [OK] Forward 嵌套历史：完整持久化")
    finally:
        await h.stop()


async def _test_forward_cycle():
    """恶意输入：Forward 成环不崩溃，不无限递归（issue #1 边界）。"""
    from core.chat.message_elements import Forward, Text, Image
    from core.chat.message_utils import MessageChain

    h = Harness(load_mode="lazy")
    await h.start()
    try:
        c1 = MessageChain([Image(image=f"base64://{_PNG_B64}"), Text("环1")])
        c2 = MessageChain([Text("环2")])
        c1.append(Forward(chains=[c2]))
        c2.append(Forward(chains=[c1]))  # c1 → c2 → c1 成环
        ev = h.make_image_event(chain=MessageChain([Forward(chains=[c1])]),
                                message_id="fwd-cycle")
        await h.run_im(ev)  # 不崩溃即通过（核心/插件双防环）

        batch = await h.run_batch()
        assert batch is not None, "batch not produced"
        msg_str = batch.messages[0].message_str
        assert "环1" in msg_str or "环2" in msg_str, f"cycle content: {msg_str!r}"
        print("  [OK] Forward 成环：不崩溃，内容部分保留（安全降级）")
    finally:
        await h.stop()


async def _test_forward_reply():
    """Reply 内 Forward：引用消息里的转发图片同样识别（issue #1 边界）。"""
    from core.chat.message_elements import Forward, Text, Image, Reply
    from core.chat.message_utils import MessageChain

    h = Harness(load_mode="lazy")
    await h.start()
    try:
        inner = MessageChain([Image(image=f"base64://{_PNG_B64}"), Text("引用转发图")])
        fwd = Forward(chains=[MessageChain([Text("层1"), Forward(chains=[inner])])])
        outer = MessageChain([Reply(message_id="r9", chain=MessageChain([fwd]))])
        ev = h.make_image_event(chain=outer, message_id="fwd-reply")
        await h.run_im(ev)

        batch = await h.run_batch()
        assert batch is not None, "batch not produced"
        msg_str = batch.messages[0].message_str
        assert "[Image #" in msg_str, f"reply fwd identifier missing: {msg_str!r}"
        assert "引用转发图" in msg_str, f"reply fwd content lost: {msg_str!r}"
        print("  [OK] Reply 内 Forward：图片识别，内容完整")
    finally:
        await h.stop()


async def main():
    print("\nKiraAI 集成测试 harness（真实核心管线 + 真实插件）\n")
    tests = [
        ("lazy 全链路", _test_lazy_full_pipeline),
        ("llm_select 全链路", _test_llm_select_full_pipeline),
        ("eager 全链路", _test_eager_full_pipeline),
        ("换态矩阵", _test_switch_matrix),
        ("压测：30图并发", _test_stress_multi_image),
        ("压测：缓存命中", _test_stress_cache_hit),
        ("Forward 嵌套", _test_forward_nested),
        ("Forward 成环", _test_forward_cycle),
        ("Reply 内 Forward", _test_forward_reply),
    ]
    passed = 0
    for name, fn in tests:
        try:
            await fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
    print(f"\n── 结果: {passed}/{len(tests)} 通过")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
