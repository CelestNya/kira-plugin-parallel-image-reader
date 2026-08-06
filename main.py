"""
并行识图插件 v2.4.0 — Parallel Image Reader

统一标识符架构，既不污染聊天历史，也不浪费 VLM 调用：

1. ON_IM_MESSAGE（优先级 SYS_HIGH-1，早于 chat 插件 HIGH）
   递归遍历消息链，把 Image/Sticker 替换为统一标识符 [Image #id: 内容]
   （阻止本体 message_format_to_text 串行调 VLM）。三模式阶段1 完全统一：
   缓存命中 → 带描述标识符（最终态）；未命中 → 空标识符（待填充态）+
   _pir_images 暂存原图。不干预 event 策略，被 discard 的消息
   lazy/llm_select 零 VLM 开销。

2. ON_IM_BATCH_MESSAGE（优先级 SYS_HIGH-1）
   此时消息已确定会发给 LLM，本体 message_format_to_text 已执行（标识
   Text 不触发 VLM）。统一语义：执行本消息的图片描述任务并填充——
   lazy 现场 VLM、eager await 提前启动的 task、llm_select 不执行（空
   标识符合法进历史）。message_str 在持久化之前被修正，空标识符不残留。

三种加载模式（load_mode）：
- lazy（默认）：触发时才并行 VLM，被 discard 的消息零 VLM 开销
- eager：ON_IM_MESSAGE 阶段通过 asyncio.create_task 启动后台 VLM task，
  不阻塞 handler 链；ON_IM_BATCH_MESSAGE 阶段 await 复用结果。被 discard
  的消息仍然会完成 VLM 调用（结果写入缓存可被复用），VLM 调用量会增加。
- llm_select：不预调 VLM。空标识符 [Image #id: ] 进历史，LLM 通过
  describe_image 工具按需加载描述。历史标识符在 ON_LLM_REQUEST 阶段
  统一扫描替换为描述或"已过期"。

配置项见 schema.json。
"""

import asyncio
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image as PILImage

from core.plugin import BasePlugin, PluginContext, on, Priority, register, logger
from core.logging_manager import get_logger
from core.chat.message_elements import Image, Text, Sticker, Reply, Forward
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.utils.common_utils import desc_img
from core.provider import LLMRequest
from core.prompt_manager import Prompt

vlm_logger = get_logger("parallel_vlm", "purple")

# ── Prompts (not configurable) ──

DESC_PROMPT = "描述这张图片的内容，如果有文字请将其输出"

# 单次 VLM 调用超时（秒），超时返回空描述 → "(description unavailable)"
VLM_TIMEOUT = 60

# 统一图片标识符：三模式共用格式 [Image #<short_id>: <内容>]
# 内容区三态：空（待填充，llm_select 合法进历史）/ 描述 / 已过期
_IMAGE_RE = re.compile(r"\[Image #([^\]\s:]+): [^\]]*\]")


def _make_image_id(full_md5: str, existing: dict) -> str:
    """取 md5 前缀做短标识符（8 位起），碰撞时逐位加长。纯查询，不写入。

    写入统一走 _id_map_add（带 FIFO 上限）；此函数只检查现有键避免碰撞。
    """
    prefix_len = 8
    short = full_md5[:prefix_len]
    while short in existing and existing[short] != full_md5:
        prefix_len += 2
        short = full_md5[:prefix_len]
    return short


def _is_valid_desc(desc: str) -> bool:
    """检查缓存描述是否有效——排除控制字符、占位符标记与标识符自污染的注入数据。

    注入面：VLM 是第三方服务，输出不可信。若描述含标识符格式（如图片里恰好
    有 "[Image #...]" 文字），会被嵌套进外层标识符，扫描器会误匹配内层空
    标识符并改写描述内容——污染缓存且扩散到所有会话。一律拒绝。
    """
    if not desc:
        return False
    if "\x00" in desc:
        # 旧 caption 方案遗留的 \x00IMG_PENDING_ 污染
        return False
    if "<!--PIR:" in desc:
        # 旧占位符标记的自我污染（理论上不会发生，防御性检查）
        return False
    if "[Image #" in desc:
        # 标识符格式自污染：VLM/用户文本模仿标识符 → 嵌套标识符注入
        return False
    return True


class ParallelImageReader(BasePlugin):
    """并行识图插件 v2.4.0 — 统一标识符架构（lazy / eager / llm_select）"""

    def __init__(self, ctx: PluginContext, cfg: dict):
        super().__init__(ctx, cfg)

        # Config — will be overridden by initialize()
        self.max_concurrent: int = 3
        self.quality_enabled: bool = False
        self.quality_value: int = 85
        self.load_mode: str = "lazy"
        self.id_map_limit: int = 1000

        # 乐观加载 in-flight task 强引用池（防 GC 回收）
        self._optimistic_tasks: set = set()
        self._sem = None  # initialize 中创建

        # short_id → full_md5 映射（llm_select 模式历史反查用）
        self._id_map: dict[str, str] = {}

    @staticmethod
    def _plugin_version() -> str:
        """从 manifest.json 读取版本（日志用，避免硬编码过期）。"""
        try:
            path = Path(__file__).resolve().parent / "manifest.json"
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("version", "?")
        except Exception:
            return "?"

    # ── Lifecycle ──

    async def initialize(self):
        """加载用户配置"""
        self.max_concurrent = self.plugin_cfg.get("max_concurrent", 3)
        self.quality_enabled = self.plugin_cfg.get("quality_enabled", False)
        self.quality_value = self.plugin_cfg.get("quality_value", 85)

        # load_mode 三态：lazy / eager / llm_select
        # 旧配置迁移：v2.2.x 的 eager_loading 开关 → load_mode
        mode = self.plugin_cfg.get("load_mode")
        if mode not in ("lazy", "eager", "llm_select"):
            if self.plugin_cfg.get("eager_loading"):
                mode = "eager"
            else:
                mode = "lazy"
        self.load_mode = mode

        # id_map 上限（顶层配置；兼容旧版 llm_select_config section 内位置）
        llm_cfg = self.plugin_cfg.get("llm_select_config") or {}
        self.id_map_limit = int(
            self.plugin_cfg.get("id_map_limit") or llm_cfg.get("id_map_limit", 1000)
        )

        # 转发展开层数上限（防恶意超深嵌套；深层内容无痕省略）
        self.forward_max_depth = int(
            self.plugin_cfg.get("forward_max_depth")
            or ParallelImageReader._MAX_CHAIN_DEPTH
        )

        self._sem = asyncio.Semaphore(self.max_concurrent)
        self._load_id_map()

        logger.info(
            f"[ParallelImageReader] v{self._plugin_version()} initialized: "
            f"mode={self.load_mode}, max_concurrent={self.max_concurrent}, "
            f"quality={'on(' + str(self.quality_value) + ')' if self.quality_enabled else 'off'}"
        )

    async def terminate(self):
        """取消所有未完成乐观加载 task + 保存 id_map，防泄漏。"""
        for task in list(self._optimistic_tasks):
            task.cancel()
        self._optimistic_tasks.clear()
        self._save_id_map()
        logger.info("[ParallelImageReader] terminated")

    # ── id_map 持久化（llm_select 历史反查）──

    def _id_map_path(self):
        try:
            data_dir = self.ctx.get_plugin_data_dir()
            if data_dir is None:
                return None
            return data_dir / "id_map.json"
        except Exception:
            return None

    def _load_id_map(self):
        path = self._id_map_path()
        if path is None or not path.exists():
            self._id_map = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._id_map = raw if isinstance(raw, dict) else {}
        except Exception as e:
            logger.warning(f"[ParallelImageReader] failed to load id_map: {e}")
            self._id_map = {}

    def _save_id_map(self):
        path = self._id_map_path()
        if path is None:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._id_map, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"[ParallelImageReader] failed to save id_map: {e}")

    def _id_map_add(self, short_id: str, full_md5: str):
        """写入映射，超限按 FIFO 淘汰最早写入的。"""
        if short_id in self._id_map:
            return
        self._id_map[short_id] = full_md5
        if len(self._id_map) > self.id_map_limit:
            # dict 保持插入序，pop 第一个即最早写入
            oldest = next(iter(self._id_map))
            self._id_map.pop(oldest)

    # ── Chain traversal ──

    # 递归遍历的深度上限（默认值）：防恶意超深嵌套（Forward 层层套娃）触发
    # RecursionError。超深时安全降级——深层 Forward 保留原样，由核心过滤
    # （内容无痕省略但不崩溃）。可配置：schema 的 forward_max_depth。
    _MAX_CHAIN_DEPTH = 64

    @staticmethod
    def _flatten_forwards(chain, stack=None, depth=0, max_depth=None):
        """就地拍平 Forward.chains 中的嵌套 Forward，防核心渲染丢内容。

        KiraAI 核心 message_format_to_text 渲染 Forward 时会过滤嵌套
        Forward 元素（[x for x in chain if not isinstance(x, Forward)]，
        防无限递归），导致嵌套转发的内容（含图片标识符）不进 message_str，
        LLM 看不到。插件在阶段1 先把嵌套 Forward 展开为平铺元素。

        语义：depth=0 的顶层 chain 里的 Forward（消息本身是转发）保留壳，
        depth>0 的嵌套 Forward 逐层展开为其子链内容。覆盖路径：
        Forward.chains 与 Reply.chain。防环：stack 记录当前展开路径上的
        chain（id），环中子链保留 Forward 元素（核心过滤兜底）。
        深度上限 max_depth（默认 _MAX_CHAIN_DEPTH）：超限不展开（深层
        内容无痕省略，恶意超深嵌套安全降级）。
        """
        if max_depth is None:
            max_depth = ParallelImageReader._MAX_CHAIN_DEPTH
        if stack is None:
            stack = set()
        cid = id(chain)
        if cid in stack:
            return  # 环：同一展开路径上再次出现
        stack.add(cid)
        i = 0
        while i < len(chain):
            ele = chain[i]
            if isinstance(ele, Reply) and ele.chain is not None:
                if depth < max_depth:
                    ParallelImageReader._flatten_forwards(
                        ele.chain, stack, depth + 1, max_depth
                    )
            elif isinstance(ele, Forward) and ele.chains:
                # 先递归展开每个子链（内部嵌套全部处理）；深度守卫防
                # 恶意超深嵌套 RecursionError（超限子树不展开，语义等价）
                if depth < max_depth:
                    for c in ele.chains:
                        ParallelImageReader._flatten_forwards(
                            c, stack, depth + 1, max_depth
                        )
                if depth > 0 and depth < max_depth:
                    # 嵌套 Forward：展开为其子链内容（平铺替换元素本身）
                    expanded = []
                    for c in ele.chains:
                        if id(c) in stack:
                            continue  # 环：跳过该子链（内容无痕省略）
                        expanded.extend(c)
                    if expanded:
                        chain[i:i + 1] = expanded
                        i += len(expanded) - 1
            i += 1
        stack.remove(cid)

    @staticmethod
    def _walk(chain, visited=None, depth=0, max_depth=None):
        """递归遍历消息链（含 Reply.chain / Forward.chains），yield (chain_ref, index)。

        带环检测：用 id(chain_ref) 标记已访问，避免 Reply 互引导致 RecursionError。
        深度上限 max_depth（默认 _MAX_CHAIN_DEPTH）：超限跳过深层（安全降级）。
        """
        if max_depth is None:
            max_depth = ParallelImageReader._MAX_CHAIN_DEPTH
        if visited is None:
            visited = set()
        cid = id(chain)
        if cid in visited:
            return
        visited.add(cid)
        for i, ele in enumerate(chain):
            if isinstance(ele, (Image, Sticker)):
                yield chain, i
            elif isinstance(ele, Reply) and ele.chain is not None:
                if depth < max_depth:
                    yield from ParallelImageReader._walk(
                        ele.chain, visited, depth + 1, max_depth
                    )
            elif isinstance(ele, Forward) and ele.chains:
                if depth < max_depth:
                    for c in ele.chains:
                        yield from ParallelImageReader._walk(
                            c, visited, depth + 1, max_depth
                        )

    # ── Stage 1: ON_IM_MESSAGE — replace with identifier, stash originals ──

    @on.im_message(priority=Priority.SYS_HIGH - 1)
    async def on_im_message(self, event: KiraMessageEvent):
        """把 Image/Sticker 替换为统一标识符 [Image #id: 内容]，原图暂存 _pir_images。

        三模式阶段1 完全统一：缓存命中 → 带描述标识符（最终态）；
        未命中 → 空标识符 [Image #id: ]（待填充态）+ _pir_images 暂存。
        差异仅在 eager 模式额外启动后台 VLM task。不干预 event 策略，
        被 discard 的消息 lazy/llm_select 零 VLM 开销。
        """
        session_key = event.session.sid if event.session else "default"
        try:
            db = self.ctx.db
            cached_count = 0

            # 先拍平嵌套 Forward（防核心渲染丢内容），再遍历替换图片
            self._flatten_forwards(event.message.chain,
                                   max_depth=self.forward_max_depth)

            for chain_ref, idx in self._walk(event.message.chain,
                                             max_depth=self.forward_max_depth):
                ele = chain_ref[idx]
                try:
                    md5 = await ele.hash_image()
                except Exception as e:
                    logger.debug(
                        f"[ParallelImageReader] hash failed [{session_key}]: "
                        f"{type(e).__name__}: {e}"
                    )
                    md5 = None

                # 查缓存：命中则直接用描述做标识符内容（后续无需 VLM）
                # 防御：描述含控制字符或占位符标记的视为无效（历史污染数据），不当作命中
                desc_cached = None
                if md5:
                    try:
                        entry = await db.get_image_desc_cache(md5)
                        if entry and entry.get("description"):
                            raw_desc = entry["description"]
                            if _is_valid_desc(raw_desc):
                                desc_cached = raw_desc
                                cached_count += 1
                            else:
                                logger.warning(
                                    f"[ParallelImageReader] cache polluted [{session_key}] "
                                    f"md5={md5[:8]}... ignoring invalid cached desc"
                                )
                    except Exception as e:
                        logger.debug(
                            f"[ParallelImageReader] cache query failed [{session_key}]: {e}"
                        )

                # 统一标识符：三模式同路径
                if desc_cached is not None:
                    short_id = _make_image_id(md5, self._id_map)
                    self._id_map_add(short_id, md5)  # 命中路径也写（换态反查）
                    chain_ref[idx] = Text(f"[Image #{short_id}: {desc_cached}]")
                else:
                    if md5:
                        short_id = _make_image_id(md5, self._id_map)
                        self._id_map_add(short_id, md5)  # 三模式都写（换态反查）
                    else:
                        short_id = f"noid_{id(ele)}"
                    chain_ref[idx] = Text(f"[Image #{short_id}: ]")
                    images_map = getattr(event.message, "_pir_images", None)
                    if images_map is None:
                        images_map = {}
                        event.message._pir_images = images_map
                    images_map[short_id] = ele

            # eager：立即启动后台 VLM task（不应 await，handler 立即返回）
            images_map = getattr(event.message, "_pir_images", None)
            if self.load_mode == "eager" and images_map:
                images = list(images_map.values())
                task = asyncio.create_task(
                    self._describe_parallel(images, session_key)
                )
                self._optimistic_tasks.add(task)
                task.add_done_callback(self._optimistic_tasks.discard)
                event.message._pir_optimistic = task
                logger.info(
                    f"[ParallelImageReader] eager VLM started "
                    f"[{session_key}]: {len(images)} images "
                    f"({cached_count} cache-hit)"
                )
            elif images_map:
                logger.info(
                    f"[ParallelImageReader] stashed {len(images_map)} images, "
                    f"{cached_count} cache-hit [{session_key}]"
                )
            if self._id_map:
                # 新标识符已写内存映射，立即落盘（1000 条上限，json.dump 开销可忽略）
                try:
                    self._save_id_map()
                except Exception as e:
                    logger.debug(f"[ParallelImageReader] id_map save failed: {e}")
        except Exception as e:
            logger.error(
                f"[ParallelImageReader] on_im_message failed [{session_key}]: "
                f"{type(e).__name__}: {e}"
            )

    # ── Stage 2: ON_IM_BATCH_MESSAGE — parallel VLM + replace placeholders ──

    @on.im_batch_message(priority=Priority.SYS_HIGH - 1)
    async def on_im_batch_message(self, event: KiraMessageBatchEvent):
        """并行描述暂存图片，替换 message_str 和 chain 中的占位符为 [Image: 描述]。

        此时本体 message_format_to_text 已执行完毕（占位 Text 不触发 VLM）。
        message_str 在持久化之前被修正，占位符不进入历史。
        """
        session_key = event.session.sid if event.session else "default"
        try:
            # 统一语义：收集所有消息的图片描述任务 → 跨消息并行 → 逐消息填充
            groups: list = []  # [(message, images_map, coro/task), ...]
            for message in event.messages:
                images_map = getattr(message, "_pir_images", None)
                if not images_map:
                    continue  # 无图或全缓存命中
                # llm_select：空标识符合法进历史（最终态），不 VLM
                if self.load_mode == "llm_select":
                    continue
                # eager 复用阶段1 提前启动的 task；lazy 现场启动
                task = getattr(message, "_pir_optimistic", None)
                if task is not None:
                    groups.append((message, images_map, task))
                else:
                    groups.append((message, images_map, self._describe_parallel(
                        list(images_map.values()), session_key
                    )))

            if groups:
                # 跨消息并行执行所有 VLM 任务（lazy 多消息也并行，eager task 已跑完则零等待）
                results = await asyncio.gather(
                    *[g[2] for g in groups],
                    return_exceptions=True,
                )
                for (message, images_map, _), descs in zip(groups, results):
                    # _pir_images 为 dict，保序：阶段1 task 与这里 zip 同序
                    if not isinstance(descs, list):
                        descs = [""] * len(images_map)

                    # 填充：精确替换本消息的空标识符
                    fill_map: dict[str, str] = {}
                    for (short_id, _), desc in zip(images_map.items(), descs):
                        fill_map[f"[Image #{short_id}: ]"] = \
                            f"[Image #{short_id}: {desc or '(description unavailable)'}]"

                    if message.message_str:
                        for ph, final in fill_map.items():
                            message.message_str = message.message_str.replace(ph, final)
                    self._replace_in_chain(message.chain, fill_map,
                                            max_depth=self.forward_max_depth)

            mode = "quality" if self.quality_enabled else "native"
            logger.info(
                f"[ParallelImageReader] described images"
                f" ({mode}, concurrency={self.max_concurrent}) [{session_key}]"
            )
        except asyncio.CancelledError:
            # Ctrl+C / 任务取消：空标识符是合法状态，直接重新抛出
            raise
        except Exception as e:
            logger.error(
                f"[ParallelImageReader] on_im_batch_message failed [{session_key}]: "
                f"{type(e).__name__}: {e}"
            )

    # ── Chain 标识符替换（填充）──

    @staticmethod
    def _replace_in_chain(chain, fill_map: dict, visited=None, depth=0, max_depth=None):
        """递归遍历 chain（含 Reply/Forward），把空标识符 Text 替换为带描述 Text。"""
        if max_depth is None:
            max_depth = ParallelImageReader._MAX_CHAIN_DEPTH
        if visited is None:
            visited = set()
        cid = id(chain)
        if cid in visited:
            return
        visited.add(cid)
        for i, ele in enumerate(chain):
            if isinstance(ele, Text) and ele.text in fill_map:
                chain[i] = Text(fill_map[ele.text])
            elif isinstance(ele, Reply) and ele.chain is not None:
                if depth < max_depth:
                    ParallelImageReader._replace_in_chain(
                        ele.chain, fill_map, visited, depth + 1, max_depth
                    )
            elif isinstance(ele, Forward) and ele.chains:
                if depth < max_depth:
                    for c in ele.chains:
                        ParallelImageReader._replace_in_chain(
                            c, fill_map, visited, depth + 1, max_depth
                        )

    # ── System hint + 标识符扫描替换 + 工具增删 ──

    @on.llm_request(priority=Priority.SYS_HIGH - 1)
    async def on_llm_request(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """LLM 请求前：换态工具增删 + 历史标识符扫描替换 + 注入格式 hint。

        - 非 llm_select 模式：移除 describe_image 工具（换态后 LLM 无读取手段）
        - llm_select 模式：describe_image 常驻——有图/无图消息工具集一致，
          避免工具前缀抖动破坏 LLM 上下文缓存（缓存按请求前缀精确匹配）
        - 所有模式：扫描 req.messages + user_prompt 中的 [Image #id: ] 空标识符：
          缓存命中 → 填描述；未命中且可追溯（当前回合有原图）且 lazy/eager →
          触发 VLM；不可追溯 → "已过期"
        """
        try:
            # 1. 换态工具增删（模仿 file 插件 filter_tools 模式）
            if req.tool_set is not None:
                try:
                    if self.load_mode != "llm_select":
                        req.tool_set.remove("describe_image")
                except KeyError:
                    pass  # 工具不在 tool_set 中，忽略

            # 2. 扫描替换 req.messages（历史）与 req.user_prompt（当前消息）中的空标识符
            session_key = event.session.sid if event.session else "default"
            replaced = await self._fill_identifiers(event, req.messages, session_key)
            for p in req.user_prompt:
                if isinstance(p, Prompt) and p.content:
                    new_content = await self._fill_text_identifiers(
                        p.content, event, session_key
                    )
                    if new_content != p.content:
                        p.content = new_content
                        replaced = True

            # 3. 注入格式 hint（统一标识符格式说明）
            if self.load_mode == "llm_select":
                hint = (
                    "当消息中包含 [Image #xxxx: ] 格式的标记时，这表示用户发送了"
                    "一张图片，其内容尚未加载。如需了解图片内容，请调用 "
                    "describe_image 工具并传入标识符中的 xxxx。"
                    "已加载的图片会显示为 [Image #xxxx: 描述内容]。"
                )
            else:
                hint = (
                    "当消息中包含 [Image #xxxx: 描述内容] 格式的标记时，"
                    "这表示用户发送了一张图片，其内容由「描述内容」说明。"
                )
            for p in req.system_prompt:
                if getattr(p, "name", None) == "chat_env":
                    if hint not in p.content:
                        p.content += "\n" + hint
                    break
            return replaced
        except Exception as e:
            logger.warning(
                f"[ParallelImageReader] on_llm_request failed: {type(e).__name__}: {e}"
            )
            return False

    async def _fill_identifiers(self, event, messages: list, session_key: str) -> bool:
        """扫描消息列表中的 [Image #id: ] 空标识符并填充。返回是否替换过。"""
        replaced = False
        for m in messages:
            content = getattr(m, "content", None)
            if not isinstance(content, str) or not content:
                continue
            new_content = await self._fill_text_identifiers(content, event, session_key)
            if new_content != content:
                m.content = new_content
                replaced = True
        return replaced

    async def _fill_text_identifiers(self, text: str, event, session_key: str) -> str:
        """填充单个文本中的所有 [Image #id: ] 空标识符。

        规则：
        - 内容非空（已有描述/已过期）→ 跳过
        - 空内容 + 缓存命中 → 填描述
        - 空内容 + 未命中 + 当前回合有原图（_pir_images）→ 触发 VLM（lazy/eager 换态）
        - 空内容 + 不可追溯 → "已过期"
        """
        # re.sub 无法 await，手写异步替换循环
        parts: list[str] = []
        pos = 0
        for m in _IMAGE_RE.finditer(text):
            parts.append(text[pos:m.start()])
            parts.append(await self._fill_one_identifier(m, event, session_key))
            pos = m.end()
        parts.append(text[pos:])
        return "".join(parts)

    async def _fill_one_identifier(self, m: re.Match, event, session_key: str) -> str:
        short_id = m.group(1)
        # [Image #id: ] 空内容；[Image #id: xxx] 有内容（跳过）
        if m.group(0) != f"[Image #{short_id}: ]":
            return m.group(0)

        # 1. 缓存命中？
        full_md5 = self._id_map.get(short_id)
        if full_md5:
            try:
                entry = await self.ctx.db.get_image_desc_cache(full_md5)
                if entry and entry.get("description") \
                        and _is_valid_desc(entry["description"]):
                    return f"[Image #{short_id}: {entry['description']}]"
            except Exception as e:
                logger.debug(f"[ParallelImageReader] fill cache query failed: {e}")

        # 2. 当前回合有原图 → 可 VLM（lazy/eager 换态填充）
        ele = None
        for msg in event.messages:
            images_map = getattr(msg, "_pir_images", None)
            if images_map and short_id in images_map:
                ele = images_map[short_id]
                break
        if ele is not None:
            if self.load_mode == "llm_select":
                # llm_select：当前回合原图可追溯（describe_image 工具可用），
                # 保持空标识符，不得误标"已过期"
                return m.group(0)
            desc = await self._describe_one(0, ele, session_key, 1)
            if desc:
                return f"[Image #{short_id}: {desc}]"
            return f"[Image #{short_id}: 已过期]"

        # 3. 不可追溯
        return f"[Image #{short_id}: 已过期]"

    # ── 工具：describe_image（llm_select 模式）──

    @register.tool(
        "describe_image",
        "获取图片内容描述。当消息中包含 [Image #xxxx: ] 格式标识符且你需要了解图片内容时调用，传入标识符中的 xxxx。",
        {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "图片标识符（[Image #xxxx: ] 中的 xxxx）"},
            },
            "required": ["image_id"],
        },
    )
    async def describe_image(self, event: KiraMessageBatchEvent, image_id: str) -> str:
        """LLM 按需加载图片描述：当前回合原图 → 缓存/VLM；历史回合 → id_map 反查缓存。"""
        session_key = getattr(event, "sid", None) or "tool"
        # 1. 当前回合：_pir_images 找原图 → 缓存/VLM
        for msg in getattr(event, "messages", []) or []:
            images_map = getattr(msg, "_pir_images", None)
            if images_map and image_id in images_map:
                ele = images_map[image_id]
                desc = await self._describe_one(0, ele, session_key, 1)
                return desc or "图片描述不可用"
        # 2. 历史回合：id_map → 缓存
        full_md5 = self._id_map.get(image_id)
        if full_md5:
            try:
                entry = await self.ctx.db.get_image_desc_cache(full_md5)
                if entry and entry.get("description") \
                        and _is_valid_desc(entry["description"]):
                    return entry["description"]
            except Exception as e:
                logger.debug(f"[ParallelImageReader] describe_image cache query failed: {e}")
        return "图片已过期或不可追溯"

    # ── VLM call with JPEG compression ──

    async def _vlm_call(self, pil_image: PILImage.Image, prompt: str, quality: int) -> str:
        """Encode PIL Image as JPEG at *quality*, send to VLM, return text.

        静默失败——异常由调用方 _one 统一处理。
        """
        w, h = pil_image.size
        prompt_preview = prompt[:80].replace("\n", " ")
        vlm_logger.debug(
            f"[VLM] request | image={w}x{h} | quality={quality} | prompt={prompt_preview}..."
        )
        vlm = self.ctx.provider_mgr.get_default_vlm()
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        request = LLMRequest(messages=messages)
        resp = await vlm.chat(request)
        result = (resp.text_response or "").strip()
        result_preview = result[:100].replace("\n", " ")
        vlm_logger.debug(
            f"[VLM] response | len={len(result)} | {result_preview}..."
        )
        return result

    # ── Parallel VLM ──

    async def _describe_one(self, idx: int, elem, session_key: str, total: int) -> str:
        """描述单张图片：缓存检查 → VLM → 写缓存。失败返回 ""（由调用方降级）。"""
        sem = self._sem
        db = self.ctx.db
        try:
            md5 = await elem.hash_image()

            # 双重检查缓存（on_im_message 已查过，但可能并发刷新）
            # 防御：描述含控制字符或占位符标记的视为无效（历史污染数据）
            cached = await db.get_image_desc_cache(md5)
            if cached and cached.get("description") and _is_valid_desc(cached["description"]):
                desc = cached["description"]
                vlm_logger.debug(
                    f"[VLM] #{idx + 1}/{total} cache HIT [{session_key}] | "
                    f"md5={md5[:8]}... | {desc[:80].replace(chr(10), ' ')}..."
                )
                return desc

            prompt = DESC_PROMPT

            async with sem:
                if self.quality_enabled:
                    data_url = await elem.to_data_url()
                    raw_b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
                    buf = io.BytesIO(base64.b64decode(raw_b64))
                    pil_image = PILImage.open(buf).convert("RGB")
                    vlm_logger.info(
                        f"[VLM] #{idx + 1}/{total} describing [{session_key}] "
                        f"(quality={self.quality_value}, {pil_image.size[0]}x{pil_image.size[1]})"
                    )
                    desc = await asyncio.wait_for(
                        self._vlm_call(pil_image, prompt, self.quality_value),
                        timeout=VLM_TIMEOUT,
                    )
                else:
                    vlm = self.ctx.provider_mgr.get_default_vlm()
                    vlm_logger.info(
                        f"[VLM] #{idx + 1}/{total} describing [{session_key}] "
                        f"(native, md5={md5[:8]}...)"
                    )
                    vlm_logger.debug(
                        f"[VLM] #{idx + 1}/{total} desc_img [{session_key}] | "
                        f"prompt={prompt[:60].replace(chr(10), ' ')}..."
                    )
                    desc = await asyncio.wait_for(
                        desc_img(client=vlm, image=elem, prompt=prompt),
                        timeout=VLM_TIMEOUT,
                    )
                    desc_preview = desc[:80].replace(chr(10), " ") if desc else "(empty)"
                    vlm_logger.debug(
                        f"[VLM] #{idx + 1}/{total} done [{session_key}] | "
                        f"len={len(desc or '')} | {desc_preview}..."
                    )

            if not (desc and _is_valid_desc(desc)):
                # VLM 返回空 / 污染（\x00、旧占位符）：统一降级，且不写缓存
                return ""
            try:
                await db.add_image_desc_cache(md5, desc, count=1, last_seen=0)
            except Exception as e:
                logger.debug(
                    f"[ParallelImageReader] failed to cache desc "
                    f"[{session_key}] md5={md5[:8]}: {e}"
                )
            return desc
        except asyncio.TimeoutError:
            logger.warning(
                f"[ParallelImageReader] describe #{idx + 1} timed out "
                f"[{session_key}] ({VLM_TIMEOUT}s)"
            )
            return ""
        except Exception as e:
            logger.warning(
                f"[ParallelImageReader] describe #{idx + 1} failed "
                f"[{session_key}]: {type(e).__name__}: {e}"
            )
            return ""

    async def _describe_parallel(self, images: list, session_key: str) -> list[str]:
        """Concurrent VLM calls with Semaphore + image_desc_cache."""
        results = await asyncio.gather(
            *[self._describe_one(i, e, session_key, len(images))
              for i, e in enumerate(images)],
            return_exceptions=True,
        )
        return [r if isinstance(r, str) else "" for r in results]
