"""
并行识图插件 v2.2.0 — Parallel Image Reader

两阶段架构，既不污染聊天历史，也不浪费 VLM 调用：

1. ON_IM_MESSAGE（优先级 SYS_HIGH-1，早于 chat 插件 HIGH）
   递归遍历消息链，把 Image/Sticker 替换为 Text 占位符（阻止本体
   message_format_to_text 串行调 VLM），图片元素暂存到 message 的
   _pir_pending 属性。不调 VLM、不干预 event 策略。被 discard 的消息
   零 VLM 开销。

2. ON_IM_BATCH_MESSAGE（优先级 SYS_HIGH-1）
   此时消息已确定会发给 LLM，本体 message_format_to_text 已执行（占位
   Text 不触发 VLM）。插件并行描述所有暂存图片，替换 message_str 中的
   占位符为 [Image: 描述]，同时替换 chain 元素。message_str 在持久化
   之前被修正，占位符不进入历史。

乐观加载（eager_loading，默认关闭）：ON_IM_MESSAGE 阶段通过
asyncio.create_task 启动后台 VLM task，不阻塞 handler 链；
ON_IM_BATCH_MESSAGE 阶段 await 复用结果。被 discard 的消息仍然会
完成 VLM 调用（结果写入缓存可被复用），启用后 VLM 调用量会增加。

配置项见 schema.json。
"""

import asyncio
import base64
import io
import re

from PIL import Image as PILImage

from core.plugin import BasePlugin, PluginContext, on, Priority, logger
from core.logging_manager import get_logger
from core.chat.message_elements import Image, Text, Sticker, Reply, Forward
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.utils.common_utils import desc_img
from core.provider import LLMRequest

vlm_logger = get_logger("parallel_vlm", "purple")

# ── Prompts (not configurable) ──

DESC_PROMPT = "描述这张图片的内容，如果有文字请将其输出"

# 单次 VLM 调用超时（秒），超时返回空描述 → "(description unavailable)"
VLM_TIMEOUT = 60

# 占位符：替换 chain 中的 Image/Sticker，阻止本体 message_format_to_text 调 VLM。
# 格式 <!--PIR:md5--> —— XML 注释样式，即使泄漏到 LLM 输入也会被当作注释忽略，
# 人类可读，且不与正常文本冲突。在 ON_IM_BATCH_MESSAGE 阶段（持久化之前）被
# 替换为真实描述，不进入历史。
_PLACEHOLDER_RE = re.compile(r"<!--PIR:[^>]*-->")


def _make_placeholder(md5: str) -> str:
    """生成占位符文本。"""
    return f"<!--PIR:{md5}-->"


def _is_valid_desc(desc: str) -> bool:
    """检查缓存描述是否有效——排除控制字符和占位符标记的污染数据。"""
    if not desc:
        return False
    if "\x00" in desc:
        # 旧 caption 方案遗留的 \x00IMG_PENDING_ 污染
        return False
    if "<!--PIR:" in desc:
        # 新占位符标记的自我污染（理论上不会发生，防御性检查）
        return False
    return True


class ParallelImageReader(BasePlugin):
    """并行识图插件 v2.2.0 — 两阶段架构 + 乐观加载"""

    def __init__(self, ctx: PluginContext, cfg: dict):
        super().__init__(ctx, cfg)

        # Config — will be overridden by initialize()
        self.max_concurrent: int = 3
        self.quality_enabled: bool = False
        self.quality_value: int = 85

        # 乐观加载 in-flight task 强引用池（防 GC 回收）
        self._optimistic_tasks: set = set()
        self._sem = None  # initialize 中创建

    # ── Lifecycle ──

    async def initialize(self):
        """加载用户配置"""
        self.max_concurrent = self.plugin_cfg.get("max_concurrent", 3)
        self.quality_enabled = self.plugin_cfg.get("quality_enabled", False)
        self.quality_value = self.plugin_cfg.get("quality_value", 85)
        self.eager_loading = self.plugin_cfg.get("eager_loading", False)
        self._sem = asyncio.Semaphore(self.max_concurrent)

        eager_str = f", eager={'on' if self.eager_loading else 'off'}"
        logger.info(
            f"[ParallelImageReader] v2.2.0 initialized: "
            f"max_concurrent={self.max_concurrent}, "
            f"quality={'on(' + str(self.quality_value) + ')' if self.quality_enabled else 'off'}"
            f"{eager_str}"
        )

    async def terminate(self):
        """取消所有未完成乐观加载 task，防泄漏。"""
        for task in list(self._optimistic_tasks):
            task.cancel()
        self._optimistic_tasks.clear()
        logger.info("[ParallelImageReader] terminated")

    # ── Chain traversal ──

    @staticmethod
    def _walk(chain, visited=None):
        """递归遍历消息链（含 Reply.chain / Forward.chains），yield (chain_ref, index)。

        带环检测：用 id(chain_ref) 标记已访问，避免 Reply 互引导致 RecursionError。
        """
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
                yield from ParallelImageReader._walk(ele.chain, visited)
            elif isinstance(ele, Forward) and ele.chains:
                for c in ele.chains:
                    yield from ParallelImageReader._walk(c, visited)

    # ── Stage 1: ON_IM_MESSAGE — replace with placeholder, stash originals ──

    @on.im_message(priority=Priority.SYS_HIGH - 1)
    async def on_im_message(self, event: KiraMessageEvent):
        """把 Image/Sticker 替换为 Text 占位符，原图暂存到 message._pir_pending。

        不调 VLM、不干预 event 策略。被 discard 的消息零 VLM 开销。
        """
        session_key = event.session.sid if event.session else "default"
        try:
            db = self.ctx.db
            # 暂存：(placeholder, image_element, md5) 列表，挂在 message 上
            pending: list = []
            cached_count = 0

            for chain_ref, idx in self._walk(event.message.chain):
                ele = chain_ref[idx]
                try:
                    md5 = await ele.hash_image()
                except Exception as e:
                    logger.debug(
                        f"[ParallelImageReader] hash failed [{session_key}]: "
                        f"{type(e).__name__}: {e}"
                    )
                    md5 = None

                # 查缓存：命中则直接用描述做占位符（后续无需 VLM）
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

                if desc_cached is not None:
                    # 缓存命中：直接替换为 [Image: 描述]，无需进 batch 阶段
                    chain_ref[idx] = Text(f"[Image: {desc_cached}]")
                else:
                    # 缓存未命中：替换为占位符，暂存原图供 batch 阶段 VLM
                    placeholder = _make_placeholder(md5 or f"noid_{id(ele)}")
                    chain_ref[idx] = Text(placeholder)
                    pending.append((placeholder, ele, md5))

            if pending:
                # 挂到 message 上，batch 阶段读取（message 实例在两阶段间是同一个对象）
                event.message._pir_pending = pending
                if self.eager_loading:
                    # 乐观加载：立即启动 VLM task，不应 await
                    images = [p[1] for p in pending]
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
                else:
                    logger.info(
                        f"[ParallelImageReader] stashed {len(pending)} pending, "
                        f"{cached_count} cache-hit [{session_key}]"
                    )
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
            # 1. 收集所有 pending 图片，分流乐观/非乐观
            optimistic_groups: list = []      # [(task, pending_list), ...]
            non_optimistic_pending: list = []  # [(message, placeholder, ele, md5), ...]
            for message in event.messages:
                pending = getattr(message, "_pir_pending", None)
                if not pending:
                    continue
                task = getattr(message, "_pir_optimistic", None)
                if task is not None:
                    optimistic_groups.append((task, pending))
                else:
                    for placeholder, ele, md5 in pending:
                        non_optimistic_pending.append((message, placeholder, ele, md5))

            if not optimistic_groups and not non_optimistic_pending:
                # 全部缓存命中或无图片——仍需清理可能残留的占位符
                self._cleanup_placeholders(event)
                return

            placeholder_map: dict[str, str] = {}

            # 2. 并行：乐观 task await + 非乐观 on-demand VLM
            async def _do_optimistic():
                if not optimistic_groups:
                    return
                results = await asyncio.gather(
                    *[t for t, _ in optimistic_groups],
                    return_exceptions=True,
                )
                for (_, pending), res in zip(optimistic_groups, results):
                    descs = res if isinstance(res, list) else [""] * len(pending)
                    for (placeholder, _, _), desc in zip(pending, descs):
                        placeholder_map[placeholder] = \
                            f"[Image: {desc or '(description unavailable)'}]"

            async def _do_non_optimistic():
                if not non_optimistic_pending:
                    return
                imgs = [item[2] for item in non_optimistic_pending]
                descs = await self._describe_parallel(imgs, session_key)
                for (_, placeholder, _, _), desc in zip(non_optimistic_pending, descs):
                    placeholder_map[placeholder] = \
                        f"[Image: {desc or '(description unavailable)'}]"

            await asyncio.gather(_do_optimistic(), _do_non_optimistic())

            # 3. 替换每个 message 的 message_str 和 chain
            for message in event.messages:
                if message.message_str:
                    for ph, final in placeholder_map.items():
                        message.message_str = message.message_str.replace(ph, final)
                self._replace_placeholders_in_chain(message.chain, placeholder_map)

            total = sum(len(p) for _, p in optimistic_groups) + len(non_optimistic_pending)
            mode = "quality" if self.quality_enabled else "native"
            eager_info = f", eager={sum(len(p) for _, p in optimistic_groups)}" \
                if optimistic_groups else ""
            logger.info(
                f"[ParallelImageReader] described {total} images"
                f" ({mode}, concurrency={self.max_concurrent}{eager_info}) "
                f"[{session_key}]"
            )
        except asyncio.CancelledError:
            # Ctrl+C / 任务取消：清理占位符后重新抛出，不拦截取消
            self._cleanup_placeholders(event)
            raise
        except Exception as e:
            logger.error(
                f"[ParallelImageReader] on_im_batch_message failed [{session_key}]: "
                f"{type(e).__name__}: {e}"
            )
            # 兜底：确保占位符不泄漏到历史
            self._cleanup_placeholders(event)

    # ── 占位符清理（兜底）──

    @staticmethod
    def _replace_placeholders_in_chain(chain, placeholder_map: dict, visited=None):
        """递归遍历 chain（含 Reply/Forward），把占位 Text 替换为真实描述 Text。"""
        if visited is None:
            visited = set()
        cid = id(chain)
        if cid in visited:
            return
        visited.add(cid)
        for i, ele in enumerate(chain):
            if isinstance(ele, Text) and ele.text in placeholder_map:
                chain[i] = Text(placeholder_map[ele.text])
            elif isinstance(ele, Reply) and ele.chain is not None:
                ParallelImageReader._replace_placeholders_in_chain(ele.chain, placeholder_map, visited)
            elif isinstance(ele, Forward) and ele.chains:
                for c in ele.chains:
                    ParallelImageReader._replace_placeholders_in_chain(c, placeholder_map, visited)

    @staticmethod
    def _cleanup_placeholders(event: KiraMessageBatchEvent):
        """把所有 message_str 和 chain 中残留的占位符替换为降级文本。

        复用 _replace_placeholders_in_chain 避免重复 chain 遍历逻辑。
        """
        try:
            for message in event.messages:
                # 修 message_str
                if hasattr(message, "message_str") and message.message_str:
                    message.message_str = _PLACEHOLDER_RE.sub(
                        "(description unavailable)", message.message_str
                    )
                # 修 chain：收集占位符 → 复用 _replace_placeholders_in_chain
                if hasattr(message, "chain"):
                    _pmap: dict[str, str] = {}

                    def _collect(chain, visited=None):
                        if visited is None:
                            visited = set()
                        cid = id(chain)
                        if cid in visited:
                            return
                        visited.add(cid)
                        for i, ele in enumerate(chain):
                            if isinstance(ele, Text) and _PLACEHOLDER_RE.match(ele.text):
                                _pmap[ele.text] = "[Image: (description unavailable)]"
                            elif isinstance(ele, Reply) and ele.chain is not None:
                                _collect(ele.chain, visited)
                            elif isinstance(ele, Forward) and ele.chains:
                                for c in ele.chains:
                                    _collect(c, visited)

                    _collect(message.chain)
                    if _pmap:
                        ParallelImageReader._replace_placeholders_in_chain(
                            message.chain, _pmap
                        )
        except Exception:
            pass

    # ── System hint injection ──

    @on.llm_request(priority=Priority.SYS_HIGH - 1)
    async def on_llm_request(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """注入 system hint 说明 [Image: ...] 格式。

        图片描述已在 ON_IM_BATCH_MESSAGE 阶段替换完毕并写入 message_str，
        此 handler 仅注入格式说明，不触碰图片内容。
        """
        hint = (
            "当消息中包含 [Image: 描述内容] 格式的标记时，"
            "这表示用户发送了一张图片，其内容由「描述内容」说明。"
        )
        for p in req.system_prompt:
            if getattr(p, "name", None) == "chat_env":
                if hint not in p.content:
                    p.content += "\n" + hint
                return
        vlm_logger.debug("no chat_env prompt found to inject image description hint")

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

    async def _describe_parallel(self, images: list, session_key: str) -> list[str]:
        """Concurrent VLM calls with Semaphore + image_desc_cache."""
        sem = self._sem
        db = self.ctx.db

        async def _one(idx: int, elem) -> str:
            try:
                md5 = await elem.hash_image()

                # 双重检查缓存（on_im_message 已查过，但可能并发刷新）
                # 防御：描述含控制字符或占位符标记的视为无效（历史污染数据）
                cached = await db.get_image_desc_cache(md5)
                if cached and cached.get("description") and _is_valid_desc(cached["description"]):
                    desc = cached["description"]
                    vlm_logger.debug(
                        f"[VLM] #{idx + 1}/{len(images)} cache HIT [{session_key}] | "
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
                            f"[VLM] #{idx + 1}/{len(images)} describing [{session_key}] "
                            f"(quality={self.quality_value}, {pil_image.size[0]}x{pil_image.size[1]})"
                        )
                        desc = await asyncio.wait_for(
                            self._vlm_call(pil_image, prompt, self.quality_value),
                            timeout=VLM_TIMEOUT,
                        )
                    else:
                        vlm = self.ctx.provider_mgr.get_default_vlm()
                        vlm_logger.info(
                            f"[VLM] #{idx + 1}/{len(images)} describing [{session_key}] "
                            f"(native, md5={md5[:8]}...)"
                        )
                        vlm_logger.debug(
                            f"[VLM] #{idx + 1}/{len(images)} desc_img [{session_key}] | "
                            f"prompt={prompt[:60].replace(chr(10), ' ')}..."
                        )
                        desc = await asyncio.wait_for(
                            desc_img(client=vlm, image=elem, prompt=prompt),
                            timeout=VLM_TIMEOUT,
                        )
                        desc_preview = desc[:80].replace(chr(10), " ") if desc else "(empty)"
                        vlm_logger.debug(
                            f"[VLM] #{idx + 1}/{len(images)} done [{session_key}] | "
                            f"len={len(desc)} | {desc_preview}..."
                        )

                if desc:
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

        results = await asyncio.gather(
            *[_one(i, e) for i, e in enumerate(images)],
            return_exceptions=True,
        )
        return [r if isinstance(r, str) else "" for r in results]
