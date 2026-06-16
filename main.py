"""
并行识别插件 - Parallel Image Reader

拦截 IM 消息中的图片替换为占位符（零阻塞），在 ON_LLM_REQUEST 时并发调用 VLM
描述图片后注入回用户提示词。

配置项见 schema.json
"""

import asyncio
import base64
import io
import time
from pathlib import Path

from PIL import Image as PILImage

from core.plugin import BasePlugin, PluginContext, register_tool, on, Priority, logger
from core.logging_manager import get_logger
from core.chat.message_elements import Image, Text, Sticker, Reply, Forward
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.utils.common_utils import desc_img
from core.provider import LLMRequest

vlm_logger = get_logger("parallel_vlm", "purple")

# ── Prompts (not configurable) ──

# Single image — KiraAI built-in, kept as-is
SINGLE_DESC_PROMPT = "描述这张图片的内容，如果有文字请将其输出"

# Multi-image parallel — each VLM only sees one image.
# Giving it context that other images exist avoids overly-brief descriptions.
MULTI_DESC_PROMPT = (
    "这是用户发送的第 {index} 张图片。"
    "请详细描述这张图片的内容，画面主体、人物特征、场景、动作、"
    "图中文字等，如果有文字请将其输出。"
)

# Stash 清理：未被 llm_request 消费的会话最长保留时间（秒）。
# 防止消息进 buffer 但未触发 llm_request（撤回/超时/丢弃）导致的内存泄漏。
_STASH_TTL_SEC = 600


class ParallelImageReader(BasePlugin):
    """并行识别插件"""

    def __init__(self, ctx: PluginContext, cfg: dict):
        super().__init__(ctx, cfg)

        # Ensure per-plugin data directory exists
        self.plugin_data_dir = Path(ctx.get_plugin_data_dir())
        self.plugin_data_dir.mkdir(parents=True, exist_ok=True)

        # Config — will be overridden by initialize()
        self.max_concurrent: int = 3
        self.quality_enabled: bool = False
        self.quality_value: int = 85

        # Stash: session_id -> {placeholder_tag: Image|Sticker}
        self._stash: dict[str, dict[str, object]] = {}
        # session_id -> 最后写入时间戳（用于 TTL 清理，防内存泄漏）
        self._stash_ts: dict[str, float] = {}

    def _evict_stale_stash(self):
        """清理超过 TTL 未被消费的 stash 会话，防内存泄漏。"""
        now = time.monotonic()
        stale = [
            sk for sk, ts in self._stash_ts.items()
            if now - ts > _STASH_TTL_SEC
        ]
        for sk in stale:
            self._stash.pop(sk, None)
            self._stash_ts.pop(sk, None)
        if stale:
            logger.info(f"[ParallelImageReader] evicted {len(stale)} stale stash session(s)")

    # ── Lifecycle ──

    async def initialize(self):
        """加载用户配置"""
        self.max_concurrent = self.plugin_cfg.get("max_concurrent", 3)
        self.quality_enabled = self.plugin_cfg.get("quality_enabled", False)
        self.quality_value = self.plugin_cfg.get("quality_value", 85)

        logger.info(
            f"[ParallelImageReader] initialized: "
            f"max_concurrent={self.max_concurrent}, "
            f"quality={'on(' + str(self.quality_value) + ')' if self.quality_enabled else 'off'}"
        )

    async def terminate(self):
        self._stash.clear()
        self._stash_ts.clear()
        logger.info("[ParallelImageReader] terminated")

    # ── Chain extraction helpers ──

    @staticmethod
    def _extract_and_replace(
        chain, stash: dict, session_key: str, idx_counter: list
    ):
        """Recursively traverse a message chain, replace Image/Sticker with
        Text placeholders and store the originals in *stash*."""
        for i, ele in enumerate(chain):
            if isinstance(ele, (Image, Sticker)):
                tag = f"__IMG__{session_key}__{idx_counter[0]}__"
                stash[tag] = ele
                chain[i] = Text(tag)
                idx_counter[0] += 1
            elif isinstance(ele, Reply) and ele.chain is not None:
                ParallelImageReader._extract_and_replace(
                    ele.chain, stash, session_key, idx_counter
                )
            elif isinstance(ele, Forward) and ele.chains:
                for c in ele.chains:
                    ParallelImageReader._extract_and_replace(
                        c, stash, session_key, idx_counter
                    )

    # ── VLM call with JPEG compression ──

    async def _vlm_call(self, pil_image: PILImage.Image, prompt: str, quality: int) -> str:
        """Encode PIL Image as JPEG at *quality*, send to VLM, return text.

        Always logs request (image size, prompt) and response (length, preview).
        Never raises — returns ``""`` on any error.
        """
        w, h = pil_image.size
        prompt_preview = prompt[:80].replace("\n", " ")
        vlm_logger.info(
            f"[VLM] request | image={w}x{h} | quality={quality} | prompt={prompt_preview}..."
        )
        try:
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
            vlm_logger.info(
                f"[VLM] response | len={len(result)} | {result_preview}..."
            )
            return result
        except Exception as e:
            logger.warning(f"[ParallelImageReader] VLM call failed: {type(e).__name__}: {e}")
            vlm_logger.info(
                f"[VLM] FAILED | {type(e).__name__}: {e}"
            )
            return ""

    # ── Parallel VLM (core capability) ──

    async def _describe_parallel(self, images: list) -> list[str]:
        """Concurrent VLM calls with Semaphore + image_desc_cache.

        *quality_enabled=True* → convert to PIL → re-encode as JPEG at
        ``quality_value`` before sending (smaller payload, faster upload,
        reduces provider-side processing).

        *quality_enabled=False* → use KiraAI ``desc_img`` (native path).
        """
        is_multi = len(images) > 1
        sem = asyncio.Semaphore(self.max_concurrent)
        vlm = self.ctx.provider_mgr.get_default_vlm()
        db = self.ctx.db

        async def _one(idx: int, elem) -> str:
            try:
                md5 = await elem.hash_image()

                # Cache check
                cached = await db.get_image_desc_cache(md5)
                if cached and cached.get("description"):
                    desc = cached["description"]
                    vlm_logger.info(
                        f"[VLM] #{idx + 1}/{len(images)} cache HIT | "
                        f"md5={md5[:8]}... | {desc[:80].replace(chr(10), ' ')}..."
                    )
                    return desc

                # Pick prompt
                prompt = (
                    MULTI_DESC_PROMPT.format(index=idx + 1)
                    if is_multi else SINGLE_DESC_PROMPT
                )

                async with sem:
                    if self.quality_enabled:
                        # Quality path: convert to PIL → JPEG(quality) → VLM
                        data_url = await elem.to_data_url()
                        raw_b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
                        buf = io.BytesIO(base64.b64decode(raw_b64))
                        pil_image = PILImage.open(buf).convert("RGB")
                        desc = await self._vlm_call(pil_image, prompt, self.quality_value)
                    else:
                        # Native path: desc_img (detail="high", no re-encode)
                        vlm_logger.info(
                            f"[VLM] #{idx + 1}/{len(images)} desc_img | "
                            f"md5={md5[:8]}... | prompt={prompt[:60].replace(chr(10), ' ')}..."
                        )
                        desc = await desc_img(client=vlm, image=elem, prompt=prompt)
                        desc_preview = desc[:80].replace(chr(10), " ") if desc else "(empty)"
                        vlm_logger.info(
                            f"[VLM] #{idx + 1}/{len(images)} done | "
                            f"len={len(desc)} | {desc_preview}..."
                        )

                # Write back to cache
                if desc:
                    try:
                        await db.add_image_desc_cache(md5, desc, count=1, last_seen=0)
                    except Exception:
                        pass
                return desc
            except Exception as e:
                logger.warning(f"[ParallelImageReader] describe #{idx + 1} failed: {type(e).__name__}: {e}")
                return ""

        results = await asyncio.gather(
            *[_one(i, e) for i, e in enumerate(images)],
            return_exceptions=True,
        )
        return [r if isinstance(r, str) else "" for r in results]

    # ── VLM dispatch ──

    async def _describe_images(self, images: list) -> list[str]:
        """Always parallel — dispatch point kept for future expansion."""
        mode = "quality" if self.quality_enabled else "native"
        logger.info(
            f"[ParallelImageReader] parallel mode ({mode}): {len(images)} images, "
            f"concurrency={self.max_concurrent}"
        )
        return await self._describe_parallel(images)

    # ── Event: IM message ──

    @on.im_message(priority=Priority.SYS_HIGH - 1)
    async def on_im_message(self, event: KiraMessageEvent):
        """Intercept IM messages: extract images, replace with placeholders."""
        session_key = event.session.sid if event.session else "default"
        self._evict_stale_stash()
        stash = self._stash.setdefault(session_key, {})
        idx = [0]

        self._extract_and_replace(event.message.chain, stash, session_key, idx)

        if idx[0] > 0:
            self._stash_ts[session_key] = time.monotonic()
            logger.info(
                f"[ParallelImageReader] intercepted {idx[0]} images [{session_key}]"
            )
        elif not stash:
            # 没截到图且 stash 空 → 移除占位会话，避免空 dict 堆积
            self._stash.pop(session_key, None)
        # Note: we deliberately do NOT call event.buffer() here.
        # The default chat plugin (priority=HIGH) handles the buffering strategy.
        # Running at SYS_HIGH-1 ensures we run first and modify the chain before
        # the default plugin sees it.

    # ── Event: LLM request ──

    @on.llm_request(priority=Priority.SYS_HIGH - 1)
    async def on_llm_request(
        self, event: KiraMessageBatchEvent, req: LLMRequest, *_
    ):
        """Before LLM invocation: inject image descriptions into the prompt."""
        session_key = event.session.sid if event.session else "default"
        stash = self._stash.pop(session_key, None)
        self._stash_ts.pop(session_key, None)
        if not stash:
            return

        images = list(stash.values())
        if not images:
            return

        # Run VLM
        descriptions = await self._describe_images(images)

        # Build placeholder → description mapping
        placeholders = list(stash.keys())
        tag_to_desc = {}
        for i, tag in enumerate(placeholders):
            desc = descriptions[i] if i < len(descriptions) else ""
            if not desc:
                desc = "[Image description unavailable]"
            tag_to_desc[tag] = desc

        # Replace in user_prompt
        for p in req.user_prompt:
            if not getattr(p, "persist", True):
                continue
            for tag, desc in tag_to_desc.items():
                if tag in p.content:
                    p.content = p.content.replace(tag, f"[Image: {desc}]")

        # Fallback: also replace in already-assembled messages
        for msg in req.messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                for tag, desc in tag_to_desc.items():
                    if tag in msg.content:
                        msg.content = msg.content.replace(tag, f"[Image: {desc}]")

        # Inject system hint about [Image: ...] format
        self._inject_system_hint(req)

        # Log final injection summary
        for tag, desc in tag_to_desc.items():
            vlm_logger.info(
                f"[Inject] {tag} -> [Image: {desc[:60].replace(chr(10), ' ')}...]"
            )

        logger.info(
            f"[ParallelImageReader] injected {len(images)} descriptions [{session_key}]"
        )

    @staticmethod
    def _inject_system_hint(req: LLMRequest):
        """Add a short note to system prompt explaining the [Image: ...] format."""
        hint = (
            "当消息中包含 [Image: 描述内容] 格式的标记时，"
            '这表示用户发送了一张图片，其内容由「描述内容」说明。'
        )
        for p in req.system_prompt:
            if getattr(p, "name", None) == "chat_env":
                if hint not in p.content:
                    p.content += "\n" + hint
                break
