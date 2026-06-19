"""
并行识别插件 v2.0.1 — Parallel Image Reader

在 on_im_message 阶段拦截 Image/Sticker，并行调 VLM 生成描述后直接
替换为 Text 元素，写入历史的就是纯文字，无需 on_llm_request 二次替换。

配置项见 schema.json（无 schema 变更，仅 manifest 版本号 → 2.0.0）
"""

import asyncio
import base64
import io

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


class ParallelImageReader(BasePlugin):
    """并行识别插件 v2.0.1"""

    def __init__(self, ctx: PluginContext, cfg: dict):
        super().__init__(ctx, cfg)

        # Config — will be overridden by initialize()
        self.max_concurrent: int = 3
        self.quality_enabled: bool = False
        self.quality_value: int = 85

    # ── Lifecycle ──

    async def initialize(self):
        """加载用户配置"""
        self.max_concurrent = self.plugin_cfg.get("max_concurrent", 3)
        self.quality_enabled = self.plugin_cfg.get("quality_enabled", False)
        self.quality_value = self.plugin_cfg.get("quality_value", 85)

        logger.info(
            f"[ParallelImageReader] v2.0.1 initialized: "
            f"max_concurrent={self.max_concurrent}, "
            f"quality={'on(' + str(self.quality_value) + ')' if self.quality_enabled else 'off'}"
        )

    async def terminate(self):
        logger.info("[ParallelImageReader] terminated")

    # ── Chain processing ──

    async def _process_chain(self, chain, session_key: str) -> int:
        """Traverse chain tree, describe Image/Sticker via VLM in parallel,
        replace each with Text element. Returns count processed."""
        images: list = []
        positions: list[tuple] = []

        def _collect(chain_ref, _visited=None):
            if _visited is None:
                _visited = set()
            cid = id(chain_ref)
            if cid in _visited:
                logger.warning(
                    f"[ParallelImageReader] cycle detected in message chain "
                    f"[{session_key}], skipping"
                )
                return
            _visited.add(cid)
            for i, ele in enumerate(chain_ref):
                if isinstance(ele, (Image, Sticker)):
                    images.append(ele)
                    positions.append((chain_ref, i))
                elif isinstance(ele, Reply) and ele.chain is not None:
                    _collect(ele.chain, _visited)
                elif isinstance(ele, Forward) and ele.chains:
                    for c in ele.chains:
                        _collect(c, _visited)

        _collect(chain)

        if not images:
            return 0

        descriptions = await self._describe_images(images, session_key)

        for (chain_ref, idx), desc in zip(positions, descriptions):
            chain_ref[idx] = Text(f"[Image: {desc or '(description unavailable)'}]")

        return len(images)

    # ── VLM call with JPEG compression ──

    async def _vlm_call(self, pil_image: PILImage.Image, prompt: str, quality: int) -> str:
        """Encode PIL Image as JPEG at *quality*, send to VLM, return text."""
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
            resp = await asyncio.wait_for(vlm.chat(request), timeout=VLM_TIMEOUT)
            result = (resp.text_response or "").strip()
            result_preview = result[:100].replace("\n", " ")
            vlm_logger.info(
                f"[VLM] response | len={len(result)} | {result_preview}..."
            )
            return result
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.warning(f"[ParallelImageReader] VLM timed out ({VLM_TIMEOUT}s)")
            else:
                logger.warning(f"[ParallelImageReader] VLM call failed: {type(e).__name__}: {e}")
            return ""

    # ── Parallel VLM ──

    async def _describe_parallel(self, images: list, session_key: str) -> list[str]:
        """Concurrent VLM calls with Semaphore + image_desc_cache."""
        sem = asyncio.Semaphore(self.max_concurrent)
        db = self.ctx.db

        async def _one(idx: int, elem) -> str:
            try:
                md5 = await elem.hash_image()

                cached = await db.get_image_desc_cache(md5)
                if cached and cached.get("description"):
                    desc = cached["description"]
                    vlm_logger.info(
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
                        desc = await asyncio.wait_for(
                            self._vlm_call(pil_image, prompt, self.quality_value),
                            timeout=VLM_TIMEOUT,
                        )
                    else:
                        vlm = self.ctx.provider_mgr.get_default_vlm()
                        vlm_logger.info(
                            f"[VLM] #{idx + 1}/{len(images)} desc_img [{session_key}] | "
                            f"md5={md5[:8]}... | prompt={prompt[:60].replace(chr(10), ' ')}..."
                        )
                        desc = await asyncio.wait_for(
                            desc_img(client=vlm, image=elem, prompt=prompt),
                            timeout=VLM_TIMEOUT,
                        )
                        desc_preview = desc[:80].replace(chr(10), " ") if desc else "(empty)"
                        vlm_logger.info(
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

    # ── VLM dispatch ──

    async def _describe_images(self, images: list, session_key: str) -> list[str]:
        """Always parallel — dispatch point kept for future expansion."""
        mode = "quality" if self.quality_enabled else "native"
        logger.info(
            f"[ParallelImageReader] parallel mode ({mode}) [{session_key}]: "
            f"{len(images)} images, concurrency={self.max_concurrent}"
        )
        return await self._describe_parallel(images, session_key)

    # ── Event: IM message ──

    @on.im_message(priority=Priority.SYS_HIGH - 1)
    async def on_im_message(self, event: KiraMessageEvent):
        """Intercept IM messages: describe images via VLM, replace with Text."""
        session_key = event.session.sid if event.session else "default"
        try:
            n = await self._process_chain(event.message.chain, session_key)
            if n > 0:
                logger.info(
                    f"[ParallelImageReader] described {n} images [{session_key}]"
                )
        except Exception as e:
            logger.error(
                f"[ParallelImageReader] on_im_message failed [{session_key}]: "
                f"{type(e).__name__}: {e}"
            )

    # ── Event: LLM request ──

    @on.llm_request(priority=Priority.SYS_HIGH - 1)
    async def on_llm_request(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """Inject system hint explaining the [Image: ...] format."""
        session_key = event.session.sid if event.session else "default"
        try:
            self._inject_system_hint(req)
        except Exception as e:
            logger.error(
                f"[ParallelImageReader] on_llm_request failed [{session_key}]: "
                f"{type(e).__name__}: {e}"
            )

    @staticmethod
    def _inject_system_hint(req: LLMRequest):
        """Add a short note to system prompt explaining the [Image: ...] format."""
        hint = (
            "当消息中包含 [Image: 描述内容] 格式的标记时，"
            "这表示用户发送了一张图片，其内容由「描述内容」说明。"
        )
        for p in req.system_prompt:
            if getattr(p, "name", None) == "chat_env":
                if hint not in p.content:
                    p.content += "\n" + hint
                break
        else:
            vlm_logger.debug("no chat_env prompt found to inject image description hint")
