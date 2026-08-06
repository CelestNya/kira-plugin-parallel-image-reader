"""KiraAI 嵌套消息渲染样例 — 直接调真实 message_format_to_text 展示输出。"""
import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR / "tests"))

from integration_harness import Harness, _PNG_B64  # noqa: E402


def _img():
    from core.chat.message_elements import Image
    return Image(image=f"base64://{_PNG_B64}")


def _fmt(text: str) -> str:
    return text.replace("\n", "⏎")


async def main():
    from core.chat.message_elements import Text, Forward, Reply
    from core.chat.message_utils import MessageChain

    h = Harness(load_mode="lazy")
    await h.start()
    try:
        # ── 样例 A：基本 Forward（一条转发，含一条消息带图）──
        node1 = MessageChain([
            Text("第一层消息：看看这张图"),
            _img(),
        ])
        chain_a = MessageChain([Forward(chains=[node1])])
        print("=== A. 基本 Forward ===")
        print("输入 chain:", [type(e).__name__ for e in chain_a],
              "→ chains[0]:", [type(e).__name__ for e in node1])
        print("输出:", _fmt(await h.mp.message_format_to_text(chain_a)))
        print()

        # ── 样例 B：嵌套 Forward（node 内容里再含一张转发卡片）──
        # 模拟 napcat 递归拉取后的结构：外层转发 → 中层文本 + 内层转发卡片
        deep = MessageChain([Text("深层转发内容"), _img()])
        mid = MessageChain([Text("中层消息"), Forward(chains=[deep])])
        chain_b = MessageChain([Forward(chains=[mid])])
        print("=== B. 嵌套 Forward（核心过滤行为）===")
        print("输入结构: Forward → [Text, Forward → [Text, Image]]")
        print("输出:", _fmt(await h.mp.message_format_to_text(chain_b)))
        print("→ 注意：内层 Forward 被核心过滤，'深层转发内容'和图片都没了")
        print()

        # ── 样例 C：Reply（引用消息）──
        reply_chain = MessageChain([Text("被引用的原消息"), _img()])
        chain_c = MessageChain([Reply(message_id="12345", chain=reply_chain)])
        print("=== C. Reply（引用消息）===")
        print("输出:", _fmt(await h.mp.message_format_to_text(chain_c)))
        print()

        # ── 样例 D：嵌套 Forward + 插件阶段1 拍平后（修复后 LLM 实际看到的）──
        # 注意：必须用全新 chain——核心渲染 Forward 时会【就地改写】
        # chains[i].message_list（过滤嵌套 Forward），B 已破坏了 chain_b
        from core.chat.message_utils import KiraMessageEvent
        deep2 = MessageChain([Text("深层转发内容"), _img()])
        mid2 = MessageChain([Text("中层消息"), Forward(chains=[deep2])])
        chain_d = MessageChain([Forward(chains=[mid2])])
        ev = h.make_image_event(chain=chain_d, message_id="sample-d")
        await h.run_im(ev)  # 插件阶段1：拍平 + Image→Text 标识符
        print("=== D. 嵌套 Forward + 插件拍平后（v2.4.1 实际行为）===")
        print("阶段1 后顶层:", [type(e).__name__ for e in ev.message.chain],
              "| mid:", [type(e).__name__ for e in mid2])
        print("输出:", _fmt(await h.mp.message_format_to_text(ev.message.chain)))
        print("→ 内层内容保住了，图片变成 [Image #<md5前8位>: 描述]")
        print()

        # ── 样例 E：成环（恶意）渲染 —— 核心的防环行为 ──
        c1 = MessageChain([Text("环1"), _img()])
        c2 = MessageChain([Text("环2")])
        c1.append(Forward(chains=[c2]))
        c2.append(Forward(chains=[c1]))
        chain_e = MessageChain([Forward(chains=[c1])])
        print("=== E. Forward 成环（核心渲染）===")
        print("输出:", _fmt(await h.mp.message_format_to_text(chain_e))[:120])
        print("→ 不崩溃（外层过滤），内容被截断——恶意输入安全降级")
    finally:
        await h.stop()


if __name__ == "__main__":
    asyncio.run(main())
