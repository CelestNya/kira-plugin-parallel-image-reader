"""
渲染样式完整对照 — 从简到繁的嵌套场景，全部真实渲染输出。

每个场景给出：
  输入结构（树形）→ 原生渲染（无插件）→ 插件渲染（拍平+标识符）
覆盖：1 层 / 多节点 / 2 层 / 3 层嵌套 / Reply 内 Forward / Forward 内 Reply
      / 成环 / 深度隔断。完整文本，不截断。
"""
import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR / "tests"))

from integration_harness import Harness, _PNG_B64  # noqa: E402

NICK = "[2026-08-06 10:00:00] 张三: "


def _img(caption=None):
    from core.chat.message_elements import Image
    return Image(image=f"base64://{_PNG_B64}", caption=caption)


def _text(s):
    from core.chat.message_elements import Text
    return Text(s)


def _fwd(*chains):
    from core.chat.message_elements import Forward
    return Forward(chains=list(chains))


def _reply(chain, mid="r1"):
    from core.chat.message_elements import Reply
    return Reply(message_id=mid, chain=chain)


def _mc(*elems):
    from core.chat.message_utils import MessageChain
    return MessageChain(list(elems))


def tree(chain, indent=0, visited=None):
    """打印 chain 结构树（带环检测）。"""
    if visited is None:
        visited = set()
    cid = id(chain)
    if cid in visited:
        return ["  " * indent + "↺ (环)"]
    visited.add(cid)
    pad = "  " * indent
    parts = []
    for ele in chain:
        t = type(ele).__name__
        if t == "Forward":
            parts.append(f"{pad}{t}")
            for c in ele.chains:
                parts.extend(tree(c, indent + 1, visited))
        elif t == "Reply":
            parts.append(f"{pad}{t}(id={ele.message_id})")
            if ele.chain:
                parts.extend(tree(ele.chain, indent + 1, visited))
        elif t == "Image":
            parts.append(f"{pad}{t}(caption={'有' if ele.caption else '无'})")
        else:
            parts.append(f"{pad}{t}({getattr(ele, 'text', '')[:24]})")
    return parts


async def render_pair(h, name, chain):
    """同一 chain：原生渲染 vs 插件阶段1 后渲染。"""
    from core.chat.message_utils import MessageChain
    # 原生：直接渲染（需要副本——核心渲染 Forward 会就地改结构！）
    import copy
    native_chain = copy.deepcopy(chain)
    native = await h.mp.message_format_to_text(native_chain)

    # 插件：走 on_im_message（拍平 + 替换）→ 再渲染
    plug_ev = h.make_image_event(chain=copy.deepcopy(chain), message_id=f"r-{name}")
    await h.run_im(plug_ev)
    plug = await h.mp.message_format_to_text(plug_ev.message.chain)

    print(f"\n{'=' * 70}")
    print(f"【{name}】")
    print(f"--- 输入结构 ---")
    print("\n".join(tree(chain)))
    print(f"--- 原生渲染（无插件）---")
    print(f"  {native}")
    print(f"--- 插件渲染（v2.4.2）---")
    print(f"  {plug}")
    return native, plug


async def main():
    h = Harness(load_mode="lazy")
    await h.start()
    try:
        # ── S1 最简：1 条转发 1 节点 1 图 ──
        await render_pair(h, "S1 单转发单节点", _mc(
            _fwd(_mc(_text(NICK + "看看这张图"), _img())),
        ))

        # ── S2 真实聊天记录：1 条转发 3 节点（文本+图+文本）──
        await render_pair(h, "S2 单转发三节点", _mc(
            _fwd(
                _mc(_text(NICK + "第一张图"), _img()),
                _mc(_text(NICK + "这是文字消息")),
                _mc(_text(NICK + "第二张图"), _img()),
            ),
        ))

        # ── S3 两层嵌套：节点 B 里再套一条转发（各含图）──
        await render_pair(h, "S3 两层嵌套", _mc(
            _fwd(
                _mc(_text(NICK + "外层第一条"), _img()),
                _mc(_text(NICK + "外层第二条转发下面："),
                    _fwd(_mc(_text(NICK + "内层消息"), _img()))),
            ),
        ))

        # ── S4 三层嵌套 ──
        await render_pair(h, "S4 三层嵌套", _mc(
            _fwd(
                _mc(_text(NICK + "L1"),
                    _fwd(
                        _mc(_text(NICK + "L2"),
                            _fwd(_mc(_text(NICK + "L3 最深层"), _img()))),
                    )),
            ),
        ))

        # ── S5 Reply 里是转发 ──
        await render_pair(h, "S5 Reply 内 Forward", _mc(
            _text("引用了一条转发："),
            _reply(_mc(_fwd(_mc(_text(NICK + "被引用的转发"), _img())))),
        ))

        # ── S6 转发节点里是 Reply ──
        await render_pair(h, "S6 Forward 内 Reply", _mc(
            _fwd(_mc(_text(NICK + "他回复了："), _reply(_mc(_text("原消息"), _img())))),
        ))

        # ── S7 成环（恶意）──
        c1 = _mc(_text("环1"), _img())
        c2 = _mc(_text("环2"))
        c1.append(_fwd(c2))
        c2.append(_fwd(c1))
        await render_pair(h, "S7 成环", _mc(_fwd(c1)))

        # ── S8 深度隔断：16 层嵌套，插件只展开 4 层（forward_max_depth=4）──
        deep = _mc(_text("最深层"), _img())
        for _ in range(15):
            deep = _mc(_fwd(deep))
        # 用自定义配置的 harness
        h2 = Harness(load_mode="lazy")
        await h2.start()
        h2.plugin.forward_max_depth = 4  # 直接改实例配置
        print(f"\n{'=' * 70}")
        print("【S8 深度隔断 forward_max_depth=4（16 层嵌套）】")
        print(f"--- 输入结构（前 6 层）---")
        print("\n".join(tree(deep)[:6]))
        print(f"--- 原生渲染 ---")
        import copy
        native = await h2.mp.message_format_to_text(copy.deepcopy(deep))
        print(f"  {native}")
        ev = h2.make_image_event(chain=copy.deepcopy(deep), message_id="r-S8")
        await h2.run_im(ev)
        plug = await h2.mp.message_format_to_text(ev.message.chain)
        print(f"--- 插件渲染（展开 4 层）---")
        print(f"  {plug}")
        await h2.stop()
    finally:
        await h.stop()


if __name__ == "__main__":
    asyncio.run(main())
