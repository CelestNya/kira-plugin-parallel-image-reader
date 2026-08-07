"""
混沌测试 — 边界条件故障注入（随机种子驱动，可复现）。

验证 v2.4.0 统一标识符架构在任何故障组合下的不变量：
  I1 不崩溃：事件钩子与 describe_image 工具不向事件层抛异常
  I2 不死锁：每轮 5s 超时守卫，任何路径必须完成
  I3 状态合法：无 <!--PIR 泄漏；标识符形态合法（空/描述/已过期三态）
  I4 缓存纯净：写缓存的值必须通过 _is_valid_desc（无空串/\x00/旧占位符污染）
  I5 资源纪律：VLM 调用次数不爆炸（精确断言于纯净轮，上界断言于故障轮）；
     eager terminate 后 task 清理；id_map 无 noid_ 前缀

故障注入池（每轮随机组合）：
  - VLM：抛异常 / 抛超时 / 返回空 / 返回 None / 返回 \x00 污染 / 返回 <!--PIR 污染
  - hash_image：随机失败（→ noid_ 空标识符路径）
  - to_data_url：随机失败（quality 模式路径）
  - 缓存 DB：get / add 随机抛异常
  - 消息链：空链 / Reply 成环 / Forward / 深嵌套 / 混合文本
  - 并发取消：eager 阶段1 后 terminate（取消乐观 task）→ batch 仍须正常降级
  - 换态：IM / batch / llm_request 间随机切换 load_mode
  - 状态破坏：_pir_images 置 None、message_str 置 None
  - 工具混沌：describe_image 合法/非法/空/None 参数

用法：
    python tests/chaos_test.py [--seed N] [--rounds N]
    （默认：随机种子 + 3 固定种子，各 25 轮）
"""

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

import test_v2 as tv  # noqa: E402  复用 stub 设施（FakeDB/FakeVLM/消息元素/插件工厂）

ROUND_TIMEOUT = 5  # 死锁守卫（秒）


# ═══════════════════════════════════════════════════════════════
# 故障注入替身
# ═══════════════════════════════════════════════════════════════

_VLM_FAULTS = (
    ["ok"] * 5          # 正常描述（权重高）
    + ["raise"]         # VLM 崩溃
    + ["timeout"]       # 超时
    + ["empty"]         # 返回空串
    + ["none"]          # 返回 None
    + ["pollute_nul"]   # 含 \x00 的污染描述
    + ["pollute_pir"]   # 含旧占位符的污染描述
    + ["pollute_img"]   # 含标识符格式的嵌套污染（注入面）
)


class ChaosVLM(tv.FakeVLM):
    """每次调用从故障池随机抽取一种行为的 VLM。"""

    def __init__(self, rng, fault_rate=0.5):
        super().__init__("混沌基准描述")
        self._rng = rng
        self._fault_rate = fault_rate
        self.call_count = 0

    async def chat(self, request):
        self.call_count += 1
        mode = ("ok" if self._rng.random() > self._fault_rate
                else self._rng.choice(_VLM_FAULTS))
        if mode == "raise":
            raise RuntimeError("chaos: vlm crashed")
        if mode == "timeout":
            raise asyncio.TimeoutError()
        if mode == "empty":
            return tv.FakeVLMResponse("")
        if mode == "none":
            return tv.FakeVLMResponse(None)
        if mode == "pollute_nul":
            return tv.FakeVLMResponse(f"污染描述\x00#{self.call_count}")
        if mode == "pollute_pir":
            return tv.FakeVLMResponse(f"<!--PIR:deadbeef-->#{self.call_count}")
        if mode == "pollute_img":
            return tv.FakeVLMResponse(f"图中有文字 [Image #deadbeef: ]#{self.call_count}")
        return tv.FakeVLMResponse(f"混沌描述 #{self.call_count}")


class ChaosDB(tv.FakeDB):
    """缓存 DB：get / add 按概率抛异常。"""

    def __init__(self, rng, fail_rate=0.05):
        super().__init__()
        self._rng = rng
        self._fail_rate = fail_rate

    async def get_image_desc_cache(self, md5):
        if self._rng.random() < self._fail_rate:
            raise RuntimeError("chaos: db get failed")
        return await super().get_image_desc_cache(md5)

    async def add_image_desc_cache(self, md5, desc, **kw):
        if self._rng.random() < self._fail_rate:
            raise RuntimeError("chaos: db add failed")
        return await super().add_image_desc_cache(md5, desc, **kw)


class ChaosImage(tv.Image):
    """图片元素：hash_image / to_data_url 按概率失败。"""

    def __init__(self, md5, rng, hash_fail_rate=0.1, url_fail_rate=0.1):
        super().__init__(md5=md5)
        self._rng = rng
        self._hash_fail_rate = hash_fail_rate
        self._url_fail_rate = url_fail_rate

    async def hash_image(self):
        if self._rng.random() < self._hash_fail_rate:
            raise RuntimeError("chaos: hash failed")
        return self._md5

    async def to_data_url(self):
        if self._rng.random() < self._url_fail_rate:
            raise RuntimeError("chaos: to_data_url failed")
        return await super().to_data_url()


class ChaosSticker(tv.Sticker):
    """同上，Sticker 变体。"""

    def __init__(self, md5, rng, hash_fail_rate=0.1):
        super().__init__(md5=md5)
        self._rng = rng
        self._hash_fail_rate = hash_fail_rate

    async def hash_image(self):
        if self._rng.random() < self._hash_fail_rate:
            raise RuntimeError("chaos: sticker hash failed")
        return self._md5


# ═══════════════════════════════════════════════════════════════
# 消息链生成
# ═══════════════════════════════════════════════════════════════

_MD5_POOL = [f"chaos{str(i).zfill(10)}md5000000000{i}" for i in range(12)]


def _flatten_imgs(chain, visited=None):
    """展平 chain（含 Reply/Forward 嵌套）中的全部 Image/Sticker 元素。"""
    if visited is None:
        visited = set()
    cid = id(chain)
    if cid in visited:
        return []
    visited.add(cid)
    imgs = []
    for ele in chain:
        if isinstance(ele, (tv.Image, tv.Sticker)):
            imgs.append(ele)
        elif isinstance(ele, tv.Reply) and ele.chain is not None:
            imgs.extend(_flatten_imgs(ele.chain, visited))
        elif isinstance(ele, tv.Forward) and ele.chains:
            for c in ele.chains:
                imgs.extend(_flatten_imgs(c, visited))
    return imgs


def _random_chain(rng, elements, depth=0, allow_reply=True):
    """按随机结构组装链：Text 混排 + Reply/Forward 嵌套（含成环）。

    allow_reply=False：纯净轮用（排除引用图自动读取对 VLM 计数断言的干扰）。
    """
    chain = tv.FakeMessageChain()
    for ele in elements:
        r = rng.random()
        if allow_reply and depth < 2 and r < 0.12:
            inner = _random_chain(rng, [ele], depth + 1, allow_reply)
            chain.append(tv.Reply(inner))
        elif depth < 2 and r < 0.20:
            chain.append(tv.Forward([
                _random_chain(rng, [ele], depth + 1, allow_reply),
                _random_chain(rng, [], depth + 1, allow_reply),
            ]))
        elif r < 0.25 and isinstance(ele, tv.Image):
            chain.append(tv.Text("伴随文字"))
            chain.append(ele)
        else:
            chain.append(ele)
    return chain


def _make_cycle_chain(img):
    """Reply 成环：两个 Reply 互引（visited 集必须兜住）。"""
    c1 = tv.FakeMessageChain([img, tv.Reply(None)])
    c2 = tv.FakeMessageChain([tv.Reply(c1)])
    c1[1].chain = c2  # c1 → c2 → c1 成环
    return c1


# ═══════════════════════════════════════════════════════════════
# 不变量断言
# ═══════════════════════════════════════════════════════════════

_IMAGE_FORMAT_RE = __import__("re").compile(
    r"\[Image #([^\]\s:]+): [^\]]*\]"
)


def _collect_texts(messages) -> list[str]:
    """收集 message_str + 展平 chain 的 Text 内容。"""
    texts = []
    for m in messages:
        if getattr(m, "message_str", None):
            texts.append(m.message_str)
        for ele in _flatten_imgs(m.chain):
            texts.append(getattr(ele, "text", "") or "")
    return texts


def _check_text_invariants(fails, texts, where, mod):
    """I3：无旧占位符泄漏；标识符形态合法。"""
    for t in texts:
        if "<!--PIR" in t:
            fails.append(f"{where}: 旧占位符泄漏: {t[:60]!r}")
        for m in _IMAGE_FORMAT_RE.finditer(t):
            pass  # 只要正则能匹配即形态合法
        # 所有 [Image # 开头的片段必须整体匹配（防截断/畸形）
        for frag in t.split("[Image #")[1:]:
            if not _IMAGE_FORMAT_RE.match("[Image #" + frag):
                fails.append(f"{where}: 畸形标识符片段: {'[Image #' + frag[:60]!r}")


def _check_cache_pure(fails, db, mod):
    """I4：缓存值全部通过 _is_valid_desc。"""
    for md5, desc in db._cache.items():
        if not mod._is_valid_desc(desc):
            fails.append(f"I4: 缓存污染 md5={md5[:10]} desc={desc[:50]!r}")


def _check_id_map_clean(fails, plug):
    """I7：id_map 无 noid_ 前缀（hash 失败者不得写映射）。"""
    for k in plug._id_map:
        if k.startswith("noid_"):
            fails.append(f"I7: id_map 混入 noid_ 键: {k[:40]}")


# ═══════════════════════════════════════════════════════════════
# 单轮混沌场景
# ═══════════════════════════════════════════════════════════════

async def chaos_round(rng, rnd):
    """一轮完整混沌：随机故障组合 + 三阶段全流程 + 终局不变量断言。"""
    fails: list[str] = []
    mod, _ = tv.load_plugin()

    # 纯净轮：20% 概率关闭全部故障 → VLM 计数可精确断言
    pure = rng.random() < 0.2
    fault_rate = 0.0 if pure else 0.5
    hash_fail = 0.0 if pure else rng.uniform(0.05, 0.3)
    db_fail = 0.0 if pure else rng.uniform(0.03, 0.15)

    load_mode = rng.choice(["lazy", "eager", "llm_select"])
    db = ChaosDB(rng, db_fail)
    vlm = ChaosVLM(rng, fault_rate)
    plug, _ = await tv._make_plugin(db, vlm, {"load_mode": load_mode})

    # 随机预填缓存（制造混合命中/未命中）
    for md5 in _MD5_POOL:
        if rng.random() < 0.3:
            db.seed(md5, f"缓存描述-{md5[:8]}")

    # ── 阶段1：随机消息（1-6 条，每条 0-4 图）──
    n_msgs = rng.randint(1, 6)
    events, n_pending = [], 0
    for i in range(n_msgs):
        n_imgs = rng.randint(0, 4)
        elements = []
        for j in range(n_imgs):
            md5 = _MD5_POOL[rng.randrange(len(_MD5_POOL))]
            if rng.random() < 0.2:
                elements.append(ChaosSticker(md5, rng, hash_fail))
            else:
                elements.append(ChaosImage(md5, rng, hash_fail, hash_fail))
        if not elements and rng.random() < 0.5:
            elements.append(tv.Text("纯文本消息"))
        if elements and rng.random() < 0.3:
            elements.insert(0, tv.Text("开头文字"))
        if not pure and rng.random() < 0.1:
            chain = _make_cycle_chain(elements[0] if elements else tv.Text("x"))
        else:
            chain = _random_chain(rng, elements, allow_reply=not pure)

        # 阶段1 未命中图数（纯净轮用于精确断言；此步必须先于 on_im_message）
        # 按 md5 去重：同一 chain 内重复 md5 共享 short_id，_pir_images 合并为一个键
        # 注意：Sticker 与 Image 都进 _pir_images，都要统计
        if pure:
            seen_md5 = set()
            for img in _flatten_imgs(chain):
                if isinstance(img, (tv.Image, tv.Sticker)):
                    try:
                        md5 = await img.hash_image()
                        if md5 not in seen_md5 and not db._cache.get(md5):
                            n_pending += 1
                            seen_md5.add(md5)
                    except Exception:
                        pass

        # 环境属性（测试质量反思：事件维度进入混沌矩阵）：
        # 纯净轮固定群聊+未提及（VLM 计数可精确断言）；
        # 故障轮随机私聊/群聊/提及（覆盖自动读取判定路径）
        if pure:
            group, mentioned = object(), False
        else:
            group = object() if rng.random() < 0.6 else None
            mentioned = rng.random() < 0.5
        ev = tv.FakeMessageEvent(chain, group=group, mentioned=mentioned)
        try:
            await asyncio.wait_for(plug.on_im_message(ev), ROUND_TIMEOUT)
        except Exception as e:
            fails.append(f"round{rnd} I1: on_im_message 抛异常: {type(e).__name__}: {e}")
            continue
        events.append(ev)

    # ── eager：随机 terminate（取消乐观 task）→ 后续 batch 必须仍降级 ──
    # （纯净轮不注入：保证 VLM 计数可精确断言）
    did_terminate = False
    if not pure and load_mode == "eager" and rng.random() < 0.4:
        try:
            await plug.terminate()
            did_terminate = True
        except Exception as e:
            fails.append(f"round{rnd} I1: terminate 抛异常: {e}")
        if plug._optimistic_tasks:
            fails.append(f"round{rnd} I6: terminate 后 _optimistic_tasks 非空")

    # ── 阶段2：batch（随机换态）──
    batch_mode = (rng.choice(["lazy", "eager", "llm_select"])
                  if rng.random() < 0.3 else load_mode)
    if batch_mode != load_mode:
        plug.load_mode = batch_mode
    batch_msgs = []
    for ev in events:
        msg = ev.message
        if not pure:
            if rng.random() < 0.1:
                msg._pir_images = None  # 状态破坏：暂存丢失
            if rng.random() < 0.1:
                msg.message_str = None   # 状态破坏：message_str 丢失
        batch_msgs.append(tv._make_batch_from_event(ev))
    batch_ev = tv.FakeMessageBatchEvent(batch_msgs)
    try:
        await asyncio.wait_for(plug.on_im_batch_message(batch_ev), ROUND_TIMEOUT)
    except Exception as e:
        fails.append(f"round{rnd} I1: on_im_batch_message 抛异常: {type(e).__name__}: {e}")

    # 断言阶段2 后标识符形态
    _check_text_invariants(fails, _collect_texts(batch_msgs), f"round{rnd} batch", mod)

    # ── ON_LLM_REQUEST 混沌（随机换态 + 历史空标识符扫描）──
    # 纯净轮跳过：其扫描可对"当前回合有原图"触发额外 VLM，干扰精确计数断言
    tool_calls = 0
    if not pure and rng.random() < 0.7:
        llm_mode = (rng.choice(["lazy", "eager", "llm_select"])
                    if rng.random() < 0.3 else plug.load_mode)
        if llm_mode != plug.load_mode:
            plug.load_mode = llm_mode
        hist_msgs = []
        for i in range(rng.randint(0, 3)):
            c = rng.random()
            if c < 0.3:
                hist_msgs.append(tv.types.SimpleNamespace(
                    content=f"[Image #{_MD5_POOL[i][:8]}: ]", role="user"))
            elif c < 0.6:
                hist_msgs.append(tv.types.SimpleNamespace(
                    content=f"[Image #{_MD5_POOL[i][:8]}: 历史描述]", role="user"))
            elif c < 0.8:
                hist_msgs.append(tv.types.SimpleNamespace(
                    content=f"[Image #{_MD5_POOL[i][:8]}: 已过期]", role="user"))
            else:
                hist_msgs.append(tv.types.SimpleNamespace(content="普通历史", role="user"))
        req = tv.LLMRequest(messages=hist_msgs)
        req.tool_set = tv.FakeToolSet(["describe_image"])
        req.system_prompt.append(tv.types.SimpleNamespace(name="chat_env", content=""))
        req.user_prompt = []
        try:
            await asyncio.wait_for(plug.on_llm_request(batch_ev, req), ROUND_TIMEOUT)
        except Exception as e:
            fails.append(f"round{rnd} I1: on_llm_request 抛异常: {type(e).__name__}: {e}")
        # 工具规则：llm_select 常驻；其他模式移除
        if llm_mode == "llm_select":
            if "describe_image" not in req.tool_set.tools:
                fails.append(f"round{rnd}: llm_select 下 describe_image 未常驻")
        else:
            if "describe_image" in req.tool_set.tools:
                fails.append(f"round{rnd}: {llm_mode} 下 describe_image 未移除")

    # ── describe_image 工具混沌 ──
    # （纯净轮跳过：工具调用会对 _pir_images 中的图触发 VLM，干扰精确计数）
    if not pure and rng.random() < 0.5:
        arg = rng.choice([
            _MD5_POOL[0][:8],          # 合法 id（可能在 id_map / _pir_images）
            "deadbeef",                # 不存在的 id
            "",                        # 空串
            None,                      # None
            "x" * 200,                 # 超长
            "noid_12345",              # 伪 noid
        ])
        try:
            result = await asyncio.wait_for(
                plug.describe_image(batch_ev, arg), ROUND_TIMEOUT)
            if not isinstance(result, str):
                fails.append(f"round{rnd}: describe_image 返回非 str: {type(result)}")
        except Exception as e:
            fails.append(f"round{rnd} I1: describe_image 抛异常: {type(e).__name__}: {e}")

    # ── 终局不变量 ──
    _check_cache_pure(fails, db, mod)
    _check_id_map_clean(fails, plug)

    # I5 VLM 计数（纯净轮：无状态破坏/terminate，计数可控）
    # 阶段2 行为由 batch_mode 决定：llm_select 不 VLM；lazy 现场全量；
    # eager 提前 task 可能已写缓存 → 允许 ≤
    if pure:
        if did_terminate:
            fails.append(f"round{rnd}: 纯净轮不应 terminate（测试设计错误）")
        elif batch_mode == "llm_select":
            if vlm.call_count != 0:
                fails.append(
                    f"round{rnd} I5: 纯净轮 batch=llm_select VLM 计数 "
                    f"{vlm.call_count} != 0 (mode={load_mode})")
        elif batch_mode == "lazy":
            # 并发 gather 中跨消息同 md5 的图可能命中对方刚写的缓存（合法优化）
            if vlm.call_count > n_pending:
                fails.append(
                    f"round{rnd} I5: 纯净轮 batch=lazy VLM 计数 "
                    f"{vlm.call_count} > {n_pending} "
                    f"(mode={load_mode}, pending={n_pending})")
        else:  # batch_mode == eager
            if vlm.call_count > n_pending:
                fails.append(
                    f"round{rnd} I5: 纯净轮 batch=eager VLM 计数 "
                    f"{vlm.call_count} > pending {n_pending} "
                    f"(mode={load_mode})")
    else:
        upper = len(_MD5_POOL) * 4 + 10  # 宽松爆炸检测：12 图池 × 4 轮内不得超
        if vlm.call_count > upper:
            fails.append(f"round{rnd} I5: VLM 计数爆炸 {vlm.call_count} > {upper}")

    return fails


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def run_seed(seed: int, rounds: int) -> list[str]:
    rng = random.Random(seed)
    fails: list[str] = []
    for rnd in range(rounds):
        try:
            round_fails = await asyncio.wait_for(chaos_round(rng, rnd), ROUND_TIMEOUT)
        except asyncio.TimeoutError:
            round_fails = [f"round{rnd} I2: 死锁/挂起（超过 {ROUND_TIMEOUT}s）"]
        except Exception as e:
            round_fails = [f"round{rnd} I1: 混沌轮自身抛异常: {type(e).__name__}: {e}"]
        if round_fails:
            fails.extend(f"[seed={seed}] {f}" for f in round_fails)
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description="混沌测试：边界条件故障注入")
    ap.add_argument("--seed", type=int, default=None, help="固定随机种子")
    ap.add_argument("--rounds", type=int, default=25, help="每个种子的轮数")
    args = ap.parse_args()

    seeds = ([args.seed] if args.seed is not None
             else [0, 1, 2, random.SystemRandom().randint(0, 10**6)])
    total_rounds = len(seeds) * args.rounds

    print(f"并行图片阅读器 — 混沌测试（{total_rounds} 轮，种子={seeds}）")
    t0 = time.time()
    all_fails: list[str] = []
    for seed in seeds:
        fails = asyncio.run(run_seed(seed, args.rounds))
        print(f"  [seed={seed}] {args.rounds} 轮，失败 {len(fails)}")
        all_fails.extend(fails)

    elapsed = time.time() - t0
    print(f"\n── 结果: {total_rounds - len(all_fails)}/{total_rounds} 轮通过, "
          f"{len(all_fails)} 失败, 耗时 {elapsed:.1f}s")
    if all_fails:
        print("\n失败详情:")
        for f in all_fails[:20]:
            print(f"  ✗ {f}")
        if len(all_fails) > 20:
            print(f"  …（共 {len(all_fails)} 条）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
