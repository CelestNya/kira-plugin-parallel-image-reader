"""parallel_image_reader 行为测试 —— 直接 python 运行，零外部依赖。"""
import asyncio
import types as _t

from test_helpers import load_module, FakeCtx

mod, T = load_module()
ParallelImageReader = mod.ParallelImageReader
Image = T["Image"]
Text = T["Text"]

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'} {name}" + ("" if cond else f" :: {detail}"))


def make_plugin(cfg=None):
    p = ParallelImageReader(FakeCtx(), cfg or {})
    return p


class FakeSession:
    def __init__(self, sid):
        self.sid = sid


class FakeEvent:
    def __init__(self, sid, chain):
        self.session = FakeSession(sid)
        self.message = _t.SimpleNamespace(chain=chain)


# ── P-S1: MULTI_DESC_PROMPT 只含 {index}，format 单参不炸 ──────
def test_ps1_prompt_format():
    try:
        out = mod.MULTI_DESC_PROMPT.format(index=2)
        ok = "第 2 张" in out
        check("P-S1: MULTI_DESC_PROMPT.format(index=) 正常", ok, f"out={out[:40]}")
    except Exception as e:
        check("P-S1: format 不抛异常", False, str(e))


# ── P-B1a: 截图后打时间戳，消费后清 ts ────────────────────────
async def test_pb1a_stash_lifecycle():
    p = make_plugin()
    ev = FakeEvent("s1", [Text("hi"), Image(image="a"), Image(image="b")])
    await p.on_im_message(ev)

    check("P-B1a: 截图后 stash 有会话", "s1" in p._stash, f"stash={list(p._stash)}")
    check("P-B1a: 截图后写入时间戳", "s1" in p._stash_ts, f"ts={list(p._stash_ts)}")
    check("P-B1a: 两张图都进 stash", len(p._stash["s1"]) == 2, f"n={len(p._stash['s1'])}")
    check("P-B1a: chain 中图被替换为占位符",
          all(isinstance(e, Text) for e in ev.message.chain), "not all Text")


# ── P-B1b: 无图消息不留空会话 ─────────────────────────────────
async def test_pb1b_no_image_no_leak():
    p = make_plugin()
    ev = FakeEvent("s2", [Text("just text")])
    await p.on_im_message(ev)
    check("P-B1b: 无图消息不留 stash 会话", "s2" not in p._stash, f"stash={list(p._stash)}")
    check("P-B1b: 无图消息不留时间戳", "s2" not in p._stash_ts, f"ts={list(p._stash_ts)}")


# ── P-B1c: TTL 过期会话被清理 ─────────────────────────────────
async def test_pb1c_ttl_eviction():
    p = make_plugin()
    # 手动塞一个"老"会话
    p._stash["old"] = {"__IMG__old__0__": Image(image="x")}
    p._stash_ts["old"] = -99999  # monotonic 远古值 → 必过期

    # 新消息触发清理
    ev = FakeEvent("fresh", [Text("hello")])
    await p.on_im_message(ev)

    check("P-B1c: TTL 过期会话被清理", "old" not in p._stash, f"stash={list(p._stash)}")
    check("P-B1c: 过期会话时间戳被清理", "old" not in p._stash_ts, f"ts={list(p._stash_ts)}")


class FakeImageElem:
    """模拟 message_elements.Image，支持 hash_image / to_data_url。"""
    def __init__(self, md5, fail=False):
        self._md5 = md5
        self._fail = fail

    async def hash_image(self):
        if self._fail:
            raise RuntimeError("corrupted image bytes")
        return self._md5


class FakeDB:
    async def get_image_desc_cache(self, md5):
        return None

    async def add_image_desc_cache(self, md5, desc, count=1, last_seen=0):
        pass


def _wire_vlm(p, desc_map):
    """注入 fake db / provider_mgr，并 patch desc_img 返回 desc_map[md5]。"""
    p.ctx.db = FakeDB()
    p.ctx.provider_mgr = _t.SimpleNamespace(get_default_vlm=lambda: object())

    async def fake_desc_img(client, image, prompt):
        return desc_map.get(image._md5, "")

    mod.desc_img = fake_desc_img


# ── P-B2: 一张图 hash 抛异常，其余正常返回 ────────────────────
async def test_pb2_describe_isolation():
    p = make_plugin({"max_concurrent": 3})
    _wire_vlm(p, {"good1": "desc-good1", "good2": "desc-good2"})

    imgs = [
        FakeImageElem("good1"),
        FakeImageElem("bad", fail=True),
        FakeImageElem("good2"),
    ]

    raised = None
    try:
        results = await p._describe_parallel(imgs)
    except Exception as e:
        raised = e
        results = None

    check("P-B2: _describe_parallel 不向上抛异常", raised is None, f"raised={raised}")
    if results is not None:
        check("P-B2: 返回长度与输入一致", len(results) == 3, f"len={len(results)}")
        check("P-B2: 好图正常描述", results[0] == "desc-good1" and results[2] == "desc-good2", f"got={results}")
        check("P-B2: 坏图降级为空串", results[1] == "", f"got={results[1]!r}")


async def run_all():
    print("\n[P-S1] prompt 死参数移除")
    test_ps1_prompt_format()
    print("\n[P-B1] stash 内存泄漏防护")
    await test_pb1a_stash_lifecycle()
    await test_pb1b_no_image_no_leak()
    await test_pb1c_ttl_eviction()
    print("\n[P-B2] describe 单图异常隔离")
    await test_pb2_describe_isolation()


def main():
    asyncio.run(run_all())
    print(f"\n{'='*50}\nPASSED: {len(PASSED)}  FAILED: {len(FAILED)}")
    if FAILED:
        for n, d in FAILED:
            print(f"  x {n}: {d}")
        raise SystemExit(1)
    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
