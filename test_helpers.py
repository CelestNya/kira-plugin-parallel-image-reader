"""
parallel_image_reader 生产级行为测试。
stub 重依赖后加载真实 ParallelImageReader，零网络/DB 依赖。
直接 `python test_plugin_review.py` 运行。

覆盖修复：
  P-B1  stash 内存泄漏（TTL 清理 + 空会话回收 + 消费清 ts）
  P-B2  describe 单图异常隔离（一张崩不影响其余）
  P-S1  MULTI_DESC_PROMPT 死参数 total 已移除
"""
import sys
import types
import asyncio
import importlib.util
from pathlib import Path


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def load_module():
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
        def llm_request(**k):
            def deco(f): return f
            return deco

    # message element 类型 —— 真实 isinstance 判定需要它们是真 class
    class Image:
        def __init__(self, image=None):
            self.image = image

    class Sticker:
        def __init__(self, *a, **k): pass

    class Text:
        def __init__(self, content):
            self.content = content

    class Reply:
        def __init__(self, chain=None):
            self.chain = chain

    class Forward:
        def __init__(self, chains=None):
            self.chains = chains

    class LLMRequest:
        def __init__(self, messages=None):
            self.messages = messages or []

    _stub("core.logging_manager", get_logger=lambda *a, **k: _Logger())
    _stub("core.plugin", BasePlugin=_BasePlugin, PluginContext=object,
          register_tool=lambda *a, **k: (lambda f: f), on=_on, Priority=_Priority,
          logger=_Logger())

    # PIL 可能未装 —— stub 掉，测试不走真实图像编码
    if "PIL" not in sys.modules:
        pil = _stub("PIL")
        _stub("PIL.Image", new=lambda *a, **k: None, open=lambda *a, **k: None)
        pil.Image = sys.modules["PIL.Image"]
    _stub("core.chat.message_elements", Image=Image, Text=Text, Sticker=Sticker,
          Reply=Reply, Forward=Forward)
    _stub("core.chat.message_utils", KiraMessageEvent=object, KiraMessageBatchEvent=object)
    _stub("core.utils.common_utils", desc_img=None)
    _stub("core.provider", LLMRequest=LLMRequest)

    main_path = Path(__file__).parent / "main.py"
    spec = importlib.util.spec_from_file_location("pir_under_test", str(main_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, {"Image": Image, "Sticker": Sticker, "Text": Text,
                 "Reply": Reply, "Forward": Forward, "LLMRequest": LLMRequest}


class FakeCtx:
    def __init__(self):
        self._data_dir = "/tmp/pir_test"

    def get_plugin_data_dir(self):
        return self._data_dir
