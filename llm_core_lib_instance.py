from llm_core_lib.llm_core_lib import LlmCoreLib


class LlmCoreLibInstance(object):
    _app_instance = None

    @staticmethod
    def init(core_lib_cfg):
        if not LlmCoreLibInstance._app_instance:
            LlmCoreLibInstance._app_instance = LlmCoreLib(core_lib_cfg)

    @staticmethod
    def get() -> LlmCoreLib:
        return LlmCoreLibInstance._app_instance
