from importlib import util
from pathlib import Path
import sys

_BRIDGE_PATH = Path(__file__).resolve().parents[1] / "src" / "mamba" / "mamba_ssm" / "modules" / "s6_attention_bridge.py"
_MODULE_NAME = "modules._s6_attention_bridge_impl"
_SPEC = util.spec_from_file_location(_MODULE_NAME, _BRIDGE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load S6 attention bridge from {_BRIDGE_PATH}")

_MODULE = util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)

S6AttentionBridge = _MODULE.S6AttentionBridge
compute_s6_attention_bridge_loss = _MODULE.compute_s6_attention_bridge_loss
configure_s6_attention_bridge = _MODULE.configure_s6_attention_bridge
get_s6_attention_bridge_log_interval = _MODULE.get_s6_attention_bridge_log_interval
is_s6_attention_bridge_active = _MODULE.is_s6_attention_bridge_active
record_s6_attention_bridge_loss = _MODULE.record_s6_attention_bridge_loss
reset_s6_attention_bridge_losses = _MODULE.reset_s6_attention_bridge_losses
s6_attention_bridge_is_enabled = _MODULE.s6_attention_bridge_is_enabled
set_s6_attention_bridge_runtime = _MODULE.set_s6_attention_bridge_runtime
