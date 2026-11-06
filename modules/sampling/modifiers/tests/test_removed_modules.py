import json


def test_settings_saved_for_the_removed_video_detail_module_still_load(comfyui):
    from modules.sampling.modifiers.nodes import FunUIModifierSettings
    payload = json.dumps({"video_detail": {"enabled": True, "amount": 1.6}, "sharpen": {"enabled": False}})
    assert FunUIModifierSettings.validate_inputs(payload) is True
    out = FunUIModifierSettings.execute(payload)
    assert "video_detail" not in out.args[0] and "sharpen" in out.args[0]
