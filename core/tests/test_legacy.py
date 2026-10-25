from core import legacy


def test_old_node_ids_and_keys_are_renamed():
    d = {"slots": [{"node": "FunPackSampler", "key": "funpack.clock"}], "conditioning_slot": "funpack"}
    assert legacy.data(d) == {"slots": [{"node": "FunUISampler", "key": "funui.clock"}], "conditioning_slot": "funui"}


def test_prose_and_other_packs_are_left_alone():
    assert legacy.name("FunPack is nice") == "FunPack is nice"
    assert legacy.name("funpackage") == "funpackage"
    assert legacy.name("KSampler") == "KSampler"
