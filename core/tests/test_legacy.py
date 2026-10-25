from core import legacy


def test_old_node_ids_and_keys_are_renamed():
    d = {"slots": [{"node": "FunPackSampler", "key": "funpack.clock"}], "conditioning_slot": "funpack"}
    assert legacy.data(d) == {"slots": [{"node": "FunUISampler", "key": "funui.clock"}], "conditioning_slot": "funui"}


def test_prose_and_other_packs_are_left_alone():
    assert legacy.name("FunPack is nice") == "FunPack is nice"
    assert legacy.name("funpackage") == "funpackage"
    assert legacy.name("KSampler") == "KSampler"


def test_a_project_saved_under_the_old_name_opens_with_the_new_one():
    from core import projects
    old = {"name": "Old", "conditioning_slot": "funpack", "sampler_slot": "funpack",
           "models": {"slots": [{"id": "sampler", "node": "FunPackSampler", "inputs": {"seed": 1}}]}}
    p = projects.Project.from_dict(old)
    assert p.conditioning_slot == "funui" and p.sampler_slot == "funui"
    assert p.models["slots"][0]["node"] == "FunUISampler"


def test_an_imported_workflow_with_old_node_ids_is_read_with_the_new_ones():
    from core import workflow_import
    wf = {"1": {"class_type": "FunPackSampler", "inputs": {"seed": 1}}}
    assert workflow_import._from_api(legacy.data(wf), [])[0]["node"] == "FunUISampler"
