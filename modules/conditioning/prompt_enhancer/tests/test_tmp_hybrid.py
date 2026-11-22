import threading
def test_seq(comfyui):
    from modules.conditioning.prompt_enhancer import nodes, rules, enhance as en, engine as eng
    path = "/tmp/q35/Qwen3.5-0.8B-Q4_K_M.gguf"
    def on_event(ev):
        if ev["kind"] == "action_request":
            print("ACTION", ev["tool"], ev["args"])
            threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    for i in range(2):
        for msg in ("What is a good lens for a close up?", "Make the clip 8 seconds long"):
            out = nodes.chat_reply("h:hy", msg, engine_model=path, max_length=60, do_sample=False, on_event=on_event, helper={"sheet": "Scenes: 1) a woman on a beach", "scenes": 1})
            print("HELPER", msg, "->", out["ok"], out["reply"][:80].replace("\n", " "), "|", out["status"][:60])
        out = nodes.chat_reply("c:hy", "Say hi in one sentence", engine_model=path, max_length=40, do_sample=False)
        print("CHAT", out["ok"], out["reply"][:60], out["status"][:60])
        clip = eng.EngineClip(nodes.ENGINE, live="rewrite")
        raw, status = en.generate(clip, "Rewrite the prompt with more detail.", "a woman walks on a beach", max_length=40, do_sample=False)
        print("REWRITE", raw[:60].replace("\n", " "), "|", status[:80])
        assert "failed" not in status
