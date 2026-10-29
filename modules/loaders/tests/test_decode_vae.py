def test_a_decode_only_vae_rides_on_the_main_one_and_only_the_decode_uses_it(monkeypatch):
    import torch
    from modules.loaders.vae import nodes as loader
    from modules.output.decode import nodes as decode

    made = {}

    def load(name, dtype):
        made[name] = type("Vae", (), {"name": name})()
        return made[name]

    monkeypatch.setattr(loader.FunUIVAELoader, "_load", staticmethod(load))
    plain = loader.FunUIVAELoader.execute("main.safetensors", "bfloat16")
    assert not hasattr(plain.result[0], "funui_decode_vae") and "decodes" not in plain.result[1]
    both = loader.FunUIVAELoader.execute("main.safetensors", "bfloat16", "sol.safetensors")
    main = both.result[0]
    assert main.name == "main.safetensors" and main.funui_decode_vae.name == "sol.safetensors"
    assert "sol.safetensors decodes" in both.result[1]

    seen = []
    monkeypatch.setattr(decode, "_plain_decode", lambda vae, latent, tile: seen.append(vae.name) or torch.zeros(1, 4, 4, 3))
    latent = {"samples": torch.zeros(1, 4, 1, 2, 2)}
    decode.FunUIDecode.execute(latent, main)
    decode.FunUIDecode.execute(latent, plain.result[0])
    assert seen == ["sol.safetensors", "main.safetensors"]
