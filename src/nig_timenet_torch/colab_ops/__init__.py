"""PURPOSE: Operate a Colab runtime from outside: bridge client, setup / run recipes, one CLI; the steps that were
    retyped in every session (connect, get the code, copy the artifact, start a job, wait, read the report).
TAGS: colab ops, colab bridge, remote runtime, setup recipe, run recipe, cli, repeated steps
PITFALLS: the bridge URL and token come from the environment (CB_URL, COLAB_BRIDGE_TOKEN), never from a file in the
    repository; recipes build shell text that reads the GitHub token from a Drive file ON the runtime, so no secret is
    ever in a command; a runtime switch (CPU -> GPU, restart) is a new empty machine: run ``setup`` again.

``python -m nig_timenet_torch.colab_ops --help``. Nothing here imports ``colab_bridge`` until a command talks to a
runtime (the package is optional: ``pip install git+https://github.com/yuosef772424/colab-mcp-bridge``).
"""
