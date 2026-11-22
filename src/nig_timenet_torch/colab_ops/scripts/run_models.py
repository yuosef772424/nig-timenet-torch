"""PURPOSE: Script shipped to a Colab runtime: single-model runs (several architectures / encoders) of the
    sequential-gradient branch, then the evaluation battery at n_variants 1 and 6.
TAGS: run models script, grad stack, single model, battery, colab script, sequential gradient branch
PITFALLS: runs with THAT branch's code on PYTHONPATH (it imports ``workflow.grad_stack``, absent on this branch);
    never imported by the package or the tests; the artifact is read from /content/art1.

Shipped by ``colab_ops runs``: ``run_models.py MODELS LR LR_MIN EPOCHS ROWS [DEVICE]`` (MODELS: comma list of
architectures / encoder names)."""

import sys
import time


def main(argv: list[str]) -> None:
    import nig_timenet_torch
    from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE
    from nig_timenet_torch.workflow import battery_run
    from nig_timenet_torch.workflow import grad_stack as gs

    print("code:", nig_timenet_torch.__file__, flush=True)
    models, lr, lr_min = argv[0].split(","), float(argv[1]), float(argv[2])
    epochs, rows, device = int(argv[3]), int(argv[4]), argv[5] if len(argv) > 5 else "cuda"
    for arch in models:
        t0 = time.time()
        cfg = gs.GradStackConfig(
            data="/content/art1",
            architecture=arch,
            epochs=epochs,
            batch_size=256,
            seed=0,
            device=device,
            lr=lr,
            lr_min=lr_min,
            max_rows=rows,
            grad_target="output",
            variants=("full",),
        )
        out = f"/content/uhub_runs/{arch}_lr{lr:g}"
        gs.run_grad_stack(cfg, out)
        print(
            f"\n##### {arch}: trained in {time.time() - t0:.0f}s; battery (n_variants=1, as in the reports)", flush=True
        )
        gs.grad_stack_battery(out, cost=DEFAULT_COST_PER_SIDE, show=False)
        folder = f"{out}/runs/full"  # the same tables, n_variants = the 6 models tried so far
        battery_run.battery_from_folder(folder, cost=DEFAULT_COST_PER_SIDE, n_variants=6, out_dir=f"{folder}/n6")
        print(f"##### {arch} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
