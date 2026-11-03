"""Command line of the Colab runner: ``python -m nig_timenet_torch.colab <command>``.

    republish BUNDLE_DIR   push a finished run's bundle folder (the Drive copy) to colab-results (rule E2)
    check-job FILE         validate a job.json before queueing it
"""

import argparse
import sys

from nig_timenet_torch.colab import bootstrap, job, publish

DEFAULT_REPO = "yuosef772424/nig-timenet-torch"
DEFAULT_TOKEN_PATH = "/content/drive/MyDrive/crypto/secrets/github_token.txt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.colab", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("republish", help="push a bundle folder to the results branch")
    rp.add_argument("bundle_dir")
    rp.add_argument("--repo", default=DEFAULT_REPO, help="owner/name (default %(default)s)")
    rp.add_argument("--url", default=None, help="full git URL instead of https://github.com/<repo>.git")
    rp.add_argument("--token-path", default=DEFAULT_TOKEN_PATH, help="one-line token file (default %(default)s)")
    rp.add_argument("--workdir", default="/content/nig_runner", help="scratch folder (default %(default)s)")
    rp.add_argument("--branch", default=publish.RESULTS_BRANCH)
    rp.add_argument("--force", action="store_true", help="replace runs/<job_id>/ when it already exists")
    cj = sub.add_parser("check-job", help="validate a job.json")
    cj.add_argument("file")
    args = ap.parse_args(argv)
    if args.cmd == "check-job":
        try:
            with open(args.file, encoding="utf-8") as fh:
                j = job.parse_job(fh.read())
        except (OSError, job.JobError) as e:
            print(f"job refused: {e}", file=sys.stderr)
            return 1
        print(f"OK: job {j['job_id']} @ {j['repo_ref'][:12]} entry {j['entry']} device {j['device']}")
        return 0
    token = bootstrap.read_token(args.token_path)
    if not token:
        print(f"no usable token in {args.token_path}", file=sys.stderr)
        return 1
    try:
        publish.republish(args.bundle_dir, args.url or f"https://github.com/{args.repo}.git", token, args.workdir,
                          args.branch, args.force)
    except bootstrap.BootstrapError as e:
        print(f"republish failed: {bootstrap.Secrets(token).mask(e)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
