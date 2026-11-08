# الدفاتر (Notebooks)

> القسم العربي أدناه هو الدليل؛ بعده المرجع التفصيلي بالإنجليزية (يبقى كما هو).

## دليل الدفاتر الثلاثة

| الدفتر | لمن؟ | ما يفعل | فتح في Colab |
|---|---|---|---|
| [`main.ipynb`](main.ipynb) | **التجريب اليومي** | تشغيل كامل تفاعلي من الإعدادات إلى الربح بعد التكلفة، خطوة بخطوة، مع شرح عربي لكل قسم، ومزامنة Drive في الاتجاهين وبيئة بايثون مخزّنة | [افتح](https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/main.ipynb) |
| [`colab_nig_timenet_torch.ipynb`](colab_nig_timenet_torch.ipynb) | التشغيلات الطويلة المسجَّلة | مُشغِّل مؤتمت: طابور وظائف على GitHub ← تشغيل ← دفع النتائج إلى فرع `colab-results` | [افتح](https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/colab_nig_timenet_torch.ipynb) |
| [`data_pipeline.ipynb`](data_pipeline.ipynb) | قبل أي تدريب | يبني ملف البيانات (عقد artifact) من الشموع الخام | [افتح](https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/data_pipeline.ipynb) |

### `main.ipynb` في سطور
* **خلية إعدادات واحدة** (الثالثة): مصدر الكود، المسارات، وكل حقل من `RunSettings` بقيمته الافتراضية وتعليق عربي (المعنى، القيم المسموحة، الأثر). اختبار يقارن القيم بالكود فلا تبقى قيمة قديمة.
* **الكود:** `REPO_SOURCE = "auto" | "github" | "drive"`؛ نسخة المشروع الدائمة على Drive (`REPO_DIR`) تُحدَّث من GitHub عند `UPDATE_DRIVE_COPY = True` وتَرفض الكتابة فوق تعديلاتك (`DISCARD_LOCAL_CHANGES`).
  التوكن للقراءة فقط: سرّ Colab `GITHUB_TOKEN` ← ملف على Drive ← سؤال مخفي؛ لا يُطبع ولا يدخل `.git/config`. **لا `git push` من Colab.**
* **البيئة:** `python -m venv --system-site-packages` على Drive كملف tar واحد مفتاحه (بايثون + torch + المتطلبات)؛ يُعاد بناؤه عند تغيّر المفتاح (`colab/env_cache.py`).
* **المزامنة:** `WORK_ON = "drive"` (الافتراضي، يعمل على Drive مباشرة) أو `"local"` (قرص الجهاز ثم `push_run`)؛ `RESTORE_FROM` يستعيد تشغيلاً سابقاً من Drive (`colab/drive_sync.py`).
* **الأقسام:** البيانات والتقسيم ← النموذج ← التدريب ← تقرير الصحة ← تقرير الاختبار ← بطارية T-profit ← الملخّص ← بطارية مجلد منتهٍ، ثم جدول بأقسام الدفتر القديم (TensorFlow) التي ليس لها مقابل بعد وخطوة ROADMAP التي ستأتي بها.
* **لا منطق في الدفاتر (القاعدة G6):** كل دالة في وحدة بالحزمة ولها بطاقة واختبار؛ `tests/notebooks/test_main_notebook.py` يشغّل الدفتر على بيانات الاختبار من أوله لآخره مع Drive مزيّف (مجلد مؤقت).
* **ما لم يُجرَّب على Colab حقيقي:** مزامنة Drive الفعلية، بناء البيئة بـ pip حقيقي، وتشغيل GPU — الاختبارات تغطي المنطق بمجلدات مؤقتة ومستودع git محلي.

# Notebooks (reference, English)

[`colab_nig_timenet_torch.ipynb`](colab_nig_timenet_torch.ipynb) is the ONE notebook every Colab run starts from
(ROADMAP T3; owner, 2026-11-03: no cells pasted by hand). It is a thin caller of `src/nig_timenet_torch/colab/`; the
logic and its tests (`tests/colab/`) live there, and `tests/notebooks/test_colab_notebook.py` checks the notebook
statically (no outputs, no `def`, short cells, every call matches a real signature, every name is defined earlier).

Open it with [Open in Colab](https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/colab_nig_timenet_torch.ipynb),
choose a T4 GPU, then Runtime, then Run all. The first run asks for a GitHub token once (kept on Drive, never printed).

| Section | What it does | Code |
|---|---|---|
| 1. Setup | mounts Drive; clean-clones this repo and the TF repo (`rm -rf` first) and prints both commits; `pip install --no-deps` (Colab's CUDA torch is never replaced; the install is refused if torch's version changes); asserts and records the device (`setup.json`, rule E1) | `colab/bootstrap.py`, `colab/session.py`, `colab/device.py` |
| 2. Data | exports `preprocessing_output_latest.pkl.gz` as a contract artifact with the TF tool and checks it with the loader (`--verify-hashes`); skips the export when a sound artifact from the same source exists | `colab/prepare.py` |
| 3. Runner | queue -> run -> results, then republish if the push failed | `colab/runner.py`, `colab/publish.py`, `colab/contract.py` |

## The data notebook

[`data_pipeline.ipynb`](data_pipeline.ipynb) builds the training artifact with the native pipeline
(`src/nig_timenet_torch/data/`, same windows and targets as the TF project, `tests/data/test_parity_tf.py`) from the raw
candle files on Drive and writes `<NAME>_<stamp>.dataset` plus `<NAME>_latest.dataset`. It is a thin caller too:
`tests/notebooks/test_pipeline_notebooks.py` checks it statically and runs its non-Colab cells headless.

## The main notebook

[`main.ipynb`](main.ipynb) is one run from settings to profit, in Arabic (T3b): one settings cell (code source and Drive paths, then
`CHANGES` = every `RunSettings` field with its default and an Arabic comment, compared with the code by a test), then the Colab-only cells
(tagged `colab-only`: code source `colab/bootstrap.py::load_code`, cached env `colab/env_cache.py`, device check), the Drive folders
(`colab/drive_sync.py`: `run_paths`, `restore_run`, `push_run`), then the steps of `src/nig_timenet_torch/workflow/run.py` one cell each: `prepare`
(load, split) -> `build` -> `train` -> `health_step` -> `test_step` -> `battery_step` -> `write_summary`; the same steps
`jobs/baseline.py` calls. The last section reads a finished run folder and runs the battery again (`BATTERY_FOLDER`,
another cost per side or more variants, no model needed), and a last table of the TF `main.ipynb` sections that have no torch
equivalent yet. `tests/notebooks/test_main_notebook.py` checks it (every field in the settings cell, Arabic text, no `git push`, ROADMAP steps
exist) and runs it headless on the fixture artifact with a fake Drive.

## The runner

1. **Queue.** A job is a `job.json` at the root of branch `colab-queue` (schema and field meanings in
   `src/nig_timenet_torch/colab/job.py`; `python -m nig_timenet_torch.colab check-job job.json` validates one before
   you push it). `device` (`cuda` or `cpu`) is required. `repo_ref` pins the commit the job's code runs at; the runner
   itself comes from the notebook's clone of `TORCH_REF`.
2. **Run.** A finished job id (a `runs/<job_id>/` on `colab-results`) is refused unless the job sets `"rerun": true`. The
   runner asserts the device before starting the entry (a `cuda` job on a CPU VM fails without running), runs
   `python -m nig_timenet_torch.jobs.<entry>` under a wall-clock guard with `TMPDIR` inside the work dir, and validates the
   outputs, including the `device.json` the entry wrote (a `cuda` job whose record says `cpu` fails, rule E1).
3. **Results.** The bundle (`job.json`, `env.json` with device and package versions, `run_log.txt`, `bundle_summary.md`,
   `manifest.json`, `device.json`, the outputs) is written to Drive FIRST (`MyDrive/nig_timenet_runs/<job_id>/`; the work
   dir if Drive is not mounted), then pushed to branch `colab-results` under `runs/<job_id>/`. Outputs above 1 MiB stay on
   Drive. A later session reads `runs/<job_id>/` from GitHub; the owner copies nothing.
4. **Republish (rule E2).** If the push failed, the bundle is already safe on Drive. The notebook's last cell retries it, or in a fresh
   runtime: `python -m nig_timenet_torch.colab republish /content/drive/MyDrive/nig_timenet_runs/<job_id>`. It needs only
   that folder and the token file; nothing under `/tmp`, no earlier clone and no runner object.

Entries are modules of `nig_timenet_torch.jobs`. `device_smoke` proves the whole path on a new VM (queue it first with
`"device": "cuda"` and `"config": {"device": "cuda"}`). Training entries (T5r: `baseline`) use
`colab/identity.py::identity_for_run` for the run identity.

Smoke job (rule E5): before a long `baseline` job on a commit, queue `docs/jobs/baseline-smoke-001.json` (the same entry on the
real artifact, `max_rows` 20000 per split, 1 epoch, a few minutes on a T4; `max_minutes` 15) on that commit and read its
`baseline_summary.json` (targets, `reg_target_mode`, `input_key`, window counts) before queueing the long job.

The notebook is committed without outputs. First Colab runs on a T4 (2026-11-03, `docs/STATUS.md` T3 row): setup, export,
device and publish worked; the baseline job has not yet finished on the GPU (`docs/bottlenecks.csv` B-024). The previous demo notebook (parity
table, all model variants, a small training run on the fixture) is in git history at commit `b34579c`.

`baseline` (T5r) trains one seed of the default model; the job files `docs/jobs/baseline-001-s{0,1}.json` carry the
placeholder `repo_ref` of 40 zeros: the planner replaces it with the merge commit at queue time. Running the same job
again with the same `run_dir` and `"rerun": true` resumes from the Drive checkpoint.
