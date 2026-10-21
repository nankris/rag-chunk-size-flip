# Resume after a sandbox reboot

1. `source /workspace/tools/setup.sh`; remain in this experiment directory.
2. `colab sessions` and `colab usage`. Only use `b9-132b`; never stop any other session.
3. Run `/workspace/tools/mlenv/bin/python scripts/sync.py`. It snapshots only `/content/132b` and prints the tail of the detached job log.
4. If `results/DONE` exists, run validate.py, analyze.py and figures.py locally, copy the final run log to figures/colab_run.txt, record usage and stop `b9-132b`.
5. If incomplete, inspect remote `job.pid`, process existence and run.log before relaunching. Do not duplicate a live process. Completed embedding batches, retrieval files and answer keys are reused. `scripts/launch.py` can relaunch a dead job after an updated script archive is uploaded. The running job is independent of the notebook kernel.
6. Do not allocate a replacement before checking existing sessions. Before every allocation read `/workspace/work/tig/_batch9/cu_floor.txt` and compare fresh `colab usage` against it. Baseline was 127.05 CU; shared-account difference is an upper bound. Stop at a conservative 5.6-CU account decrease, never exceed 6 for this experiment.

Preregistration: c9907fa. 3,600 fixed questions; analysis must not use a smaller completed subset as the confirmatory test. Main hypothesis is budget-controlled 1024 minus 128. Six simultaneous contrasts across two policies and three comparisons. A truncated run is INCONCLUSIVE.

Local monitoring can be restarted using Python subprocess.Popen with start_new_session=True for scripts/budget_guard.py and scripts/watch.py. Inspect existing processes first to avoid duplicate monitors. The GPU job remains independent of both monitors. After validation and analysis, run scripts/finalize_timing.py --eval-end; finish the writing, then scripts/finalize_timing.py, report.py, blog.py, export_log.py and check_publication.py.
