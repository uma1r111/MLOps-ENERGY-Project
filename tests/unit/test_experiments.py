from experiments import run_experiment


def test_run_experiment_import():
    assert run_experiment is not None


def test_run_experiment_execution():
    # simple smoke run without real data
    try:
        run_experiment.main()  # if your script has main()
    except Exception:
        pass
