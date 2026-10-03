import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "experiments/self_evolving/final_method/unified/scripts"

def test_task_splits_are_disjoint():
    split = json.load(open(ROOT / "splits/timemmd_531_181_182_v1.json"))
    train, dev, test = map(lambda name: set(split[name]), ("train", "dev", "test"))
    assert not train & dev and not train & test and not dev & test


def test_posttime_parser_is_strict():
    source = (SCRIPTS / "posttime.py").read_text()
    compile(source, "posttime.py", "exec")
    assert "len(values) == horizon" in source


def test_scripts_compile():
    for name in ("learned_data", "learned_common", "learned_models", "time_llm",
                 "mm_tsflib", "posttime", "learned_score"):
        compile((SCRIPTS / f"{name}.py").read_text(), name, "exec")


def test_parallel_jobs_have_separate_outputs():
    script = (SCRIPTS / "run_learned_parallel.sh").read_text()
    assert "runs/learned_baselines/$model/$dataset" in script
    assert "CUDA_VISIBLE_DEVICES=$gpu" in script
