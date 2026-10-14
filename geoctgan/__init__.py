from .config import Config
__all__ = ['Config', 'run_pipeline', 'run_sensitivity_pipeline', 'run_evaluation']

def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run_pipeline
    return _run_pipeline(*args, **kwargs)

def run_sensitivity_pipeline(*args, **kwargs):
    from .pipeline import run_sensitivity_pipeline as _run_sensitivity_pipeline
    return _run_sensitivity_pipeline(*args, **kwargs)

def run_evaluation(*args, **kwargs):
    from .evaluation import run_evaluation as _run_evaluation
    return _run_evaluation(*args, **kwargs)
