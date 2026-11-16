"""
ragregress/rag/resolver.py

"""

import importlib
import inspect
import sys
import pathlib
import re
from typing import Tuple, Any
import os

from pydantic import create_model

from ragregress.rag.registry import PipelineParams, PipelineEntry, get_entry
from ragregress.core.errors import UnknownPipelineError

import logging

logger = logging.getLogger(__name__)  

_params_model_cache = {}

class ResolverError(ValueError):
    """Raised when resolver configuration or resolution fails."""



def parse_target(target: str)-> Tuple[str, str]:
    """Parse a module.path:ClassName target."""

    _TARGET_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")

    if not isinstance(target, str) or not _TARGET_PATTERN.fullmatch(target):
        raise ResolverError(f"expected module.path:ClassName, got {target!r}")

    module_name, class_name = target.split(":")

    return module_name, class_name



def load_target(target:str)->type:
    """Import and return the class specified by target."""

    module_name, class_name = parse_target(target)

    cwd = os.getcwd()

    try:
        if cwd not in sys.path:
            sys.path.append(cwd)
        module = importlib.import_module(module_name)

    except ModuleNotFoundError as e:
        logger.error("ModuleNotFoundError: %s", module_name)
        missing = e.name or ""

        if missing == module_name or module_name.startswith(missing + "."):
            raise ResolverError(f"Target module '{module_name}' not found. Run from the folder that contains it.") from e

        raise ResolverError(f"Target module '{module_name}' imports missing dependency '{missing}'.") from e

    except Exception as e:
        raise ResolverError(f"Error importing '{module_name}': {type(e).__name__}: {e}") from e

    if not hasattr(module, class_name):
        raise ResolverError(f"Module '{module_name}' has no attribute '{class_name}'.")

    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise ResolverError(f"'{module_name}:{class_name}' is {type(cls).__name__}, not a class.")

    return cls


def params_model_for(cls: type) -> type[PipelineParams]:
    """
    Create or retrieve the params model for a pipeline class.
    """

    if cls in _params_model_cache:
        return _params_model_cache[cls]

    existing = getattr(cls, "Params", None)

    if isinstance(existing, type) and issubclass(existing, PipelineParams):
        _params_model_cache[cls] = existing
        return existing

    try:
        signature = inspect.signature(cls.__init__)

    except (TypeError, ValueError) as e:
        raise ResolverError(f"Cannot inspect {cls.__name__}; define a Params class.") from e

    fields = {}

    for name, param in signature.parameters.items():
        if name=="self":
            continue

        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY,):
            raise ResolverError(f"Unsupported constructor signature for '{cls.__name__}'; define a Params class")

        annotation = (param.annotation if param.annotation is not inspect.Parameter.empty else Any)


        default = param.default

        if default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)

        else:
            if not isinstance(default, (str, int, float, bool, type(None), list, dict)):
                raise ResolverError(f"Parameter '{name}' has a non-JSON-compatible default; define a Params class.")

            fields[name] = (annotation, default)


    model = create_model(f"{cls.__name__}Params", __base__=PipelineParams, **fields)

    _params_model_cache[cls] = model

    return model

def resolve_entry(target:str) -> PipelineEntry:
    """Resolve a registered pipeline ID or custom target."""

    try:
        return get_entry(target)
    except UnknownPipelineError:
        pass

    cls = load_target(target)
    params_model = params_model_for(cls)

    def factory(params: PipelineParams):
        if not isinstance(params, params_model):
            raise ResolverError(f"Expected params of type {params_model.__name__}.")

        values = {name: getattr(params, name) for name in params_model.model_fields}

        return cls(**values)


    return PipelineEntry(params_model=params_model, factory=factory)
