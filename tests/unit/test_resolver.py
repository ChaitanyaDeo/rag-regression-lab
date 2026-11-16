"""
tests/unit/test_resolver.py


"""

import sys
from pathlib import Path
import textwrap
import pytest


from ragregress.rag.resolver import (ResolverError, parse_target, load_target, params_model_for, resolve_entry)

from ragregress.rag.registry import PipelineParams, get_entry


def create_module(tmp_path, monkeypatch, request, name, source):
    source = textwrap.dedent(source).lstrip()

    module_file = tmp_path / f"{name}.py"
    module_file.write_text(source, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    sys.modules.pop(name, None)
    request.addfinalizer(lambda: sys.modules.pop(name, None))

    return name

def create_module(tmp_path, monkeypatch, request, name, source):
    (tmp_path / f"{name}.py").write_text(source)

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    sys.modules.pop(name, None)

    request.addfinalizer(lambda: sys.modules.pop(name, None))

    return name


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("pkg.mod:MyClass", ("pkg.mod", "MyClass")),
        ("my_package.module:Pipeline", ("my_package.module", "Pipeline")),
        ("_internal.mod2:Class_2", ("_internal.mod2", "Class_2")),
    ],
)
def test_parse_target_valid(target, expected):
    assert parse_target(target) == expected


@pytest.mark.parametrize(
    "target",
    [
        "",
        ":Class",
        "pkg.mod:",
        "pkg..mod:Class",
        "pkg.mod:Class:Other",
        "pkg/mod.py:Class",
        "pkg.mod:Class Name",
        "pkg.mod :Class",
    ],
)

def test_parse_target_invalid(target):
    with pytest.raises(ResolverError) as exc:
        parse_target(target)

    assert "expected module.path:ClassName" in str(exc.value)
    assert target in str(exc.value)


def test_load_target_success(tmp_path, monkeypatch, request):
    module = create_module(
        tmp_path,
        monkeypatch,
        request,
        "resolver_test_load",
        "class MyPipeline:\n"
        "    pass\n",
    )

    cls = load_target(f"{module}:MyPipeline")

    assert cls.__name__ == "MyPipeline"


def test_load_target_missing_module():
    with pytest.raises(ResolverError) as exc:
        load_target("missing_resolver_module:MyPipeline")

    assert "missing_resolver_module" in str(exc.value)
    assert "Run from the folder that contains it" in str(exc.value)


def test_load_target_missing_class(tmp_path, monkeypatch, request):
    module = create_module(
        tmp_path,
        monkeypatch,
        request,
        "resolver_test_missing_class",
        "class ExistingPipeline:\n    pass\n",
    )

    with pytest.raises(ResolverError) as exc:
        load_target(f"{module}:MissingPipeline")

    assert module in str(exc.value)
    assert "MissingPipeline" in str(exc.value)


def test_load_target_missing_dependency(tmp_path, monkeypatch, request):
    module = create_module(
        tmp_path,
        monkeypatch,
        request,
        "resolver_test_dependency",
        "import definitely_missing_resolver_dependency\n",
    )

    with pytest.raises(ResolverError) as exc:
        load_target(f"{module}:MyPipeline")

    assert module in str(exc.value)
    assert "definitely_missing_resolver_dependency" in str(exc.value)


def test_load_target_not_a_class(tmp_path, monkeypatch, request):
    module = create_module(
        tmp_path,
        monkeypatch,
        request,
        "resolver_test_not_class",
        "MyPipeline = 42\n",
    )

    with pytest.raises(ResolverError) as exc:
        load_target(f"{module}:MyPipeline")

    assert "int" in str(exc.value)



def test_params_model_from_constructor():
    class MyPipeline:
        def __init__(
            self,
            name: str,
            temperature: float = 0.5,
            enabled: bool = True,
        ):
            pass

    model = params_model_for(MyPipeline)

    assert issubclass(model, PipelineParams)

    assert model.model_fields["name"].annotation is str
    assert model.model_fields["temperature"].annotation is float
    assert model.model_fields["temperature"].default == 0.5
    assert model.model_fields["enabled"].default is True


def test_params_model_is_cached():
    class MyPipeline:
        def __init__(self, name: str):
            pass

    first = params_model_for(MyPipeline)
    second = params_model_for(MyPipeline)

    assert first is second


def test_params_model_uses_existing_params():
    class CustomParams(PipelineParams):
        name: str

    class MyPipeline:
        Params = CustomParams

        def __init__(self, name: str):
            pass

    assert params_model_for(MyPipeline) is CustomParams


def test_params_model_rejects_kwargs():
    class MyPipeline:
        def __init__(self, **kwargs):
            pass

    with pytest.raises(ResolverError, match="define a Params class"):
        params_model_for(MyPipeline)


def test_params_model_rejects_object_default():
    class MyPipeline:
        def __init__(self, client=object()):
            pass

    with pytest.raises(ResolverError, match="client"):
        params_model_for(MyPipeline)



def test_resolve_entry_custom_target(tmp_path, monkeypatch, request):
    module = create_module(
        tmp_path,
        monkeypatch,
        request,
        "resolver_test_factory",
        "class MyPipeline:\n"
        "    def __init__(self, name: str, count: int = 2):\n"
        "        self.name = name\n"
        "        self.count = count\n",
    )

    entry = resolve_entry(f"{module}:MyPipeline")

    params = entry.params_model(name="example", count=5)
    instance = entry.factory(params)

    assert instance.name == "example"
    assert instance.count == 5


def test_resolve_entry_builtin_rag():
    expected = get_entry("builtin_rag")

    actual = resolve_entry("builtin_rag")

    assert actual is expected