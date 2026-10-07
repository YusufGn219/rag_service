from rag_service.version import code_version


def _package(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_same_files_give_the_same_version(tmp_path):
    pkg = _package(tmp_path, {"a.py": "x = 1\n", "b.py": "y = 2\n"})
    assert code_version(pkg) == code_version(pkg)
    assert len(code_version(pkg)) == 10


def test_changing_a_file_changes_the_version(tmp_path):
    pkg = _package(tmp_path, {"a.py": "x = 1\n"})
    before = code_version(pkg)
    (pkg / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_version(pkg) != before


def test_adding_removing_or_renaming_a_module_changes_the_version(tmp_path):
    pkg = _package(tmp_path, {"a.py": "x = 1\n"})
    base = code_version(pkg)
    (pkg / "b.py").write_text("y = 2\n", encoding="utf-8")
    added = code_version(pkg)
    assert added != base
    (pkg / "b.py").rename(pkg / "c.py")
    assert code_version(pkg) != added
    (pkg / "c.py").unlink()
    assert code_version(pkg) == base


def test_only_the_modules_count_not_other_files_or_folders(tmp_path):
    pkg = _package(tmp_path, {"a.py": "x = 1\n"})
    before = code_version(pkg)
    (pkg / "notes.txt").write_text("hello", encoding="utf-8")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\x00\x01")
    (pkg / "sub").mkdir()
    (pkg / "sub" / "z.py").write_text("z = 3\n", encoding="utf-8")
    assert code_version(pkg) == before


def test_touching_a_file_without_changing_it_keeps_the_version(tmp_path):
    import os

    pkg = _package(tmp_path, {"a.py": "x = 1\n"})
    before = code_version(pkg)
    os.utime(pkg / "a.py", (1, 1))
    assert code_version(pkg) == before


def test_the_real_package_has_a_version():
    assert len(code_version()) == 10
