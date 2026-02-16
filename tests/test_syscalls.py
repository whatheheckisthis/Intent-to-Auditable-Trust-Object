from iato_ops.syscalls import collect_runtime_syscalls


def test_collect_runtime_syscalls_shape():
    data = collect_runtime_syscalls()

    assert isinstance(data["pid"], int)
    assert isinstance(data["ppid"], int)
    assert isinstance(data["cwd"], str)
    assert isinstance(data["monotonic_ns"], int)
    assert isinstance(data["time_ns"], int)

    uname = data["uname"]
    assert set(uname.keys()) == {"system", "release", "version", "machine"}
    assert all(isinstance(uname[key], str) for key in uname)
