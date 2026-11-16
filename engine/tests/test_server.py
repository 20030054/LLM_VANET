"""End-to-end protocol test with the tiny random model on CPU."""
import socket, threading, time
import pytest
from kvrsu.server import Controller, build_parser, serve_forever

MSG = """WINDOW 1 30.0 500 500 2e10 0.0 4 2
V 11 a 480 500 10 0 2e9 0.0
V 12 b 520 500 10 3.14 3e9 0.1
V 13 c 500 560 0 0 1e9 0.0
V 14 d 900 900 5 1 1e9 0.0
R 100 11 4e8 4000 1.0 0.02
R 101 14 2e8 2000 0.8 0.01
END
"""


@pytest.mark.parametrize("policy", ["greedy", "weighted", "sequential", "prefix_cache", "kvrsu"])
def test_round_trip(policy, tmp_path):
    args = build_parser().parse_args(["--policy", policy, "--model", "tiny-random", "--device", "cpu",
                                      "--port-file", str(tmp_path / "port"), "--log", str(tmp_path / "w.jsonl")])
    th = threading.Thread(target=lambda: serve_forever(Controller(args), "127.0.0.1", 0, args.port_file), daemon=True)
    th.start()
    for _ in range(200):
        if (tmp_path / "port").exists() and (tmp_path / "port").read_text():
            break
        time.sleep(0.05)
    s = socket.create_connection(("127.0.0.1", int((tmp_path / "port").read_text())))
    f = s.makefile("r")
    for _ in range(2):                                   # two windows: second one reuses the cache
        s.sendall(MSG.encode())
        lines = []
        while (ln := f.readline().strip()) != "END":
            lines.append(ln.split())
        assert lines[0][0] == "RESULT" and int(lines[0][1]) == 2
        d = {int(l[1]): int(l[2]) for l in lines[1:]}
        assert set(d) == {100, 101}
        assert d[100] in (-1, -2, 12, 13) and d[101] in (-1, -2)   # vehicle 14 has no neighbour in range
    s.sendall(b"BYE\n")
    th.join(5)
    assert len((tmp_path / "w.jsonl").read_text().splitlines()) == 2
