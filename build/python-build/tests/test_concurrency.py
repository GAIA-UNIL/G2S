"""Exercise the installed native client against a concurrent ZeroMQ server.

Run directly with each supported interpreter; no simulation executable or
publication credentials are needed. Gates require every client's request to
arrive before replying to any client, so serialized network I/O fails the test.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import struct
import sys
import sysconfig
import threading
import unittest
import warnings

import numpy as np
import zmq
import g2s


def integer(value):
    return struct.pack("=i", value)


class ProtocolServer:
    def __init__(self, gate=None, clients=4, fail=False, warn=False):
        self.gate = gate
        self.clients = clients
        self.fail = fail
        self.warn = warn
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.gate_passed = False
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("Protocol server did not start")
        if self.error:
            raise self.error
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(5)
        if self.error:
            raise self.error
        if self.thread.is_alive():
            raise RuntimeError("Protocol server did not stop")

    def serve(self):
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.linger = 0
        try:
            self.port = socket.bind_to_random_port("tcp://127.0.0.1")
            self.ready.set()
            uploads, jobs, pending, seen = {}, {}, [], set()
            while not self.stop.is_set():
                if not socket.poll(20):
                    continue
                route, empty, message = socket.recv_multipart()
                version, task = struct.unpack_from("=ii", message)
                assert version == 1
                name = message[8:72].split(b"\0", 1)[0].decode(errors="replace")
                if task == 11:  # SERVER_STATUS
                    reply = integer(1)
                elif task == 14:  # SERVER_VERSION
                    reply = b"1.0.1"
                elif task == 4:  # EXIST
                    if (name.startswith("meta_") or (self.fail and name.startswith("error_"))
                            or (self.warn and name.startswith("warning_"))):
                        reply = integer(3)  # text artifact
                    else:
                        present = name in uploads or name.startswith(("im_1_", "im_2_"))
                        reply = integer(int(present))
                elif task == 2:  # UPLOAD
                    uploads[name] = message[72:]
                    reply = integer(0)
                elif task == 1:  # JOB
                    job = json.loads(message[8:])
                    job_id = len(jobs) + 1
                    jobs[job_id] = job
                    reply = struct.pack("=I", job_id)
                elif task in (5, 6, 7, 12):
                    # PROGRESSION, DURATION, KILL, JOB_STATUS
                    reply = integer({5: 100000, 6: 10, 7: 0, 12: int(self.fail)}[task])
                elif task == 3:  # DOWNLOAD: echo each job's own training image
                    job_id = int(name.rsplit("_", 1)[1])
                    reply = uploads[jobs[job_id]["Parameter"]["-ti"][0]]
                elif task == 13:  # DOWNLOAD_TEXT
                    if name.startswith("error_"):
                        reply = b"error_message=deliberate test failure\n\0"
                    elif name.startswith("warning_"):
                        reply = b"message=deliberate test warning\n"
                    elif name.startswith("meta_"):
                        reply = (b"status=finished\nduration_ms=10\n"
                                 b"result_output_1_name=simulation\n"
                                 b"result_output_2_name=indexmap\n")
                    else:
                        reply = b""
                else:
                    raise AssertionError("Unexpected protocol task: %s" % task)

                if task == self.gate and not self.gate_passed and route not in seen:
                    seen.add(route)
                    pending.append((route, empty, reply))
                    if len(pending) < self.clients:
                        continue
                    self.gate_passed = True
                    for response in pending:
                        socket.send_multipart(list(response))
                    pending.clear()
                else:
                    socket.send_multipart([route, empty, reply])
        except BaseException as error:
            self.error = error
            self.ready.set()
        finally:
            socket.close()
            context.term()


class ConcurrencyTests(unittest.TestCase):
    def test_import_preserves_free_threading(self):
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            self.assertFalse(sys._is_gil_enabled(), "Importing G2S enabled the GIL")

    def test_simultaneous_network_calls(self):
        shared_types = [0.0]
        # Gate status, binary uploads, job submission, and result downloads.
        for task in (11, 2, 1, 3):
            with self.subTest(task=task), ProtocolServer(gate=task) as server:
                start = threading.Barrier(server.clients)

                def call(index):
                    source = np.arange(24, dtype=np.float32).reshape(4, 6) + index * 100
                    start.wait(timeout=5)
                    result = g2s.run("-sa", "127.0.0.1", "-p", server.port,
                                     "-TO", 3000, "-silent", a="echo",
                                     ti=[source], dt=shared_types, s=index)
                    np.testing.assert_array_equal(result["simulation"], source)
                    np.testing.assert_array_equal(result["indexmap"], source)
                    self.assertEqual(result["status"], "finished")
                    return result["job_id"]

                with ThreadPoolExecutor(max_workers=server.clients) as executor:
                    ids = list(executor.map(call, range(server.clients)))
                self.assertEqual(len(set(ids)), server.clients)
                self.assertTrue(server.gate_passed)
        self.assertEqual(shared_types, [0.0])

    def test_control_async_and_legacy(self):
        with ProtocolServer() as server:
            options = ("-sa", "127.0.0.1", "-p", server.port, "-TO", 3000, "-silent")
            self.assertEqual(g2s.run(*options, "-serverVersion"), "1.0.1")
            self.assertEqual(g2s.run(*options, "-serverStatus"), 1)
            source = np.arange(12, dtype=np.float32).reshape(3, 4)
            submitted = g2s.run(*options, "-submitOnly", a="echo", ti=source, dt=[0])
            job_id = submitted["job_id"]
            self.assertEqual(submitted["status"], "finished")
            status = g2s.run(*options, "-statusOnly", job_id)
            self.assertEqual(status["progress"], 100)
            result = g2s.run(*options, "-waitAndDownload", job_id)
            np.testing.assert_array_equal(result["simulation"], source)
            legacy = g2s.run(*options, "-legacy_output", "-waitAndDownload", job_id)
            np.testing.assert_array_equal(legacy[0], source)
            self.assertEqual(g2s.run(*options, "-kill", job_id)["job_id"], job_id)

    def test_worker_errors_restore_thread_state(self):
        with ProtocolServer(fail=True) as server:
            def call(index):
                with self.assertRaisesRegex(Exception, "deliberate test failure"):
                    g2s.run("-sa", "127.0.0.1", "-p", server.port, "-TO", 3000,
                            "-silent", a="echo", ti=np.ones((2, 2)) * index, dt=[0])
                # A failed detached polling operation must leave Python usable.
                return g2s.run("-sa", "127.0.0.1", "-p", server.port, "-serverVersion")

            with ThreadPoolExecutor(max_workers=4) as executor:
                self.assertEqual(list(executor.map(call, range(4))), ["1.0.1"] * 4)

    def test_warning_filters_and_polling(self):
        with ProtocolServer(warn=True) as server:
            options = ("-sa", "127.0.0.1", "-p", server.port, "-TO", 3000,
                       "-showLogs", "-a", "echo", "-ti", np.ones((2, 2)), "-dt", [0])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = g2s.run(*options)
                self.assertEqual(result["status"], "finished")
                self.assertTrue(any("deliberate test warning" in str(w.message) for w in caught))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with self.assertRaisesRegex(Warning, "deliberate test warning"):
                    g2s.run(*options)
            self.assertEqual(g2s.run("-sa", "127.0.0.1", "-p", server.port,
                                     "-serverVersion"), "1.0.1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
