from g2s import g2s


def unpack_probe_result(result):
    if not isinstance(result, tuple):
        result = (result,)
    if len(result) < 2:
        raise RuntimeError(f"unexpected report_probe result shape: {result!r}")

    elapsed = result[0]
    meta = result[1]
    extra = result[2:]
    return elapsed, meta, extra


def run_warning_probe():
    elapsed, meta, extra = unpack_probe_result(g2s(
        "-a", "report_probe",
        "-mode", "warning",
        "-steps", 5,
        "-sleepMs", 120,
        "-showLogs",
        "-returnMeta",
    ))
    print("\nwarning probe completed")
    print("elapsed seconds:", float(elapsed))
    print("meta:", meta)
    if extra:
        print("extra return values:", extra)


def run_error_probe():
    job_id = g2s(
        "-a", "report_probe",
        "-mode", "error",
        "-steps", 5,
        "-sleepMs", 120,
        "-submitOnly",
    )
    print("\nerror probe submitted with job id:", int(job_id))
    g2s(
        "-waitAndDownload", job_id,
        "-showLogs",
        "-returnMeta",
    )


if __name__ == "__main__":
    run_warning_probe()
    run_error_probe()
