from g2s import g2s


def run_warning_probe():
    elapsed, *_ = g2s(
        "-a", "report_probe",
        "-mode", "warning",
        "-steps", 5,
        "-sleepMs", 120,
        "-showLogs", '-legacy_output')
    print("\nwarning probe completed")
    print("elapsed seconds:", float(elapsed))


def run_error_probe():
    job_id = g2s(
        "-a", "report_probe",
        "-mode", "error",
        "-steps", 5,
        "-sleepMs", 120,
        "-submitOnly", '-legacy_output')
    print("\nerror probe submitted with job id:", int(job_id))
    g2s(
        "-waitAndDownload", job_id,
        "-showLogs", '-legacy_output')


if __name__ == "__main__":
    run_warning_probe()
    run_error_probe()
