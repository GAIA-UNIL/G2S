from g2s import g2s


def run_warning_probe():
    _schema_result = g2s(
        "-a", "report_probe",
        "-mode", "warning",
        "-steps", 5,
        "-sleepMs", 120,
        "-showLogs")
    elapsed = _schema_result.get("time")
    print("\nwarning probe completed")
    print("elapsed seconds:", float(elapsed))


def run_error_probe():
    _schema_result = g2s(
        "-a", "report_probe",
        "-mode", "error",
        "-steps", 5,
        "-sleepMs", 120,
        "-submitOnly")
    job_id = _schema_result["job_id"]
    print("\nerror probe submitted with job id:", int(job_id))
    g2s(
        "-waitAndDownload", job_id,
        "-showLogs")


if __name__ == "__main__":
    run_warning_probe()
    run_error_probe()
