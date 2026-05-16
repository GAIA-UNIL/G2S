from g2s import g2s


def run_warning_probe():
    result = g2s(
        "-a", "report_probe",
        "-mode", "warning",
        "-steps", 5,
        "-sleepMs", 120,
        "-showLogs",
        "-returnFormat", "schema",
    )
    print("\nwarning probe completed")
    print("elapsed seconds:", float(result["time"]))
    print("result:", result)


def run_error_probe():
    submit_result = g2s(
        "-a", "report_probe",
        "-mode", "error",
        "-steps", 5,
        "-sleepMs", 120,
        "-submitOnly",
        "-returnFormat", "schema",
    )
    job_id = submit_result["job_id"]
    print("\nerror probe submitted with job id:", int(job_id))
    g2s(
        "-waitAndDownload", job_id,
        "-showLogs",
        "-returnFormat", "schema",
    )


if __name__ == "__main__":
    run_warning_probe()
    run_error_probe()
