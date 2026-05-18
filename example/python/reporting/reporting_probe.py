from g2s import g2s


warning_result = g2s(
    "-a", "report_probe",
    "-mode", "warning",
    "-steps", 5,
    "-sleepMs", 120,
    "-showLogs",
)

print("warning probe", warning_result["job_id"], warning_result["status"])
print("artifacts", warning_result["artifacts"])

submitted = g2s(
    "-a", "report_probe",
    "-mode", "error",
    "-steps", 5,
    "-sleepMs", 120,
    "-submitOnly",
)

print("submitted error probe", submitted["job_id"])
g2s("-waitAndDownload", submitted["job_id"])
