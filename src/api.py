from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from run_employee_report import build_employee_reports

app = FastAPI(title="Burnout Risk Dashboard API")

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache reports in memory so we don't retrain the model on every request
_cached_reports = None


@app.on_event("startup")
def startup_event():
    global _cached_reports
    print("[api] Training model and building employee reports on startup...")
    _cached_reports = build_employee_reports(n=20)  # adjust n as needed
    print(f"[api] Ready. {len(_cached_reports)} employee reports cached.")


@app.get("/")
def root():
    return {"status": "ok", "message": "Burnout Risk Dashboard API is running"}


@app.get("/api/employees")
def get_employees():
    return {"count": len(_cached_reports), "employees": _cached_reports}


@app.get("/api/employees/{employee_id}")
def get_employee(employee_id: int):
    for emp in _cached_reports:
        if emp["employee_id"] == employee_id:
            return emp
    return {"error": "Employee not found"}, 404