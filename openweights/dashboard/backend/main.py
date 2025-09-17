import os
from typing import Dict, List, Optional

from database import Database
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from models import (
    Job,
    JobWithRuns,
    Organization,
    OrganizationCreate,
    Run,
    RunWithJobAndWorker,
    Token,
    TokenCreate,
    Worker,
    WorkerWithRuns,
)

load_dotenv()
app = FastAPI()

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Vite dev server
        "http://localhost:8124",  # FastAPI dev server
        "http://localhost:4173",  # Vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db(authorization: str = Header(None)) -> Database:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")

    try:
        # Extract the token from the Authorization header
        # Format should be "Bearer <token>"
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Invalid authorization header format"
            )

        token = authorization.split(" ")[1]
        return Database(auth_token=token)
    except IndexError:
        raise HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )


# Organization endpoints
@app.post("/organizations/", response_model=Organization)
async def create_organization(
    org_data: OrganizationCreate, db: Database = Depends(get_db)
):
    try:
        return await db.create_organization(org_data)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/organizations/", response_model=List[Organization])
async def get_organizations(db: Database = Depends(get_db)):
    """Get list of organizations the current user has access to."""
    try:
        result = db.client.from_("organizations").select("*").execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/organizations/{organization_id}", response_model=Organization)
async def get_organization(organization_id: str, db: Database = Depends(get_db)):
    """Get details of a specific organization."""
    try:
        if not db.verify_organization_access(organization_id):
            raise HTTPException(
                status_code=403, detail="No access to this organization"
            )

        result = (
            db.client.from_("organizations")
            .select("*")
            .eq("id", organization_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Organization not found")

        return result.data
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/organizations/{organization_id}/secrets")
async def update_organization_secrets(
    organization_id: str, secrets: Dict[str, str], db: Database = Depends(get_db)
):
    """Update all organization secrets together."""
    try:
        success = await db.update_organization_secrets(organization_id, secrets)
        return {"status": "success" if success else "failed"}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/organizations/{organization_id}/jobs/", response_model=List[Job])
async def get_jobs(
    organization_id: str, status: Optional[str] = None, db: Database = Depends(get_db)
):
    try:
        return db.get_jobs(organization_id, status)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/organizations/{organization_id}/jobs/{job_id}", response_model=JobWithRuns)
async def get_job(organization_id: str, job_id: str, db: Database = Depends(get_db)):
    try:
        return db.get_job(organization_id, job_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/organizations/{organization_id}/jobs/{job_id}/cancel")
async def cancel_job(organization_id: str, job_id: str, db: Database = Depends(get_db)):
    try:
        return db.cancel_job(organization_id, job_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/organizations/{organization_id}/jobs/{job_id}/restart")
async def restart_job(
    organization_id: str, job_id: str, db: Database = Depends(get_db)
):
    try:
        return db.restart_job(organization_id, job_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/organizations/{organization_id}/runs/", response_model=List[Run])
async def get_runs(
    organization_id: str, status: Optional[str] = None, db: Database = Depends(get_db)
):
    try:
        return db.get_runs(organization_id, status)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get(
    "/organizations/{organization_id}/runs/{run_id}", response_model=RunWithJobAndWorker
)
async def get_run(organization_id: str, run_id: str, db: Database = Depends(get_db)):
    try:
        return db.get_run(organization_id, run_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/organizations/{organization_id}/runs/{run_id}/logs",
    response_class=PlainTextResponse,
)
async def get_run_logs(
    organization_id: str, run_id: str, db: Database = Depends(get_db)
):
    try:
        return db.get_log_file_content(organization_id, run_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/organizations/{organization_id}/runs/{run_id}/events")
async def get_run_events(
    organization_id: str, run_id: str, db: Database = Depends(get_db)
):
    try:
        return db.get_run_events(organization_id, run_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/organizations/{organization_id}/workers/", response_model=List[Worker])
async def get_workers(
    organization_id: str, status: Optional[str] = None, db: Database = Depends(get_db)
):
    try:
        return db.get_workers(organization_id, status)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get(
    "/organizations/{organization_id}/workers/{worker_id}",
    response_model=WorkerWithRuns,
)
async def get_worker(
    organization_id: str, worker_id: str, db: Database = Depends(get_db)
):
    try:
        return db.get_worker(organization_id, worker_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/organizations/{organization_id}/workers/{worker_id}/logs",
    response_class=PlainTextResponse,
)
async def get_worker_logs(
    organization_id: str, worker_id: str, db: Database = Depends(get_db)
):
    try:
        return db.get_worker_logs(organization_id, worker_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/organizations/{organization_id}/workers/{worker_id}/shutdown")
async def shutdown_worker(
    organization_id: str, worker_id: str, db: Database = Depends(get_db)
):
    try:
        return db.shutdown_worker(organization_id, worker_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/organizations/{organization_id}/files/{file_id}/content",
    response_class=PlainTextResponse,
)
async def get_file_content(
    organization_id: str, file_id: str, db: Database = Depends(get_db)
):
    try:
        # Print debug info
        print(f"Fetching content for file: {file_id}")
        content = db.get_file_content(organization_id, file_id)
        print(f"Content length: {len(content) if content else 0}")
        return content
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        print(f"Error fetching file content: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/organizations/{organization_id}/tokens", response_model=Token)
async def create_token(
    organization_id: str, token_data: TokenCreate, db: Database = Depends(get_db)
):
    try:
        return await db.create_token(organization_id, token_data)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/organizations/{organization_id}/tokens", response_model=List[Token])
async def list_tokens(organization_id: str, db: Database = Depends(get_db)):
    try:
        return await db.list_tokens(organization_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/organizations/{organization_id}/tokens/{token_id}")
async def delete_token(
    organization_id: str, token_id: str, db: Database = Depends(get_db)
):
    try:
        await db.delete_token(organization_id, token_id)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static file handling
if os.path.exists("static"):
    # Production mode - serve static files
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

    @app.get("/ow.svg")
    async def serve_ow():
        return FileResponse("static/ow.svg")

    @app.get("/vite.svg")
    async def serve_vite():
        return FileResponse("static/vite.svg")

    # Catch all other routes and serve index.html
    @app.get("/{full_path:path}")
    async def serve_app(full_path: str):
        # Only treat it as an API call if it starts with "organizations/" AND
        # is followed by known API endpoints
        api_endpoints = [
            r"organizations/$",  # List organizations
            r"organizations/[^/]+/jobs/?$",  # List jobs
            r"organizations/[^/]+/jobs/[^/]+/?$",  # Get job
            r"organizations/[^/]+/runs/?$",  # List runs
            r"organizations/[^/]+/runs/[^/]+/?$",  # Get run
            r"organizations/[^/]+/workers/?$",  # List workers
            r"organizations/[^/]+/workers/[^/]+/?$",  # Get worker
            r"organizations/[^/]+/tokens/?$",  # List tokens
            r"organizations/[^/]+/files/[^/]+/content/?$",  # Get file content
        ]

        import re

        is_api_call = any(re.match(pattern, full_path) for pattern in api_endpoints)

        if is_api_call:
            raise HTTPException(status_code=404, detail="API endpoint not found")

        return FileResponse("static/index.html")
