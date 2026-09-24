"""
Small local service that wraps recitation_checker.py: given a
recitation_id, downloads the submitted audio from Supabase Storage,
runs auto-detection, and inserts the results into is_recitation_errors
(source='auto') — the same table and columns the app's manual pinning
UI already reads and displays.

Uses the Supabase SERVICE ROLE key, which bypasses RLS. That key is a
real secret: it lives only in a local .env file or Railway's service
variables (never in this repo, never pasted into chat).

Scope: this checks recitation against Hafs 'an Asim only — the model
and reference text are both Hafs. Other riwayat (Warsh, Qaloon, etc.)
are not supported and will produce unreliable results.

A full check (word-accuracy Whisper pass + acoustic tajweed pass) takes
2-3+ minutes on real audio, too long to hold one HTTP request open
reliably — something in the network path (proxy, mobile network, or
the browser) was observed silently killing the connection well before
completion even though the server kept computing the whole time.
POST /check-recitation now kicks off the work in a background thread
and returns almost immediately; the frontend polls the
is_auto_check_jobs table (via Supabase directly, not this service) for
the result. See islamic_studies_auto_check_jobs_migration.sql.

Run locally:
    py -3.11 -m uvicorn service:app --reload --port 8090

Call:
    POST http://localhost:8090/check-recitation
    Authorization: Bearer <facilitator's Supabase session access_token>
    { "recitation_id": "<uuid from is_recitations>" }
"""

import os
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

from recitation_checker import check_recitation
from tajweed_checker import check_tajweed

# Load .env from next to this file, not from the process's current
# working directory — otherwise this silently finds nothing when
# launched from anywhere else (e.g. `uvicorn service:app --app-dir ...`
# run from a different folder). In production (Railway) these come from
# real environment variables instead, and this call is a harmless no-op
# since there's no .env file there.
load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY are not set. "
        "Fill in recitation-checker/.env locally (see .env.example), or set "
        "them as service variables on the Railway deployment."
    )
STORAGE_BUCKET = "quran-recitations"

# Public web origins allowed to call this service directly from the
# browser. Kept as an allowlist (not "*") since this endpoint bypasses
# RLS via the service role key.
ALLOWED_ORIGINS = [
    "https://app.tarbee.ai",
    "https://islamic.holistify.ai",
    "https://holistify-islamic-studies.vercel.app",
    "http://localhost:3001",  # local dev via proxy.js
]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
app = FastAPI(title="Holistify Recitation Auto-Checker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


class CheckRequest(BaseModel):
    recitation_id: str
    # Optional list of is_error_types.id — when set (non-empty),
    # restricts detection to just those error type(s) instead of
    # everything the facilitator has defined, so e.g. a facilitator who
    # only cares about madd length this pass isn't shown unrelated
    # word-accuracy/makhraj results too. Empty/omitted means all types.
    error_type_ids: list[str] | None = None


def require_facilitator(authorization: str = Header(None)):
    """This endpoint uses the Supabase SERVICE ROLE key internally (it
    bypasses RLS), so unlike the rest of the app it can't rely on RLS to
    keep students from checking recitations or triggering expensive
    Whisper inference on demand. Instead it verifies the caller's own
    Supabase session token (the same one the browser already holds after
    login) and checks their is_profiles.role, mirroring the
    is_facilitator() Postgres check the rest of the app's RLS relies on."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        user = supabase.auth.get_user(token).user
    except Exception:
        raise HTTPException(401, "Invalid or expired session token")
    if not user:
        raise HTTPException(401, "Invalid or expired session token")
    profile = supabase.table("is_profiles").select("role,school_id").eq("id", user.id).maybe_single().execute()
    data = profile.data or {}
    role = data.get("role")
    if role not in ("facilitator", "admin"):
        raise HTTPException(403, "Facilitator or admin access required")
    # Returned as a plain dict (not attached to `user`, a gotrue model
    # that may not accept arbitrary extra attributes) so check() below
    # can log AI usage without a second profile lookup.
    return {"user": user, "school_id": data.get("school_id")}


def _run_check_job(recitation_id: str, error_type_ids: list[str] | None, school_id, user_id):
    """Does the actual (slow) detection work in a background thread and
    writes the outcome to is_auto_check_jobs instead of returning it —
    the frontend polls that table directly via Supabase. Never raises;
    any failure is recorded as a job row with status='error' so the
    frontend can show the real reason instead of a generic timeout."""
    tmp_path = None
    try:
        rec = supabase.table("is_recitations").select("*").eq("id", recitation_id).maybe_single().execute()
        row = rec.data
        if not row:
            raise RuntimeError("Recitation not found")

        audio_bytes = supabase.storage.from_(STORAGE_BUCKET).download(row["audio_path"])
        suffix = os.path.splitext(row["audio_path"])[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Scope auto-check to whatever the facilitator has actually
        # defined in the Error Types tab, rather than always reporting a
        # fixed builtin taxonomy: check_recitation only ever produces
        # one of six fixed (category, label) pairs, and any of them
        # missing here (e.g. deleted or renamed) is dropped from the
        # results instead of surfacing a label the facilitator doesn't
        # recognise. Fetched fresh per request since a facilitator can
        # edit these at any time.
        error_types = supabase.table("is_error_types").select("id,label").execute()
        allowed_error_types = {t["label"]: t["id"] for t in (error_types.data or [])}
        if error_type_ids:
            wanted = set(error_type_ids)
            allowed_error_types = {
                label: id_ for label, id_ in allowed_error_types.items() if id_ in wanted
            }
            if not allowed_error_types:
                raise RuntimeError("None of the selected error type(s) exist anymore.")

        transcription, duration, errors = check_recitation(
            tmp_path, row["surah_number"], row["ayah_start"], row["ayah_end"],
            assessment_type=row.get("assessment_type") or "recitation",
            allowed_error_types=allowed_error_types,
        )
        # Separate model, separate pipeline (acoustic madd/elongation
        # checking - see tajweed_checker.py's module docstring for why
        # this only checks madd length, not qalqalah/ghunnah/idgham/
        # iqlab). Runs alongside the word-accuracy check above; failures
        # here shouldn't take down word-accuracy results the facilitator
        # is already waiting on.
        try:
            errors += check_tajweed(
                tmp_path, row["surah_number"], row["ayah_start"], row["ayah_end"],
                allowed_error_types=allowed_error_types,
            )
        except Exception as exc:
            print(f"Tajweed check failed (word-accuracy results still returned): {exc}")

        for e in errors:
            e["recitation_id"] = row["id"]
            e["student_id"] = row["student_id"]
            e["surah_number"] = row["surah_number"]
            e["created_by"] = None  # auto-detected, no facilitator author

        try:
            supabase.table("is_ai_usage_log").insert({
                "school_id": school_id,
                "user_id": user_id,
                "feature": "auto_check",
                "source": "auto_checker",
            }).execute()
        except Exception as exc:
            print(f"is_ai_usage_log insert failed (non-fatal): {exc}")

        # Not persisted to is_recitation_errors here. Results are stored
        # for client-side preview only — the facilitator must explicitly
        # click "Save" to write them there. This means an abandoned/
        # unsaved check never needs cleanup: nothing is written to that
        # table unless confirmed.
        result = {
            "recitation_id": row["id"],
            "riwaya": "hafs",  # only riwaya this model/reference text supports
            "assessment_type": row.get("assessment_type") or "recitation",
            "transcription": transcription,
            "duration_seconds": duration,
            "error_count": len(errors),
            "errors": errors,
        }
        supabase.table("is_auto_check_jobs").update({
            "status": "done",
            "result": result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", recitation_id).execute()
    except Exception as exc:
        traceback.print_exc()
        try:
            supabase.table("is_auto_check_jobs").update({
                "status": "error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", recitation_id).execute()
        except Exception as exc2:
            print(f"Could not even record the job failure: {exc2}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/check-recitation")
def check(req: CheckRequest, _caller=Depends(require_facilitator)):
    rec = (
        supabase.table("is_recitations")
        .select("id,student_id,student:is_profiles!student_id(school_id)")
        .eq("id", req.recitation_id)
        .maybe_single()
        .execute()
    )
    row = rec.data
    if not row:
        raise HTTPException(404, "Recitation not found")

    # This uses the service role, which bypasses RLS entirely — unlike
    # the rest of the app, nothing stops this endpoint from reaching
    # into another school's data unless it's checked explicitly here.
    # Without this, a facilitator/admin from School A who has (or
    # guesses) a School B recitation_id could trigger detection on
    # School B's private audio and overwrite School B's job row.
    student_school_id = (row.get("student") or {}).get("school_id")
    if student_school_id != _caller.get("school_id"):
        raise HTTPException(403, "That recitation belongs to a different school.")

    # Upserting here (rather than insert) overwrites any previous
    # result for this recitation, so re-running always reflects the
    # latest attempt.
    supabase.table("is_auto_check_jobs").upsert({
        "id": row["id"],
        "student_id": row["student_id"],
        "status": "processing",
        "result": None,
        "error_message": None,
        "requested_by": _caller["user"].id,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }).execute()

    threading.Thread(
        target=_run_check_job,
        args=(req.recitation_id, req.error_type_ids, _caller.get("school_id"), _caller["user"].id),
        daemon=True,
    ).start()

    return {"status": "processing", "recitation_id": row["id"]}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8090)))
