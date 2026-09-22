from ui.server import AppState


def test_active_jobs_only_returns_work_in_progress_and_copies_progress():
    state = AppState()
    state.jobs = {
        "later": {
            "id": "later", "status": "running", "started_at": "2026-09-21T10:01:00",
            "pitcher": "Pitcher B", "batter": "Batter B", "progress": ["running"],
        },
        "first": {
            "id": "first", "status": "queued", "started_at": "2026-09-21T10:00:00",
            "pitcher": "Pitcher A", "batter": "Batter A", "progress": [],
        },
        "done": {
            "id": "done", "status": "complete", "started_at": "2026-09-21T09:00:00",
            "pitcher": "Pitcher C", "batter": "Batter C", "progress": ["done"],
        },
    }

    rows = state.active_jobs()

    assert [row["id"] for row in rows] == ["first", "later"]
    rows[1]["progress"].append("browser mutation")
    assert state.jobs["later"]["progress"] == ["running"]
