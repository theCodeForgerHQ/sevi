"""`GET /api/v1/report` (contracts §9.1; B spec §9.4, §9.6).

`Authorization: Bearer <22-char token>` else 401. Lookup by sha256: unknown -> 410
`REPORT_EXPIRED` (deleted and never-existed are indistinguishable by design); past
`delete_after` (created_at + 72 h) -> 410; no decision yet -> 404; REPORT not among the
decision's surfaces -> 404. Else `ReportResponse` with code-selected wording only
(`evon_text` is always None: REPORT_PROSE is pre-cut).
"""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from typing import Final

from fastapi import APIRouter, Request

from sevi.app.admission import http_error
from sevi.app.routes import bearer, state_of
from sevi.app.state import AppState
from sevi.contracts.enums import PresentationKind, Surface
from sevi.contracts.errors import ErrorCode
from sevi.contracts.session import SessionRecord
from sevi.contracts.wire import ReportResponse, ReportTrial

router = APIRouter(prefix="/api/v1")

REPORT_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9_-]{22}$")
REPORT_TTL: Final = timedelta(hours=72)


def build_report(app: AppState, rec: SessionRecord) -> ReportResponse:
    d = rec.decision
    assert d is not None
    last: dict[int, ReportTrial] = {}
    for t in rec.trials:
        if t.spec.kind is PresentationKind.SCORED and t.spec.scored_index is not None:
            last[t.spec.scored_index] = ReportTrial(
                scored_index=t.spec.scored_index,
                snr_db=float(t.spec.snr_db),
                correct=t.result.triplet_correct,
            )
    return ReportResponse(
        lang=rec.lang,
        channel=rec.channel,
        mode=rec.mode_effective,
        transducer=rec.transducer,
        created_at=rec.created_at,
        expires_at=rec.created_at + REPORT_TTL,
        band=d.band,
        rule=d.rule,
        wording_class=d.wording_class,
        page_text=app.classifier.wording(d, rec.lang, Surface.REPORT).text,
        srt_display=d.srt_display,
        n_norm=d.n_norm,
        ages=d.ages,
        cutoff_version=d.cutoff_version,
        bank_version=rec.bank_version,
        evon_text=None,
        disclaimer=app.wording.text("disclaimer.page", rec.lang),
        trials=tuple(last[i] for i in sorted(last)),
    )


@router.get("/report", response_model=ReportResponse)
async def report(request: Request) -> ReportResponse:
    app = state_of(request)
    tok = bearer(request)
    if tok is None or not REPORT_TOKEN_RE.match(tok):
        raise http_error(ErrorCode.UNAUTHORIZED, "report token required")
    rec = await app.store.get_by_report_token(hashlib.sha256(tok.encode()).hexdigest())
    if rec is None:
        raise http_error(ErrorCode.REPORT_EXPIRED, "report expired")
    now = app.clock.now_utc()
    expires = rec.delete_after or rec.created_at + REPORT_TTL
    if now > min(expires, rec.created_at + REPORT_TTL):
        raise http_error(ErrorCode.REPORT_EXPIRED, "report expired")
    if rec.decision is None or Surface.REPORT not in rec.decision.surfaces:
        raise http_error(ErrorCode.NOT_FOUND, "no report for this session")
    return build_report(app, rec)
