## Summary

<!-- What changed and why. Decisions go here, not in a docs file. -->

Closes #

## Test evidence

```
<!-- paste the tail of `make check` -->
```

## Definition of Done

- [ ] CI green (`ci / check`).
- [ ] Every spec test ID for the changed behaviour exists in a test name or docstring and passes; new behaviour has new IDs.
- [ ] Core coverage gate holds; integration/E2E with fakes cover any new FSM row.
- [ ] No new config key without: `Settings` field, `.env.example` line, validation rule.
- [ ] No new event name, error code or wire field outside `src/sevi/contracts/` (otherwise this is a `contract-change` PR with a `CONTRACTS_VERSION` bump and updated schema snapshot).
- [ ] Logs contain no PII (redaction tests pass).
- [ ] Kill-switch default set and shown on `/op/api/readyz` for any new external dependency.
- [ ] Public text only through `src/sevi/copy/copy.yaml` + `scripts/select_copy.py`.
- [ ] **No real API calls in tests**: Gnani/Twilio are faked; anything live is `@pytest.mark.live` + `@pytest.mark.max_credits(n)` and skipped unless `SEVI_ALLOW_LIVE=1`.
- [ ] **Credit-safe**: every Gnani call goes through `adapters/ledger.py`, which refuses past `SEVI_LEDGER__BUDGET_CREDITS_PER_KEY`.
- [ ] No secrets committed; values only via `infisical run`.
- [ ] Zero spend: no dependency, service or account that costs money.
- [ ] Code only: no specs, research, ADRs or logs added to the repo.
- [ ] Reviewed and merged by the other person (never self-merge); commits from the author's own account.
