import hashlib
import json
import logging
import os
import socket
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger('MarkSec')


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_dashboard_payload(config: Any) -> dict[str, Any]:
    value = _config_value(config, 'dashboard')
    if isinstance(value, dict):
        return dict(value)

    extras = _config_value(config, 'model_extra')
    if isinstance(extras, dict):
        maybe_dashboard = extras.get('dashboard')
        if isinstance(maybe_dashboard, dict):
            return dict(maybe_dashboard)
    return {}


def get_dashboard_job_kind(config: Any) -> str:
    payload = get_dashboard_payload(config)
    raw = str(payload.get('job_kind') or 'primary').strip().lower()
    if raw in {'primary', 'quality_backfill'}:
        return raw
    return 'primary'


def _project_root_from_config(config: Any) -> Path:
    raw = _config_value(config, 'project_root', '.') or '.'
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def resolve_sqlite_db_path(config: Any) -> Path | None:
    db_cfg = _config_value(config, 'database', {}) or {}
    if not isinstance(db_cfg, dict) or not db_cfg or not db_cfg.get('enabled'):
        return None

    try:
        if db_cfg.get('connection'):
            conn_info = db_cfg['connection']
        else:
            config_path = str(db_cfg.get('config_path', 'configs/db/db_config.json'))
            path = Path(config_path).expanduser()
            if not path.is_absolute():
                path = _project_root_from_config(config) / path
            with path.open('r', encoding='utf-8') as f:
                if path.suffix.lower() in {'.yaml', '.yml'}:
                    conn_info = yaml.safe_load(f) or {}
                else:
                    conn_info = json.load(f) or {}

        backend = str(conn_info.get('backend', 'sqlite')).strip().lower()
        if backend not in {'', 'sqlite'}:
            logger.warning(
                "Run registry only supports sqlite backend. Got backend='%s'. Skipping.",
                backend,
            )
            return None

        db_name = conn_info.get('database') or conn_info.get('path') or 'output/marksec.db'
        db_path = Path(str(db_name)).expanduser()
        if not db_path.is_absolute():
            db_path = _project_root_from_config(config) / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path
    except Exception:
        logger.warning('Failed to resolve sqlite db path for run registry.', exc_info=True)
        return None


def _normalize_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _extract_result_timestamp(path_text: str | None) -> str | None:
    if not path_text:
        return None
    candidate = Path(path_text).name
    if len(candidate) == 15 and candidate[8] == '-' and candidate.replace('-', '').isdigit():
        return candidate
    return None


def resolve_run_binding(config: Any, run_id: str | None = None) -> dict[str, Any] | None:
    requested_run_id = str(run_id or get_dashboard_payload(config).get('run_id') or '').strip()
    if not requested_run_id:
        return None

    db_path = resolve_sqlite_db_path(config)
    if db_path is None:
        return None

    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                r.run_id,
                r.exp_id,
                r.attempt_index,
                r.output_dir,
                r.data_path,
                r.summary_report_path,
                r.raw_results_path,
                r.quality_metric_details_path,
                r.result_path,
                g.template_id
            FROM experiment_run r
            LEFT JOIN experiment_group g ON g.exp_id = r.exp_id
            WHERE r.run_id = ? AND COALESCE(r.is_deleted, 0) = 0
            """,
            (requested_run_id,),
        ).fetchone()

    if row is None:
        return None

    output_dir = _normalize_optional_path(row['output_dir'])
    if output_dir is None:
        for key in (
            'data_path',
            'summary_report_path',
            'raw_results_path',
            'quality_metric_details_path',
            'result_path',
        ):
            candidate = _normalize_optional_path(row[key])
            if candidate is None:
                continue
            candidate_path = Path(candidate)
            output_dir = str(candidate_path.parent if candidate_path.suffix else candidate_path)
            break

    data_path = _normalize_optional_path(row['data_path'])
    if data_path is None and output_dir is not None:
        data_path = str((Path(output_dir) / 'generated_samples.jsonl').resolve())

    return {
        'run_id': str(row['run_id']),
        'exp_id': str(row['exp_id']) if row['exp_id'] is not None else None,
        'template_id': str(row['template_id']) if row['template_id'] is not None else None,
        'attempt_index': row['attempt_index'],
        'output_dir': output_dir,
        'data_path': data_path,
        'timestamp': _extract_result_timestamp(output_dir),
    }


class RunRegistry:
    """
    Registers experiment runs into sqlite early (at pipeline startup) and keeps
    run status/path metadata aligned through completion.
    """

    def __init__(self, config: Any, context: Any):
        self.config = config
        self.context = context

        self.spec_id: str | None = None
        self.run_id: str | None = None
        self.template_id: str | None = None
        self._registered = False

        self._db_path = self._resolve_sqlite_db_path()

    def is_enabled(self) -> bool:
        return self._db_path is not None

    def register_run_start(self) -> None:
        if not self.is_enabled():
            return

        now = _utcnow_iso()
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)

                dashboard = self._dashboard_payload()
                if dashboard.get('exp_id') and dashboard.get('run_id'):
                    self._register_scheduler_run(conn, now=now, dashboard=dashboard)
                else:
                    self._register_cli_run(conn, now=now)

                self._refresh_group_status_summary(conn, now=now)
                conn.commit()
                self._registered = True
        except Exception:
            logger.warning('Run registry startup registration failed.', exc_info=True)

    def finalize_run(self, *, success: bool, error_message: str | None = None) -> None:
        if not self.is_enabled() or not self._registered or not self.run_id:
            return

        now = _utcnow_iso()
        status = 'success' if success else 'failed'
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE experiment_run
                    SET
                        job_status = ?,
                        end_time = ?,
                        error_message = ?,
                        heartbeat_at = ?,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (status, now, error_message, now, now, self.run_id),
                )
                self._refresh_group_status_summary(conn, now=now)
                conn.commit()
        except Exception:
            logger.warning('Run registry finalization failed.', exc_info=True)

    def update_artifact_paths(
        self,
        *,
        summary_report_path: str | None = None,
        raw_results_path: str | None = None,
        quality_metric_details_path: str | None = None,
    ) -> None:
        if not self.is_enabled() or not self._registered or not self.run_id:
            return

        now = _utcnow_iso()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE experiment_run
                    SET
                        summary_report_path = COALESCE(?, summary_report_path),
                        raw_results_path = COALESCE(?, raw_results_path),
                        quality_metric_details_path = COALESCE(?, quality_metric_details_path),
                        result_path = COALESCE(?, result_path),
                        heartbeat_at = ?,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        summary_report_path,
                        raw_results_path,
                        quality_metric_details_path,
                        summary_report_path,
                        now,
                        now,
                        self.run_id,
                    ),
                )
                conn.commit()
        except Exception:
            logger.warning('Run registry artifact-path update failed.', exc_info=True)

    def _connect(self) -> sqlite3.Connection:
        if self._db_path is None:
            raise RuntimeError('sqlite db path is not available')
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON;')
        return conn

    def _dashboard_payload(self) -> dict[str, Any]:
        return get_dashboard_payload(self.config)

    def _set_dashboard_payload(self, payload: dict[str, Any]) -> None:
        existing = self._dashboard_payload()
        merged = dict(existing)
        merged.update(payload)

        try:
            setattr(self.config, 'dashboard', merged)
            return
        except Exception:
            pass

        extras = getattr(self.config, 'model_extra', None)
        if isinstance(extras, dict):
            extras['dashboard'] = merged

    def _set_context_ids(self, *, spec_id: str, run_id: str, template_id: str | None) -> None:
        self.spec_id = spec_id
        self.run_id = run_id
        self.template_id = template_id
        self.context.experiment_id = run_id
        self.context.spec_id = spec_id

    def _register_scheduler_run(
        self, conn: sqlite3.Connection, *, now: str, dashboard: dict[str, Any]
    ) -> None:
        spec_id = str(dashboard['exp_id'])
        run_id = str(dashboard['run_id'])
        template_id = dashboard.get('template_id')
        template_id = str(template_id) if template_id else None

        self._set_context_ids(spec_id=spec_id, run_id=run_id, template_id=template_id)
        run_token = str(dashboard.get('run_token') or run_id.replace('-', '')[:12])
        attempt_index = self._coerce_int(dashboard.get('attempt_index'))
        if attempt_index is None:
            attempt_index = self._next_attempt_index(conn, spec_id)

        self._ensure_group_row_for_exp_id(conn, spec_id=spec_id, template_id=template_id, now=now)
        self._set_dashboard_payload(
            {
                'exp_id': spec_id,
                'run_id': run_id,
                'run_token': run_token,
                'attempt_index': attempt_index,
                'template_id': template_id,
            }
        )
        self._upsert_run_start(
            conn,
            now=now,
            spec_id=spec_id,
            run_id=run_id,
            run_token=run_token,
            attempt_index=attempt_index,
            queued_at=str(dashboard.get('queued_at') or now),
            launcher='scheduler',
            job_kind=str(dashboard.get('job_kind') or 'primary'),
        )

    def _register_cli_run(self, conn: sqlite3.Connection, *, now: str) -> None:
        group_cfg = self._group_level_config()
        config_hash = self._compute_config_hash(group_cfg)

        existing = conn.execute(
            """
            SELECT g.exp_id, g.template_id
            FROM experiment_template t
            JOIN experiment_group g ON g.template_id = t.template_id
            WHERE t.config_hash = ?
            """,
            (config_hash,),
        ).fetchone()

        if existing is not None:
            spec_id = str(existing['exp_id'])
            template_id = str(existing['template_id'])
        else:
            spec_id, template_id = self._create_template_and_group(
                conn, now=now, group_cfg=group_cfg, config_hash=config_hash
            )

        run_id = str(uuid.uuid4())
        run_token = run_id.replace('-', '')[:12]
        attempt_index = self._next_attempt_index(conn, spec_id)

        self._set_context_ids(spec_id=spec_id, run_id=run_id, template_id=template_id)
        self._set_dashboard_payload(
            {
                'exp_id': spec_id,
                'run_id': run_id,
                'run_token': run_token,
                'attempt_index': attempt_index,
                'template_id': template_id,
                'queued_at': now,
            }
        )
        self._upsert_run_start(
            conn,
            now=now,
            spec_id=spec_id,
            run_id=run_id,
            run_token=run_token,
            attempt_index=attempt_index,
            queued_at=now,
            launcher='cli',
            job_kind=get_dashboard_job_kind(self.config),
        )

    def _upsert_run_start(
        self,
        conn: sqlite3.Connection,
        *,
        now: str,
        spec_id: str,
        run_id: str,
        run_token: str,
        attempt_index: int,
        queued_at: str,
        launcher: str,
        job_kind: str,
    ) -> None:
        log_path = self._resolve_log_path()
        output_dir = str(Path(self.context.output_dir).expanduser().resolve())
        data_path_raw = getattr(self.config, 'data_path', None)
        data_path = (
            str(Path(data_path_raw).expanduser().resolve())
            if isinstance(data_path_raw, str)
            else None
        )
        seed = self._coerce_int(
            getattr(getattr(self.config, 'runner_settings', None), 'seed', None)
        )
        host = socket.gethostname()
        pid = os.getpid()

        existing = conn.execute(
            'SELECT run_id FROM experiment_run WHERE run_id = ?',
            (run_id,),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO experiment_run (
                    run_id, run_token, exp_id, attempt_index, seed,
                    job_status, is_valid, is_deleted, excluded_from_aggregate,
                    created_at, updated_at, queued_at, start_time,
                    log_path, output_dir, data_path,
                    heartbeat_at, host, pid, launcher, job_kind
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, 1, 0, 0,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    run_token,
                    spec_id,
                    attempt_index,
                    seed,
                    'running',
                    now,
                    now,
                    queued_at,
                    now,
                    log_path,
                    output_dir,
                    data_path,
                    now,
                    host,
                    pid,
                    launcher,
                    job_kind,
                ),
            )
            return

        conn.execute(
            """
            UPDATE experiment_run
            SET
                exp_id = ?,
                run_token = ?,
                attempt_index = ?,
                seed = COALESCE(?, seed),
                job_status = ?,
                start_time = COALESCE(start_time, ?),
                queued_at = COALESCE(queued_at, ?),
                log_path = COALESCE(?, log_path),
                output_dir = COALESCE(?, output_dir),
                data_path = COALESCE(?, data_path),
                heartbeat_at = ?,
                host = COALESCE(?, host),
                pid = COALESCE(?, pid),
                launcher = COALESCE(?, launcher),
                job_kind = COALESCE(?, job_kind),
                updated_at = ?
            WHERE run_id = ?
            """,
            (
                spec_id,
                run_token,
                attempt_index,
                seed,
                'running',
                now,
                queued_at,
                log_path,
                output_dir,
                data_path,
                now,
                host,
                pid,
                launcher,
                job_kind,
                now,
                run_id,
            ),
        )

    def _ensure_group_row_for_exp_id(
        self, conn: sqlite3.Connection, *, spec_id: str, template_id: str | None, now: str
    ) -> None:
        existing_group = conn.execute(
            'SELECT exp_id FROM experiment_group WHERE exp_id = ?',
            (spec_id,),
        ).fetchone()
        if existing_group is not None:
            return

        effective_template_id = template_id
        if effective_template_id:
            existing_template = conn.execute(
                'SELECT template_id FROM experiment_template WHERE template_id = ?',
                (effective_template_id,),
            ).fetchone()
            if existing_template is None:
                effective_template_id = None

        if not effective_template_id:
            group_cfg = self._group_level_config()
            config_hash = self._compute_config_hash(group_cfg)
            existing = conn.execute(
                """
                SELECT t.template_id, g.exp_id
                FROM experiment_template t
                LEFT JOIN experiment_group g ON g.template_id = t.template_id
                WHERE t.config_hash = ?
                """,
                (config_hash,),
            ).fetchone()
            if existing and existing['template_id']:
                effective_template_id = str(existing['template_id'])

        if not effective_template_id:
            effective_template_id = str(uuid.uuid4())
            group_cfg = self._group_level_config()
            conn.execute(
                """
                INSERT INTO experiment_template (
                    template_id, config_hash, dataset, watermark, attack, base_model,
                    group_config_json, run_config_json, yaml_snapshot,
                    created_by, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    effective_template_id,
                    self._compute_config_hash(group_cfg),
                    str(getattr(getattr(self.config, 'dataset', None), 'name', 'unknown')),
                    str(getattr(self.config, 'watermark', 'unknown')),
                    str(getattr(self.config, 'attack', 'unknown')),
                    str(getattr(self.config, 'base_model', 'unknown')),
                    json.dumps(group_cfg, ensure_ascii=False, sort_keys=True),
                    json.dumps(self.config.model_dump(), ensure_ascii=False, sort_keys=True),
                    yaml.safe_dump(self.config.model_dump(), sort_keys=False),
                    'scheduler',
                    'autocreated by src run registry',
                    now,
                    now,
                ),
            )

        conn.execute(
            """
            INSERT INTO experiment_group (
                exp_id, exp_token, template_id, status_summary, source,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec_id,
                spec_id.replace('-', '')[:8],
                effective_template_id,
                'running',
                'scheduler',
                now,
                now,
            ),
        )

    def _create_template_and_group(
        self,
        conn: sqlite3.Connection,
        *,
        now: str,
        group_cfg: dict[str, Any],
        config_hash: str,
    ) -> tuple[str, str]:
        template_id = str(uuid.uuid4())
        spec_id = str(uuid.uuid4())

        dataset_cfg = getattr(self.config, 'dataset', None)
        dataset_name = str(getattr(dataset_cfg, 'name', 'unknown'))
        num_samples = self._coerce_int(getattr(dataset_cfg, 'num_samples', None))

        conn.execute(
            """
            INSERT INTO experiment_template (
                template_id, config_hash, dataset, watermark, attack, base_model,
                group_config_json, run_config_json, yaml_snapshot,
                created_by, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                config_hash,
                dataset_name,
                str(getattr(self.config, 'watermark', 'unknown')),
                str(getattr(self.config, 'attack', 'unknown')),
                str(getattr(self.config, 'base_model', 'unknown')),
                json.dumps(group_cfg, ensure_ascii=False, sort_keys=True),
                json.dumps(self.config.model_dump(), ensure_ascii=False, sort_keys=True),
                yaml.safe_dump(self.config.model_dump(), sort_keys=False),
                'cli',
                None,
                now,
                now,
            ),
        )

        conn.execute(
            """
            INSERT INTO experiment_group (
                exp_id, exp_token, template_id, status_summary, source,
                canonical_num_samples, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec_id,
                spec_id.replace('-', '')[:8],
                template_id,
                'running',
                'cli',
                num_samples,
                now,
                now,
            ),
        )
        return spec_id, template_id

    def _next_attempt_index(self, conn: sqlite3.Connection, spec_id: str) -> int:
        row = conn.execute(
            'SELECT COALESCE(MAX(attempt_index), 0) AS max_attempt FROM experiment_run WHERE exp_id = ?',
            (spec_id,),
        ).fetchone()
        return int(row['max_attempt']) + 1 if row else 1

    def _refresh_group_status_summary(self, conn: sqlite3.Connection, *, now: str) -> None:
        if not self.spec_id:
            return

        rows = conn.execute(
            """
            SELECT job_status
            FROM experiment_run
            WHERE
                exp_id = ?
                AND COALESCE(is_deleted, 0) = 0
                AND COALESCE(job_kind, 'primary') != 'quality_backfill'
            """,
            (self.spec_id,),
        ).fetchall()
        statuses = [str(row['job_status']) for row in rows]
        success_runs = sum(1 for s in statuses if s == 'success')
        failed_runs = sum(1 for s in statuses if s == 'failed')
        queued_runs = sum(1 for s in statuses if s == 'queued')
        running_runs = sum(1 for s in statuses if s == 'running')

        if running_runs > 0:
            summary = 'running'
        elif queued_runs > 0 and success_runs == 0 and failed_runs == 0:
            summary = 'queued'
        elif success_runs > 0 and (queued_runs > 0 or failed_runs > 0):
            summary = 'partial'
        elif success_runs > 0:
            summary = 'success'
        elif failed_runs > 0:
            summary = 'failed'
        else:
            summary = 'empty'

        conn.execute(
            'UPDATE experiment_group SET status_summary = ?, updated_at = ? WHERE exp_id = ?',
            (summary, now, self.spec_id),
        )

    def _resolve_sqlite_db_path(self) -> Path | None:
        return resolve_sqlite_db_path(self.config)

    def _group_level_config(self) -> dict[str, Any]:
        group_cfg = self.config.model_dump()
        group_cfg.pop('dashboard', None)
        group_cfg.pop('data_path', None)

        runner_settings = group_cfg.get('runner_settings')
        if isinstance(runner_settings, dict):
            runner_settings = dict(runner_settings)
            runner_settings.pop('seed', None)
            if runner_settings:
                group_cfg['runner_settings'] = runner_settings
            else:
                group_cfg.pop('runner_settings', None)
        return group_cfg

    @staticmethod
    def _compute_config_hash(group_cfg: dict[str, Any]) -> str:
        canonical = json.dumps(group_cfg, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _resolve_log_path(self) -> str | None:
        if getattr(self.context, 'log_path', None):
            return str(Path(self.context.log_path).expanduser().resolve())

        run_logger = logging.getLogger('MarkSec')
        for handler in run_logger.handlers:
            filename = getattr(handler, 'baseFilename', None)
            if filename:
                return str(Path(str(filename)).expanduser().resolve())
        return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS experiment_template (
            template_id TEXT PRIMARY KEY,
            config_hash TEXT NOT NULL UNIQUE,
            dataset TEXT NOT NULL,
            watermark TEXT NOT NULL,
            attack TEXT NOT NULL,
            base_model TEXT NOT NULL,
            group_config_json TEXT NOT NULL,
            run_config_json TEXT,
            yaml_snapshot TEXT NOT NULL,
            created_by TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experiment_group (
            exp_id TEXT PRIMARY KEY,
            exp_token TEXT NOT NULL UNIQUE,
            template_id TEXT NOT NULL UNIQUE,
            status_summary TEXT NOT NULL DEFAULT 'queued',
            batch_id TEXT NULL,
            source TEXT NOT NULL DEFAULT 'scheduler',
            canonical_num_samples INTEGER NULL,
            paper_status TEXT NULL,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(template_id) REFERENCES experiment_template(template_id)
        );

        CREATE TABLE IF NOT EXISTS experiment_run (
            run_id TEXT PRIMARY KEY,
            run_token TEXT NOT NULL UNIQUE,
            exp_id TEXT NOT NULL,
            attempt_index INTEGER NOT NULL,
            seed INTEGER,
            job_status TEXT NOT NULL,
            is_valid INTEGER NOT NULL DEFAULT 1,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            excluded_from_aggregate INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            queued_at TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_seconds REAL,
            machine TEXT,
            gpu_ids TEXT,
            log_path TEXT,
            result_path TEXT,
            error_message TEXT,
            state_file_yaml TEXT,
            state_file_log TEXT,
            queue_filename TEXT,
            deleted_at TEXT,
            output_dir TEXT,
            data_path TEXT,
            summary_report_path TEXT,
            raw_results_path TEXT,
            quality_metric_details_path TEXT,
            heartbeat_at TEXT,
            host TEXT,
            pid INTEGER,
            launcher TEXT,
            job_kind TEXT NOT NULL DEFAULT 'primary',
            FOREIGN KEY(exp_id) REFERENCES experiment_group(exp_id),
            UNIQUE(exp_id, attempt_index)
        );
        """
        conn.executescript(ddl)

        required_columns = {
            'experiment_group': {
                'batch_id': 'TEXT',
                'source': "TEXT NOT NULL DEFAULT 'scheduler'",
                'canonical_num_samples': 'INTEGER',
                'paper_status': 'TEXT',
                'archived': 'INTEGER NOT NULL DEFAULT 0',
            },
            'experiment_run': {
                'output_dir': 'TEXT',
                'data_path': 'TEXT',
                'summary_report_path': 'TEXT',
                'raw_results_path': 'TEXT',
                'quality_metric_details_path': 'TEXT',
                'heartbeat_at': 'TEXT',
                'host': 'TEXT',
                'pid': 'INTEGER',
                'launcher': 'TEXT',
                'job_kind': "TEXT NOT NULL DEFAULT 'primary'",
            },
        }

        for table_name, columns in required_columns.items():
            existing = {
                str(row['name'])
                for row in conn.execute(f'PRAGMA table_info({table_name})').fetchall()
            }
            for column_name, definition in columns.items():
                if column_name in existing:
                    continue
                conn.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}')
