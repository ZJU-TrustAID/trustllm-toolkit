import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pymysql
    import pymysql.cursors
except Exception:  # pragma: no cover - defensive import for optional runtime dependency
    pymysql = None

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    A lightweight multi-backend database manager.

    Supported backends:
    - sqlite (default, local file)
    - mysql (pymysql)
    - postgres (psycopg/psycopg2, optional dependency)
    """

    _IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

    def __init__(self, db_config: Optional[dict]):
        self.config = db_config or {}
        self.backend = self._infer_backend(self.config)
        self.connection = None
        self._placeholder = '?' if self.backend == 'sqlite' else '%s'
        self._pg2_cursor_factory = None

    @staticmethod
    def _infer_backend(db_config: dict) -> str:
        backend = str(db_config.get('backend', '')).strip().lower()
        if backend in {'sqlite', 'mysql', 'postgres', 'postgresql'}:
            return 'postgres' if backend == 'postgresql' else backend
        if 'host' in db_config and 'user' in db_config:
            return 'mysql'
        return 'sqlite'

    def __enter__(self):
        if self.backend == 'sqlite':
            self._connect_sqlite()
            return self
        if self.backend == 'mysql':
            self._connect_mysql()
            return self
        if self.backend == 'postgres':
            self._connect_postgres()
            return self
        raise ValueError(f'Unsupported database backend: {self.backend}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            if exc_type:
                logger.warning(f'An exception occurred. Rolling back transaction: {exc_val}')
                self.connection.rollback()
            else:
                self.connection.commit()
            self.connection.close()
            logger.info(f'Database connection closed ({self.backend}).')

    def _connect_sqlite(self) -> None:
        db_path = self.config.get('database') or self.config.get('path') or 'output/marksec.db'
        path = Path(db_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            path,
            timeout=self.config.get('connect_timeout', 30),
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA foreign_keys = ON;')
        logger.info(f'Database connection established using sqlite3 at {path}.')

    def _connect_mysql(self) -> None:
        if pymysql is None:
            raise ImportError('MySQL backend requires `pymysql`. Please install it first.')
        retryable_errnos = {2003, 2006, 2013}
        max_attempts = self.config.get('connect_retries', 3)
        backoff = self.config.get('connect_retry_backoff', 3)
        last_exc = None
        for attempt in range(max_attempts):
            try:
                self.connection = pymysql.connect(
                    host=self.config['host'],
                    user=self.config['user'],
                    password=self.config.get('password', ''),
                    database=self.config['database'],
                    port=self.config.get('port', 3306),
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=self.config.get('connect_timeout', 30),
                )
                logger.info('Database connection established using pymysql.')
                return
            except pymysql.OperationalError as exc:
                if exc.args[0] not in retryable_errnos or attempt == max_attempts - 1:
                    logger.error(f'Failed to connect to database using pymysql: {exc}')
                    raise
                last_exc = exc
                wait = backoff * (2**attempt)
                logger.warning(
                    f'Transient connection error (errno {exc.args[0]}), '
                    f'retrying in {wait}s (attempt {attempt + 1}/{max_attempts}): {exc}'
                )
                time.sleep(wait)
            except pymysql.Error as exc:
                logger.error(f'Failed to connect to database using pymysql: {exc}')
                raise
        raise last_exc

    def _connect_postgres(self) -> None:
        dbname = self.config.get('database') or self.config.get('dbname')
        if not dbname:
            raise ValueError("Postgres config must include 'database' (or 'dbname').")

        try:
            import psycopg
            from psycopg.rows import dict_row

            self.connection = psycopg.connect(
                host=self.config.get('host', '127.0.0.1'),
                user=self.config['user'],
                password=self.config.get('password', ''),
                dbname=dbname,
                port=self.config.get('port', 5432),
                connect_timeout=self.config.get('connect_timeout', 30),
                row_factory=dict_row,
            )
            logger.info('Database connection established using psycopg.')
            return
        except ImportError:
            pass

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            self.connection = psycopg2.connect(
                host=self.config.get('host', '127.0.0.1'),
                user=self.config['user'],
                password=self.config.get('password', ''),
                dbname=dbname,
                port=self.config.get('port', 5432),
                connect_timeout=self.config.get('connect_timeout', 30),
            )
            self._pg2_cursor_factory = RealDictCursor
            logger.info('Database connection established using psycopg2.')
            return
        except ImportError as exc:
            raise ImportError(
                'Postgres backend requires `psycopg` or `psycopg2`. '
                'Install one of them to use postgres.'
            ) from exc

    def _quote_identifier(self, name: str) -> str:
        if not isinstance(name, str) or not self._IDENTIFIER_RE.fullmatch(name):
            raise ValueError(f'Unsafe SQL identifier: {name!r}')
        if self.backend == 'postgres':
            return f'"{name}"'
        return f'`{name}`'

    def _new_cursor(self):
        if self.backend == 'postgres' and self._pg2_cursor_factory is not None:
            return self.connection.cursor(cursor_factory=self._pg2_cursor_factory)
        return self.connection.cursor()

    @staticmethod
    def _normalize_row(row: Any, cursor) -> Any:
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        if isinstance(row, sqlite3.Row):
            return dict(row)
        if hasattr(row, 'keys'):
            return {key: row[key] for key in row.keys()}
        if cursor.description:
            return {desc[0]: row[idx] for idx, desc in enumerate(cursor.description)}
        return row

    def _execute(self, sql: str, params: Optional[tuple] = None, fetch: Optional[str] = None):
        cursor = self._new_cursor()
        try:
            cursor.execute(sql, params or ())
            if fetch == 'one':
                return self._normalize_row(cursor.fetchone(), cursor)
            if fetch == 'all':
                rows = cursor.fetchall()
                return [self._normalize_row(row, cursor) for row in rows]
            return cursor.rowcount
        finally:
            cursor.close()

    def _where_clause(self, where_conditions: Dict[str, Any]) -> tuple[str, tuple]:
        clause = ' AND '.join(
            f'{self._quote_identifier(key)} = {self._placeholder}'
            for key in where_conditions.keys()
        )
        return clause, tuple(where_conditions.values())

    def _primary_key_columns(self, table_name: str) -> List[str]:
        if self.backend == 'sqlite':
            safe_table = self._quote_identifier(table_name)
            rows = self._execute(f'PRAGMA table_info({safe_table})', fetch='all')
            pk_rows = [
                (int(row.get('pk', 0)), row.get('name'))
                for row in rows
                if int(row.get('pk', 0)) > 0
            ]
            return [name for _, name in sorted(pk_rows)]

        if self.backend == 'mysql':
            sql = (
                'SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS '
                'WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_KEY = %s '
                'ORDER BY ORDINAL_POSITION'
            )
            rows = self._execute(sql, (table_name, 'PRI'), fetch='all')
            return [row['COLUMN_NAME'] for row in rows]

        sql = (
            'SELECT kcu.column_name FROM information_schema.table_constraints tc '
            'JOIN information_schema.key_column_usage kcu '
            'ON tc.constraint_name = kcu.constraint_name '
            'AND tc.table_schema = kcu.table_schema '
            'WHERE tc.constraint_type = %s '
            'AND tc.table_schema = current_schema() '
            'AND tc.table_name = %s '
            'ORDER BY kcu.ordinal_position'
        )
        rows = self._execute(sql, ('PRIMARY KEY', table_name), fetch='all')
        return [row['column_name'] for row in rows]

    def table_exists(self, table_name: str) -> bool:
        if self.backend == 'sqlite':
            sql = (
                f'SELECT COUNT(*) AS count FROM sqlite_master '
                f'WHERE type = {self._placeholder} AND name = {self._placeholder}'
            )
            result = self._execute(sql, ('table', table_name), fetch='one')
            return result['count'] > 0

        if self.backend == 'mysql':
            sql = (
                'SELECT COUNT(*) AS count FROM INFORMATION_SCHEMA.TABLES '
                'WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s'
            )
            result = self._execute(sql, (table_name,), fetch='one')
            return result['count'] > 0

        sql = (
            'SELECT COUNT(*) AS count FROM information_schema.tables '
            'WHERE table_schema = current_schema() AND table_name = %s'
        )
        result = self._execute(sql, (table_name,), fetch='one')
        return result['count'] > 0

    def column_exists(self, table_name: str, column_name: str) -> bool:
        if self.backend == 'sqlite':
            safe_table = self._quote_identifier(table_name)
            rows = self._execute(f'PRAGMA table_info({safe_table})', fetch='all')
            return any(row.get('name') == column_name for row in rows)

        if self.backend == 'mysql':
            sql = (
                'SELECT COUNT(*) AS count FROM INFORMATION_SCHEMA.COLUMNS '
                'WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s'
            )
            result = self._execute(sql, (table_name, column_name), fetch='one')
            return result['count'] > 0

        sql = (
            'SELECT COUNT(*) AS count FROM information_schema.columns '
            'WHERE table_schema = current_schema() AND table_name = %s AND column_name = %s'
        )
        result = self._execute(sql, (table_name, column_name), fetch='one')
        return result['count'] > 0

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        if self.backend == 'sqlite':
            safe_table = self._quote_identifier(table_name)
            rows = self._execute(f'PRAGMA table_info({safe_table})', fetch='all')
            return [
                {
                    'COLUMN_NAME': row.get('name'),
                    'COLUMN_TYPE': row.get('type'),
                    'IS_NULLABLE': 'NO' if row.get('notnull') else 'YES',
                    'COLUMN_KEY': 'PRI' if row.get('pk') else '',
                    'COLUMN_DEFAULT': row.get('dflt_value'),
                    'EXTRA': '',
                }
                for row in rows
            ]

        if self.backend == 'mysql':
            sql = (
                'SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, '
                'COLUMN_DEFAULT, EXTRA '
                'FROM INFORMATION_SCHEMA.COLUMNS '
                'WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s'
            )
            return self._execute(sql, (table_name,), fetch='all')

        sql = (
            'SELECT column_name AS "COLUMN_NAME", '
            'data_type AS "COLUMN_TYPE", '
            'is_nullable AS "IS_NULLABLE", '
            '\'\' AS "COLUMN_KEY", '
            'column_default AS "COLUMN_DEFAULT", '
            '\'\' AS "EXTRA" '
            'FROM information_schema.columns '
            'WHERE table_schema = current_schema() AND table_name = %s'
        )
        return self._execute(sql, (table_name,), fetch='all')

    def list_tables(self) -> list:
        if self.backend == 'sqlite':
            sql = (
                f'SELECT name FROM sqlite_master '
                f'WHERE type = {self._placeholder} AND name NOT LIKE {self._placeholder} '
                'ORDER BY name'
            )
            rows = self._execute(sql, ('table', 'sqlite_%'), fetch='all')
            return [row['name'] for row in rows]

        if self.backend == 'mysql':
            rows = self._execute('SHOW TABLES', fetch='all')
            if not rows:
                return []
            key = list(rows[0].keys())[0]
            return [row[key] for row in rows]

        sql = (
            'SELECT table_name FROM information_schema.tables '
            'WHERE table_schema = current_schema() ORDER BY table_name'
        )
        rows = self._execute(sql, fetch='all')
        return [row['table_name'] for row in rows]

    def create_table(self, table_name: str, columns_def: str, if_not_exists: bool = True):
        if if_not_exists and self.table_exists(table_name):
            logger.info(f"Table '{table_name}' already exists. Skipping creation.")
            return

        safe_table = self._quote_identifier(table_name)
        if_clause = 'IF NOT EXISTS ' if if_not_exists else ''
        sql = f'CREATE TABLE {if_clause}{safe_table} ({columns_def})'
        if self.backend == 'mysql':
            sql += ' ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'

        logger.info(f'Executing DDL: {sql}')
        self._execute(sql)
        logger.info(f"Table '{table_name}' created successfully.")

    def drop_table(self, table_name: str, if_exists: bool = True):
        if if_exists and not self.table_exists(table_name):
            logger.info(f"Table '{table_name}' does not exist. Skipping drop.")
            return
        safe_table = self._quote_identifier(table_name)
        sql = f'DROP TABLE {"IF EXISTS " if if_exists else ""}{safe_table}'
        logger.info(f'Executing DDL: {sql}')
        self._execute(sql)
        logger.info(f"Table '{table_name}' dropped successfully.")

    def add_column(self, table_name: str, column_def: str, if_not_exists: bool = True):
        column_name = column_def.strip().split()[0].strip('`"')

        if if_not_exists and self.column_exists(table_name, column_name):
            logger.info(f"Column '{column_name}' already exists in table '{table_name}'. Skipping.")
            return

        safe_table = self._quote_identifier(table_name)
        sql = f'ALTER TABLE {safe_table} ADD COLUMN {column_def}'

        logger.info(f'Executing DDL: {sql}')
        self._execute(sql)
        logger.info(f"Column added to table '{table_name}' successfully.")

    def upsert(self, table: str, data: dict):
        if not data:
            raise ValueError('`data` must be provided for upsert.')

        safe_table = self._quote_identifier(table)
        columns = list(data.keys())
        quoted_columns = ', '.join(self._quote_identifier(col) for col in columns)
        placeholders = ', '.join([self._placeholder] * len(columns))
        params = tuple(data[col] for col in columns)

        if self.backend == 'mysql':
            update_clause = ', '.join(
                f'{self._quote_identifier(col)} = VALUES({self._quote_identifier(col)})'
                for col in columns
            )
            sql = (
                f'INSERT INTO {safe_table} ({quoted_columns}) VALUES ({placeholders}) '
                f'ON DUPLICATE KEY UPDATE {update_clause}'
            )
            return self._execute(sql, params)

        pk_columns = self._primary_key_columns(table)
        if not pk_columns:
            sql = f'INSERT INTO {safe_table} ({quoted_columns}) VALUES ({placeholders})'
            return self._execute(sql, params)

        conflict_cols = ', '.join(self._quote_identifier(col) for col in pk_columns)
        update_columns = [col for col in columns if col not in pk_columns]
        if update_columns:
            update_clause = ', '.join(
                f'{self._quote_identifier(col)} = EXCLUDED.{self._quote_identifier(col)}'
                for col in update_columns
            )
            sql = (
                f'INSERT INTO {safe_table} ({quoted_columns}) VALUES ({placeholders}) '
                f'ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}'
            )
        else:
            sql = (
                f'INSERT INTO {safe_table} ({quoted_columns}) VALUES ({placeholders}) '
                f'ON CONFLICT ({conflict_cols}) DO NOTHING'
            )
        return self._execute(sql, params)

    def update(self, table: str, data_to_set: dict, where_conditions: dict):
        if not data_to_set or not where_conditions:
            raise ValueError("Both 'data_to_set' and 'where_conditions' must be provided.")

        safe_table = self._quote_identifier(table)
        set_clause = ', '.join(
            f'{self._quote_identifier(key)} = {self._placeholder}' for key in data_to_set.keys()
        )
        where_clause, where_params = self._where_clause(where_conditions)
        sql = f'UPDATE {safe_table} SET {set_clause} WHERE {where_clause}'
        params = tuple(data_to_set.values()) + where_params
        return self._execute(sql, params)

    def query(
        self,
        table: str,
        select_keys: Optional[list] = None,
        where_conditions: Optional[dict] = None,
    ):
        safe_table = self._quote_identifier(table)

        if select_keys:
            select_clause = ', '.join(self._quote_identifier(key) for key in select_keys)
        else:
            select_clause = '*'

        sql = f'SELECT {select_clause} FROM {safe_table}'
        params: tuple = ()

        if where_conditions:
            where_clause, params = self._where_clause(where_conditions)
            sql += f' WHERE {where_clause}'

        return self._execute(sql, params, fetch='all')

    def delete(self, table: str, where_conditions: Dict[str, Any]) -> int:
        if not where_conditions:
            raise ValueError(
                '`where_conditions` must be provided to prevent accidental full table deletion.'
            )
        safe_table = self._quote_identifier(table)
        where_clause, params = self._where_clause(where_conditions)
        sql = f'DELETE FROM {safe_table} WHERE {where_clause}'
        return self._execute(sql, params)

    def count(self, table: str, where_conditions: Optional[Dict[str, Any]] = None) -> int:
        safe_table = self._quote_identifier(table)
        sql = f'SELECT COUNT(*) AS count FROM {safe_table}'
        params: tuple = ()
        if where_conditions:
            where_clause, params = self._where_clause(where_conditions)
            sql += f' WHERE {where_clause}'
        result = self._execute(sql, params, fetch='one')
        return int(result['count'])
