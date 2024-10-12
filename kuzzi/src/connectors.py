import os
import psycopg2
import pandas as pd
from typing import Union

class Connector:
    def __init__(self):
        self.ready = False

    def get_env(self, key: str, error_msg: str) -> str:
        """Helper to retrieve env variables or raise an error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(error_msg)
        return value

    def connect(self, **options):
        """To be implemented by subclasses for connection logic."""
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, query: str) -> Union[pd.DataFrame, None]:
        """To be implemented by subclasses to run SQL queries."""
        raise NotImplementedError("Subclasses must implement this method")


class PostgresConnector(Connector):
    def __init__(self):
        super().__init__()

    def connect(
        self,
        host: str = None,
        name: str = None,
        user: str = None,
        pwd: str = None,
        port: int = None,
        **options
    ):
        """
        Connect to PostgreSQL using psycopg2.
        Use env vars if params aren't provided.
        """
        # Get from env if not provided
        host = host or self.get_env("DB_HOST", "Please set your Postgres host.")
        name = name or self.get_env("DB_NAME", "Please set your Postgres database name.")
        user = user or self.get_env("DB_USER", "Please set your Postgres user.")
        pwd = pwd or self.get_env("DB_PASSWORD", "Please set your Postgres password.")
        port = port or self.get_env("DB_PORT", "Please set your Postgres port.")

        # Internal function to establish the connection
        def make_conn():
            try:
                return psycopg2.connect(
                    host=host,
                    dbname=name,
                    user=user,
                    password=pwd,
                    port=port,
                    **options
                )
            except psycopg2.Error as e:
                raise ConnectionError(f"Failed to connect: {e}")

        # Save the connection function for reuse
        self.conn_func = make_conn
        self.ready = True

    def run(self, query: str) -> Union[pd.DataFrame, None]:
        """Runs the given SQL query and returns results as a DataFrame."""
        if not self.ready:
            raise RuntimeError("Connection not set. Call connect() first.")

        conn = None
        try:
            conn = self.conn_func()
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                # Convert results to DataFrame
                df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
                return df
        except psycopg2.InterfaceError:
            # Reconnect if needed
            if conn:
                conn.close()
            conn = self.conn_func()
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
                return df
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Query execution failed: {e}")
        finally:
            if conn:
                conn.close()
