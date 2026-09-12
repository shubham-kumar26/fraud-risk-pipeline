"""
Text-to-SQL feature for Fraud Risk Pipeline project.
Lets a user type a plain-English question and get back SQL + results
from the PostgreSQL fraud transactions database.
Uses Google Gemini's free API tier.
"""
import gradio as gr
import os
import re
import time
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, ClientError

load_dotenv()

client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.6-flash" # always points to Google's current fast model

# --- 1. Schema (matches your actual fraud pipeline database) ---
SCHEMA = """
Table: transactions
Columns:
  idx (int)
  trans_date_trans_time (timestamp)  -- when the transaction happened
  cc_num (bigint)  -- credit card number, identifies the customer
  merchant (varchar)  -- merchant name
  category (varchar)  -- merchant category, e.g. grocery, gas, entertainment
  amt (numeric)  -- transaction amount
  first (varchar)  -- customer first name
  last (varchar)  -- customer last name
  gender (char)
  street, city, state, zip (address fields)
  lat, long (numeric)  -- customer's home coordinates
  city_pop (int)  -- population of customer's city
  job (varchar)
  dob (date)  -- customer date of birth
  trans_num (varchar)  -- unique transaction ID
  unix_time (bigint)
  merch_lat, merch_long (numeric)  -- merchant's coordinates
  is_fraud (boolean)  -- true if transaction is fraudulent
"""

# --- 2. DB connection (edit host/dbname/user to match your local setup) ---
DB_CONFIG = {
    "host": "localhost",
    "dbname": "Sparkov",   # <-- fill this in, e.g. "fraud_db" or whatever you named it
    "user": "postgres",          # <-- your Postgres username, often "postgres"
    "password": "1234",
    "port": 5432,
}

DB_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)
engine = create_engine(DB_URL)

# --- Few-shot examples: help Gemini generate more accurate, consistent SQL ---
FEW_SHOT_EXAMPLES = """
Example 1:
Question: What's the average transaction amount for fraud vs non-fraud?
SQL: SELECT is_fraud, ROUND(AVG(amt)::numeric, 2) AS avg_amount FROM transactions GROUP BY is_fraud;

Example 2:
Question: Show the top 5 merchant categories by fraud rate
SQL: SELECT category, ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct FROM transactions GROUP BY category ORDER BY fraud_rate_pct DESC LIMIT 5;

Example 3:
Question: Which credit card has the most transactions?
SQL: SELECT cc_num, COUNT(*) AS total_transactions FROM transactions GROUP BY cc_num ORDER BY total_transactions DESC LIMIT 1;
"""

# Safety: only allow read-only queries
FORBIDDEN_KEYWORDS = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]


def _call_gemini(prompt: str):
    """Low-level Gemini call with retry-on-busy and retry-on-rate-limit logic."""
    response = None
    for attempt in range(4):  # try up to 4 times
        try:
            response = client_ai.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            break
        except ServerError:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            print(f"⏳ Gemini is busy, retrying in {wait}s...")
            time.sleep(wait)
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print("⚠️  Free-tier daily quota reached for this model (20 requests/day).")
                print("This resets in ~24 hours, or you can wait ~60s if it's a short-term burst limit.")
                return None
            raise  # re-raise other client errors (e.g. bad API key) instead of hiding them
    return response


def question_to_sql(question: str, previous_error: str = None, previous_sql: str = None) -> str:
    """Send the user's question + schema to Gemini, get back a SQL query.

    If previous_error/previous_sql are provided, this is a self-healing retry:
    Gemini sees what it generated last time and why it failed, and tries again.
    """
    if previous_error:
        prompt = f"""You are a SQL assistant for a PostgreSQL fraud analytics database.

Schema:
{SCHEMA}

Your previous SQL query failed with an error. Fix it.

Previous SQL:
{previous_sql}

Error from PostgreSQL:
{previous_error}

Original question: {question}

Output ONLY the corrected, valid PostgreSQL SELECT query. No explanation, no markdown, no backticks.
"""
    else:
        prompt = f"""You are a SQL assistant for a PostgreSQL fraud analytics database.

Schema:
{SCHEMA}

Here are some examples of good question-to-SQL conversions:
{FEW_SHOT_EXAMPLES}

Convert the following question into a single valid PostgreSQL SELECT query.
Only output the raw SQL query, with no explanation, no markdown formatting, no backticks.

Question: {question}
"""

    response = _call_gemini(prompt)

    if response is None:
        print("⚠️  Gemini is unavailable right now. Try again in a minute.")
        return None

    sql = response.text.strip()
    # Strip accidental markdown fences if the model adds them anyway
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def is_safe_query(sql: str) -> bool:
    """Basic guardrail: block anything that isn't a read-only SELECT."""
    upper_sql = sql.upper()
    if not upper_sql.strip().startswith("SELECT"):
        return False
    if any(keyword in upper_sql for keyword in FORBIDDEN_KEYWORDS):
        return False
    return True


def run_query(sql: str) -> pd.DataFrame:
    """Execute the SQL against the database and return results as a DataFrame."""
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df


def ask(question: str):
    """Full pipeline: question -> SQL -> results, with self-healing on failure."""
    print(f"\nQuestion: {question}")

    sql = question_to_sql(question)
    if sql is None:
        return None
    print(f"Generated SQL:\n{sql}\n")

    if not is_safe_query(sql):
        print("⚠️  Generated query failed safety check. Skipping execution.")
        return None

    try:
        df = run_query(sql)
        print(df.to_string(index=False) if not df.empty else "No results found.")
        return df
    except Exception as e:
        print(f"⚠️  Query failed: {e}")
        print("🔧 Asking Gemini to fix the query...")

        fixed_sql = question_to_sql(question, previous_error=str(e), previous_sql=sql)
        if fixed_sql is None or not is_safe_query(fixed_sql):
            print("⚠️  Could not generate a safe corrected query.")
            return None

        print(f"Corrected SQL:\n{fixed_sql}\n")
        try:
            df = run_query(fixed_sql)
            print(df.to_string(index=False) if not df.empty else "No results found.")
            return df
        except Exception as e2:
            print(f"⚠️  Corrected query also failed: {e2}")
            return None

def ask_gradio(question):
    """Gradio-friendly version of ask() — returns values instead of printing."""
    sql = question_to_sql(question)
    if sql is None:
        return "⚠️ Gemini is unavailable right now.", None

    if not is_safe_query(sql):
        return f"⚠️ Blocked unsafe query:\n{sql}", None

    try:
        df = run_query(sql)
        return sql, df
    except Exception as e:
        # try self-healing once, same as terminal version
        fixed_sql = question_to_sql(question, previous_error=str(e), previous_sql=sql)
        if fixed_sql is None or not is_safe_query(fixed_sql):
            return f"⚠️ Query failed and could not be auto-corrected:\n{e}", None
        try:
            df = run_query(fixed_sql)
            return f"{fixed_sql}\n\n(auto-corrected after an initial error)", df
        except Exception as e2:
            return f"⚠️ Corrected query also failed:\n{e2}", None





if __name__ == "__main__":
    interface = gr.Interface(
        fn=ask_gradio,
        inputs=gr.Textbox(label="Ask a question about the fraud data", placeholder="e.g. What's the fraud rate by category?"),
        outputs=[
            gr.Textbox(label="Generated SQL"),
            gr.Dataframe(label="Results"),
        ],
        title="Fraud Risk Pipeline — AI Query Interface",
        description="Ask questions in plain English. Powered by Gemini.",
    )
    interface.launch(share=True)
