import pandas as pd
import pyodbc
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManager

from common.prompts import MSSQL_AGENT_PREFIX, MSSQL_AGENT_SUFFIX, MSSQL_AGENT_FORMAT_INSTRUCTIONS

from IPython.display import Markdown, HTML, display

from dotenv import load_dotenv
load_dotenv("credentials.env")

def printmd(string):
    display(Markdown(string))

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import os

os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]

MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most interesting examples in the database.
- Never query for all the columns from a specific table, only ask for the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE. 
- Your response should be in Markdown. However, **when running  a SQL Query  in "Action Input", do not include the markdown backticks**. Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer on a section that starts with: "Explanation:".
- If the question does not seem related to the database, just return "I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the tools below.
- You will be penalized with -1000 dollars if you don't provide the sql queries used in your final answer.
- You will be rewarded 1000 dollars if you provide the sql queries used in your final answer.


## Tools:

"""

MSSQL_AGENT_SUFFIX = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer. 
Thought: you should always think about what to do. 
Action: the action to take, should be one of [{tool_names}]. 
Action Input: the input to the action. 
Observation: the result of the action. 
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. 
Final Answer: the final answer to the original input question. 

### Examples of Final Answer:

Example 1:

Action: query_sql_db
Action Input: SELECT TOP (10) [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'
Observation: [(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 people who died of covid in Texas in 2020.

Explanation:
I queried the `covidtracking` table for the `death` column where the state is 'TX' and the date starts with '2020'. The query returned a list of tuples with the number of deaths for each day in 2020. To answer the question, I took the sum of all the deaths in the list, which is 27437. 
I used the following query

```sql
SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
```

Example 2:

Action: query_sql_db
Action Input: SELECT AVG(price) AS average_price FROM sales WHERE year = '2021'
Observation: [(322.5,)]
Thought: I now know the final answer
Final Answer: The average sales price in 2021 was $322.5.

Explanation:
I queried the `sales` table for the average `price` where the year is '2021'. The SQL query used is:

```sql
SELECT AVG(price) AS average_price FROM sales WHERE year = '2021'
```
This query calculates the average price of all sales in the year 2021, which is $322.5.

Example 3:

Action: query_sql_db
Action Input: SELECT COUNT(DISTINCT customer_id) FROM orders WHERE order_date BETWEEN '2022-01-01' AND '2022-12-31'
Observation: [(150,)]
Thought: I now know the final answer
Final Answer: There were 150 unique customers who placed orders in 2022.

Explanation:
To find the number of unique customers who placed orders in 2022, I used the following SQL query:

```sql
SELECT COUNT(DISTINCT customer_id) FROM orders WHERE order_date BETWEEN '2022-01-01' AND '2022-12-31'
```
This query counts the distinct `customer_id` entries within the `orders` table for the year 2022, resulting in 150 unique customers.

Example 4:

Action: query_sql_db
Action Input: SELECT TOP 1 name FROM products ORDER BY rating DESC
Observation: [('UltraWidget',)]
Thought: I now know the final answer
Final Answer: The highest-rated product is called UltraWidget.

Explanation:
I queried the `products` table to find the name of the highest-rated product using the following SQL query:

```sql
SELECT TOP 1 name FROM products ORDER BY rating DESC
```
This query selects the product name from the `products` table and orders the results by the `rating` column in descending order. The `TOP 1` clause ensures that only the highest-rated product is returned, which is 'UltraWidget'.

"""


# Configuration for the database connection
db_config = {
    'drivername': 'mssql+pyodbc',
    'username': os.environ["SQL_SERVER_USERNAME"] + '@' + os.environ["SQL_SERVER_NAME"],
    'password': os.environ["SQL_SERVER_PASSWORD"],
    'host': os.environ["SQL_SERVER_NAME"],
    'port': 1433,
    'database': os.environ["SQL_SERVER_DATABASE"],
    'query': {'driver': 'ODBC Driver 17 for SQL Server'},
}

# Create a URL object for connecting to the database
db_url = URL.create(**db_config)

# Connect to the Azure SQL Database using the URL string
engine = create_engine(db_url)

# Test the connection using the SQLAlchemy 2.0 execution style
with engine.connect() as conn:
    try:
        # Use the text() construct for safer SQL execution
        result = conn.execute(text("SELECT @@VERSION"))
        version = result.fetchone()
        print("Connection successful!")
        print(version)
    except Exception as e:
        print(e)

llm = AzureChatOpenAI(deployment_name=os.environ["GPT35_DEPLOYMENT_NAME"], temperature=0.5, max_tokens=2000)

# Let's create the db object
db = SQLDatabase.from_uri(db_url)

# Natural Language question (query)
QUESTION = "Fetch me 10 records from dbo.Orders table grouped by CustomerID"

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    suffix=MSSQL_AGENT_SUFFIX,
    format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=llm,
    toolkit=toolkit,
    top_k=30,
    agent_type="openai-tools",
    verbose=True
)

# As we know by now, Agents use expert/tools. Let's see which are the tools for this SQL Agent
agent_executor.tools

try:
    response = agent_executor.invoke(QUESTION)
except Exception as e:
    response = str(e)


print(response)

