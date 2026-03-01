# TapTap Analytics Chatbot

A production-ready FastAPI backend for the TapTap analytics chatbot that connects to PostgreSQL and uses LLM tool calling to answer faculty questions about student performance data.

## Features

- **Natural Language Queries**: Faculty can ask questions in plain English about student data
- **LLM-Powered**: Uses Azure OpenAI with tool calling to convert questions to SQL
- **Intent & Slot Extraction**: The chatbot first detects the analytics domain and required parameters, validates them in Python, and only then builds the SQL query
- **Safe Database Access**: Only allows SELECT queries, preventing data modification
- **Async Architecture**: Built with FastAPI and asyncpg for high performance
- **Comprehensive Analytics**: Supports queries about tests, employability scores, POD submissions, and more

## Architecture

```
app/
├── main.py      # FastAPI application with /chat endpoint
├── config.py    # Configuration management
├── db.py        # Database connection and query execution (asyncpg pool)
├── tools.py     # SQL execution tool for LLM
└── llm.py       # LLM integration with Azure OpenAI tool-calling
```

## Supported Queries

- "Who solved today's POD in IT domain?"
- "Who is the top student in MET test?"
- "Top 10 students by average coding score"
- "Students with employability score above 80"
- "Students at risk (score below 40)"
- "Average verbal, coding, reasoning score per student"

## Database Schema

The system expects these key tables:
- `public.user` - Student information
- `public.test_submission` - Test submissions
- `public.employability_track_submission` - Employability test results
- `pod.pod_submission` - Problem of the Day submissions
- `pod.problem_of_the_day` - Daily problems
- `public.block` - Question metadata
- `public.domains` - Available domains

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd taptap-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set up environment variables:
```bash
cp .env.example .env   # keep the real credentials local only
# Edit `.env` with your configuration. Do **not** commit this file; it is
# ignored by `.gitignore` and may contain sensitive keys.
```

5. Optional utilities are available in the `tests/` and `scripts/` directories:
   * `tests/test_llm.py`, `tests/test_queries.py` – simple scripts that
     initialise the app and run sample natural‑language queries.
   * `scripts/print_schema.py`, `scripts/inspect_schema.py` – helpers to
     dump schema information from the database.

These are for development/demo purposes and can safely be ignored by the
application itself.

## Configuration

Create a `.env` file with the following variables:

```env
# Database Configuration
DB_HOST=your-azure-db-host.postgres.database.azure.com
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_SSL=require

# LLM Configuration
LLM_PROVIDER=anthropic  # or openai
LLM_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key  # Only if using OpenAI

# Application Configuration
DEBUG=false
MAX_QUERY_RESULTS=100
```

## Running the Application

### Development

```bash
python -m app.main
```

The server will start on `http://localhost:8000`

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### POST /chat

Process a natural language query about student analytics.  The endpoint now
supports a lightweight conversation state so that the chatbot can remember
which domain the user is discussing and ask follow‑up questions if necessary.

**Request:**
```json
{
  "query": "Who solved today's POD in IT domain?",
  "state": {}
}
```

If the model needs more information (e.g. a missing slot) it will return a
clarifying question along with an updated state object.  Clients should pass
that state back in subsequent requests until a complete answer is produced.

**Response:**
```json
{
  "answer": "Based on the data, 5 students solved today's POD in the IT domain: John Doe, Jane Smith, Mike Johnson, Sarah Williams, and Tom Brown.",
  "data": [
    {
      "user_id": 1,
      "name": "John Doe",
      "problem_id": 123,
      "submitted_at": "2024-01-15T10:30:00Z"
    }
  ],
  "state": {"domain":"pod_submission","domain_id":2},
  "success": true
}
```

### GET /health

Check the health status of the application.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "llm": "initialized",
  "version": "1.0.0"
}
```

### GET /info

Get information about available database tables and supported queries.

### GET /docs

Interactive API documentation (Swagger UI).

## Safety Features

- **Query Validation**: Only SELECT queries are allowed
- **SQL Injection Protection**: Uses parameterized queries
- **Result Limiting**: Automatically limits results to prevent excessive data transfer
- **Error Handling**: Comprehensive error handling and logging

## Deployment

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t taptap-chatbot .
docker run -p 8000:8000 --env-file .env taptap-chatbot
```

### Azure

The application is designed to work with Azure PostgreSQL. Ensure:
- SSL mode is set to `require`
- Firewall rules allow your application IP
- Connection pooling is configured appropriately

## Monitoring

The application includes comprehensive logging:
- Application startup/shutdown events
- Database connection status
- Query processing logs
- Error tracking

## Performance

- **Connection Pooling**: Uses asyncpg connection pool (5-20 connections)
- **Async Operations**: Fully async architecture for high concurrency
- **Query Optimization**: LLM generates efficient SQL queries
- **Result Caching**: Consider adding Redis for frequent queries

## Troubleshooting

### Database Connection Issues
- Verify Azure PostgreSQL credentials
- Check SSL configuration
- Ensure firewall allows your IP

### LLM API Issues
- Verify API keys are correct
- Check rate limits
- Ensure proper provider configuration

### Query Errors
- Check database schema matches expectations
- Verify table and column names
- Review LLM-generated SQL for syntax issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]
