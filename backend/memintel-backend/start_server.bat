@echo off
set MEMINTEL_CONFIG_PATH=memintel_config.yaml
set DATABASE_URL=postgresql://postgres:admin@localhost:5433/memintel
set REDIS_URL=redis://localhost:6379
set MEMINTEL_ELEVATED_KEY=test-elevated-key
uvicorn app.main:app --host 0.0.0.0 --port 8000
