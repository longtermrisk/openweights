#!/bin/bash

# Integration test runner for OpenWeights
# This script runs the full integration test suite against the dev Supabase database

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "OpenWeights Integration Test Suite"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python --version)${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check .env.worker
if [ ! -f ".env.worker" ]; then
    echo -e "${RED}✗ .env.worker file not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ .env.worker found${NC}"

# Check required env variables
source .env.worker
if [ -z "$SUPABASE_URL" ]; then
    echo -e "${RED}✗ SUPABASE_URL not set in .env.worker${NC}"
    exit 1
fi
if [ -z "$SUPABASE_ANON_KEY" ]; then
    echo -e "${RED}✗ SUPABASE_ANON_KEY not set in .env.worker${NC}"
    exit 1
fi
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}✗ RUNPOD_API_KEY not set in .env.worker${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Required environment variables set${NC}"

echo ""
echo "=========================================="
echo "Starting Integration Tests"
echo "=========================================="
echo ""

# Run the tests
python tests/test_integration.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✓ All integration tests passed!"
    echo -e "==========================================${NC}"
else
    echo ""
    echo -e "${RED}=========================================="
    echo "✗ Some integration tests failed"
    echo -e "==========================================${NC}"
fi

exit $exit_code
