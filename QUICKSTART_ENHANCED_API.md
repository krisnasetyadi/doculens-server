# Quick Start: Testing Enhanced Hybrid API

## Prerequisites
- Python environment activated
- FastAPI server running on `localhost:8000`
- Sample data loaded (PDFs, database records, chat logs)

## Step 1: Start the Server
```powershell
cd c:\Users\RentalWorks-D5CKVT2\Documents\PDFREADER\pdf-reader
.\source\Scripts\Activate.ps1
python main.py
```

Wait for: `INFO:     Application startup complete.`

## Step 2: Run Enhanced Feature Tests
Open a new PowerShell terminal:

```powershell
cd c:\Users\RentalWorks-D5CKVT2\Documents\PDFREADER\pdf-reader
.\source\Scripts\Activate.ps1
python test_enhanced_features.py
```

## Expected Output

### ✅ Test 1: Exact Match Boost
```
TEST 1: EXACT MATCH BOOST - Brand Name Search
================================================================================
✅ Status: 200
📊 Answer: Berdasarkan data dari tabel products...

🔍 Answer Metadata:
  • Confidence Score: 95.50%
  • Primary Intent: data_retrieval
  • Sources Used: {'pdf': 0, 'database': 3, 'chat': 0}

🎯 Exact Matches Found: 1
  1. Term: 'jetbrains' in database (products table)

⚡ Boost Applied:
  • Exact Match: +50.0 points
  • Person Query: 0.0x multiplier

✓ JetBrains exact match: FOUND
```

### ✅ Test 2: Person Query Prioritization
```
TEST 2: PERSON QUERY PRIORITIZATION
================================================================================
✅ Status: 200
📊 Answer: Berdasarkan percakapan dari team_chat...

🔍 Answer Metadata:
  • Is Person Query: True
  • Sources Used: {'pdf': 0, 'database': 1, 'chat': 2}

⚡ Person Query Boost: 1.8x
  ✓ Person query prioritization ACTIVE

📋 Top 3 Results (prioritized):
  1. Type: chat, Confidence: 78.00%
  2. Type: database, Confidence: 65.00%
  3. Type: pdf, Confidence: 45.00%

  Chat/DB in top 3: 2/3
```

### ✅ Test 3: Conflict Detection
```
TEST 3: CONFLICT DETECTION
================================================================================
✅ Status: 200
📊 Answer: Berdasarkan berbagai sumber...

⚠️ Conflicts Detected: True
   Number of Conflicts: 1

🔍 Conflict Details:

  Conflict #1:
    • Type: value_mismatch
    • Entity: price_database
    • Message: Conflicting information for price_database: ['500000', '450000']
    • Conflicting Values:
      - 500000 from ['database']
      - 450000 from ['chat']
    • Resolution: Prioritize 500000 (highest confidence)
```

### ✅ Test 4: Multi-Source Hybrid Search
```
TEST 4: MULTI-SOURCE HYBRID SEARCH
================================================================================
✅ Status: 200
📊 Answer Length: 256 chars
⏱️ Processing Time: 1.85s

🔍 Comprehensive Metadata:
  • Confidence: 78.00%
  • Intent: explanation
  • Strategy: hybrid_multi_source
  • Ranking: weighted_scoring_with_intent
  • Total Results: 12
  • Model: huggingface/google/flan-t5-base

📊 Search Analysis:
  • Intent Type: explanation
  • Aggregation: False
  • Comparison: False

🔄 Processing Steps (5 total):
  • Intent analysis completed
  • Sources merged and ranked
  • Detected 3 exact matches
  • Answer validated and cleaned
```

### ✅ Test 5: Intent Detection
```
TEST 5: INTENT DETECTION
================================================================================
📝 Query: Berapa total harga semua license?
   Expected Intent: aggregation
   Detected Intent: aggregation ✓

📝 Query: Bandingkan harga JetBrains dengan SAP
   Expected Intent: comparison
   Detected Intent: comparison ✓

📝 Query: Apa itu DevOps dan kenapa penting?
   Expected Intent: explanation
   Detected Intent: explanation ✓

📝 Query: Siapa IT Manager?
   Expected Intent: data_retrieval
   Detected Intent: data_retrieval ✓
```

## Step 3: Test Individual Features

### Test Exact Match Boost
```powershell
curl -X POST http://localhost:8000/api/v1/query/hybrid/enhanced `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"License JetBrains harganya berapa?\", \"include_pdf_results\": true, \"include_db_results\": true, \"include_chat_results\": false}'
```

### Test Person Query
```powershell
curl -X POST http://localhost:8000/api/v1/query/hybrid/enhanced `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"Siapa yang handle infrastructure?\", \"include_pdf_results\": true, \"include_db_results\": true, \"include_chat_results\": true}'
```

### Test Conflict Detection
```powershell
curl -X POST http://localhost:8000/api/v1/query/hybrid/enhanced `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"Berapa harga SAP license?\", \"include_pdf_results\": true, \"include_db_results\": true, \"include_chat_results\": true}'
```

## Step 4: Verify Metadata Structure

Check the response has all expected fields:

```python
import requests
import json

response = requests.post("http://localhost:8000/api/v1/query/hybrid/enhanced", json={
    "question": "Test query",
    "include_pdf_results": True,
    "include_db_results": True,
    "include_chat_results": True
})

result = response.json()

# Verify answer_metadata structure
assert 'confidence_score' in result['answer_metadata']
assert 'primary_intent' in result['answer_metadata']
assert 'sources_used' in result['answer_metadata']
assert 'exact_matches' in result['answer_metadata']
assert 'boost_applied' in result['answer_metadata']
assert 'processing_steps' in result['answer_metadata']
assert 'conflicts_detected' in result['answer_metadata']

print("✅ All metadata fields present!")
```

## Troubleshooting

### Issue: Server not responding
**Solution:**
```powershell
# Check if server is running
Get-Process python

# Restart server
python main.py
```

### Issue: No database results
**Solution:**
```powershell
# Check database connection
python -c "from database import DatabaseManager; db = DatabaseManager(); print(db.get_all_tables())"
```

### Issue: Test failures
**Solution:**
```powershell
# Check Python environment
python --version
pip list | Select-String -Pattern "langchain|transformers|psycopg2"

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### Issue: Import errors
**Solution:**
```powershell
# Verify files exist
Test-Path processor.py
Test-Path router\hybrid.py
Test-Path models.py

# Check for syntax errors
python -m py_compile processor.py
python -m py_compile router\hybrid.py
python -m py_compile models.py
```

## Success Criteria

✅ All 5 tests pass without errors
✅ Exact match boost detected for brand queries
✅ Person queries prioritize chat/DB results
✅ Conflicts detected and resolution provided
✅ Processing steps tracked in metadata
✅ Intent correctly identified
✅ Response time < 3 seconds

## Next Steps

1. **Integration Testing:** Test with real user queries
2. **Performance Testing:** Measure response times under load
3. **User Acceptance Testing:** Collect feedback on conflict resolution
4. **Production Deployment:** Deploy to staging environment first

## Additional Resources

- **Full Documentation:** `ENHANCED_API_IMPLEMENTATION.md`
- **API Endpoint:** POST `/api/v1/query/hybrid/enhanced`
- **Test Suite:** `test_enhanced_features.py`
- **Source Code:**
  - `processor.py` - Core logic
  - `router/hybrid.py` - API endpoints
  - `models.py` - Response models
  - `database.py` - FTS implementation

## Support

For issues or questions:
1. Check `ENHANCED_API_IMPLEMENTATION.md` for detailed documentation
2. Review `test_enhanced_features.py` for usage examples
3. Check server logs: `tail -f app.log` (if logging configured)
4. Enable debug mode: Set `LOG_LEVEL=DEBUG` in environment

---

**Last Updated:** January 29, 2026
**Status:** Ready for Testing
