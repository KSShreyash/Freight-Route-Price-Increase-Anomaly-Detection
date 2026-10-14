wrote 21 rows to output\run_1.csv
  flagged (unexplained): 17
  justified by a note:   4
  llm calls: 0  cache hits: 62
wrote 21 rows to output\run_2.csv
  flagged (unexplained): 17
  justified by a note:   4
  llm calls: 0  cache hits: 62
wrote 21 rows to output\run_3.csv
  flagged (unexplained): 17
  justified by a note:   4
  llm calls: 0  cache hits: 62

--- diff run 1 vs run 2 ---
identical
--- diff run 1 vs run 3 ---
identical

sha256:
02a0d5b262c662105b12571755da861c68107f8e53e1c01b6082f8420ac1db8f *output/run_1.csv
02a0d5b262c662105b12571755da861c68107f8e53e1c01b6082f8420ac1db8f *output/run_2.csv
02a0d5b262c662105b12571755da861c68107f8e53e1c01b6082f8420ac1db8f *output/run_3.csv

--- api calls per run (runs 2 and 3 must be 0) ---
run 1: llm_calls = 0  cache_hits = 62
run 2: llm_calls = 0  cache_hits = 62
run 3: llm_calls = 0  cache_hits = 62
