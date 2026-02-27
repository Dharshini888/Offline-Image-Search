#!/usr/bin/env python3
"""Test search endpoint to see if it's working"""
import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, '.')

from main import search

# Test a simple search
print("Testing search with query='dog'...")
result = search(query='dog', top_k=10)

print(f"Status: {result.get('status')}")
if result.get('status') == 'found':
    print(f"Results found: {result.get('count')}")
    for i, res in enumerate(result.get('results', [])[:3], 1):
        print(f"  {i}. {res['filename']}: score={res['score']}")
elif result.get('status') == 'error':
    print(f"Error: {result.get('message')}")
else:
    print(f"Not found: {result.get('message')}")
