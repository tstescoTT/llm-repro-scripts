#!/usr/bin/env python3
"""
Concurrent LLM API Request Script
Makes configurable concurrent requests to a local LLM API and saves results to a timestamped JSON file.
Supports multiple loops of concurrent requests with configurable timeout.
Uses only Python standard library.

CLI Options:
  --concurrency: Maximum number of concurrent requests (default: 32)
  --loops: Number of times to loop over (concurrency) requests (default: 3)
  --timeout: Timeout for each request in seconds (default: no timeout)
  --skip-trace-capture: Skip the warmup trace capture request (default: false, runs 1 warmup request)
  --batch-delay: Sleep delay in seconds between request batches/loops (default: 0, no delay)
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import threading
import time
from datetime import datetime
import uuid
import sys
import os
import argparse

# Configuration
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
API_TOKEN = os.getenv("API_TOKEN")

# Generate timestamped filename at startup
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = "output"
JSON_FILENAME = os.path.join(OUTPUT_DIR, f"concurrent_llm_requests_{TIMESTAMP}.json")

# Global counter for tracking written results (initialized after imports)
results_written_count = 0
results_written_lock = None

# Sample request payload
REQUEST_PAYLOAD = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "temperature": 0.0,
    "max_tokens": 1000,
    "stream": False,
    "messages": json.load(open("messages_prompts.json")),
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "comprehensive_morphological_analysis",
        "schema": {
          "type": "object",
          "properties": {
            "comprehensive_morphological_analysis": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "morphological_term": { "type": "string" },
                  "analytical_category": { "type": "string" },
                  "taxonomic_significance": { "type": "string" },
                  "morphological_complexity": { "type": "string" },
                  "output": { "type": "string" }
                },
                "required": ["morphological_term", "analytical_category", "taxonomic_significance", "morphological_complexity"],
                "additionalProperties": False,
              }
            },
          },
          "required": ["comprehensive_morphological_analysis"],
          "additionalProperties": False,
        },
        "strict": True
      }
    }
}

class RequestResult:
    def __init__(self, request_id, success=False, response=None, error=None, processing_time=0):
        self.request_id = request_id
        self.timestamp = datetime.now().isoformat()
        self.success = success
        self.response = response
        self.error = error
        self.processing_time = processing_time

def write_result_to_incremental_json(result, json_lock):
    """Write a single result to the incremental JSON file immediately"""
    global results_written_count
    
    try:
        request_data = {
            "request_id": result.request_id,
            "timestamp": result.timestamp,
            "success": result.success,
            "processing_time": result.processing_time
        }
        
        if result.success:
            request_data["response"] = result.response
            request_data["error"] = None
        else:
            request_data["response"] = None
            request_data["error"] = result.error
        
        # Write to JSON file with thread safety
        with json_lock:
            with results_written_lock:
                # Check if this is the first result
                if results_written_count == 0:
                    # Write opening bracket and first object
                    with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
                        f.write('[\n')
                        f.write(json.dumps(request_data, ensure_ascii=False, indent=2))
                        f.flush()
                else:
                    # Append comma and next object
                    with open(JSON_FILENAME, 'a', encoding='utf-8') as f:
                        f.write(',\n')
                        f.write(json.dumps(request_data, ensure_ascii=False, indent=2))
                        f.flush()
                
                results_written_count += 1
                
    except Exception as e:
        print(f"Error writing result to incremental JSON: {e}")

def close_json_array():
    """Close the JSON array"""
    try:
        if results_written_count > 0:
            with open(JSON_FILENAME, 'a', encoding='utf-8') as f:
                f.write('\n]\n')
                f.flush()
            print(f"Closed JSON file: {JSON_FILENAME}")
    except Exception as e:
        print(f"Error closing JSON array: {e}")

def make_api_request(request_id, results_list, lock, json_lock, timeout=None):
    """Make a single API request and store the result"""
    start_time = time.time()
    
    try:
        # Prepare the request
        data = json.dumps(REQUEST_PAYLOAD).encode('utf-8')
        
        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        if API_TOKEN:
            headers['Authorization'] = f'Bearer {API_TOKEN}'
        
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers=headers
        )
        
        # Make the request with optional timeout
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            processing_time = time.time() - start_time
            
            result = RequestResult(
                request_id=request_id,
                success=True,
                response=response_data,
                processing_time=processing_time
            )
            
    except urllib.error.HTTPError as e:
        processing_time = time.time() - start_time
        error_msg = f"HTTP {e.code}: {e.reason}"
        try:
            error_body = e.read().decode('utf-8')
            error_msg += f" - {error_body}"
        except:
            pass
            
        result = RequestResult(
            request_id=request_id,
            success=False,
            error=error_msg,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        result = RequestResult(
            request_id=request_id,
            success=False,
            error=str(e),
            processing_time=processing_time
        )
    
    # Thread-safe result storage and immediate JSONL writing
    with lock:
        results_list.append(result)
        print(f"Request {request_id} completed in {result.processing_time:.2f}s - {'SUCCESS' if result.success else 'FAILED'}")
    
    # Write result to incremental JSON file immediately
    write_result_to_incremental_json(result, json_lock)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Concurrent LLM API Request Script - Makes concurrent requests to a local LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--concurrency',
        type=int,
        default=32,
        help='Maximum number of concurrent requests (default: 32)'
    )
    
    parser.add_argument(
        '--loops',
        type=int,
        default=3,
        help='Number of times to loop over (concurrency) requests being sent to server (default: 3)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=None,
        help='Set timeout for each request in seconds (e.g., 0.01 for almost immediate timeout, default: no timeout)'
    )
    
    parser.add_argument(
        '--skip-trace-capture',
        action='store_true',
        default=False,
        help='Skip the warmup trace capture request (default: false, runs 1 warmup request)'
    )
    
    parser.add_argument(
        '--batch-delay',
        type=float,
        default=0,
        help='Sleep delay in seconds between request batches/loops (default: 0, no delay)'
    )
    
    return parser.parse_args()

def main():
    global results_written_lock
    
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_requests = args.concurrency * args.loops
    print(f"Starting {total_requests} total requests ({args.concurrency} concurrent × {args.loops} loops) to {API_URL}")
    if args.timeout is not None:
        print(f"Request timeout: {args.timeout}s")
    print(f"Results will be written incrementally to: {JSON_FILENAME}")
    print("=" * 60)
    
    # Initialize global lock
    results_written_lock = threading.Lock()
    
    # Shared data structures
    results = []
    lock = threading.Lock()
    json_lock = threading.Lock()  # Separate lock for incremental JSON file writing
    
    try:
        # Run warmup trace capture request unless skipped
        if not args.skip_trace_capture:
            print("\n--- Warmup Trace Capture Request ---")
            warmup_start_time = time.time()
            
            # Create a separate results list for warmup (not included in main results)
            warmup_results = []
            warmup_request_id = str(uuid.uuid4())
            
            print(f"Starting warmup request (ID: {warmup_request_id})")
            
            # Run single warmup request
            TRACE_CAPTURE_TIMEOUT = 7200
            warmup_thread = threading.Thread(
                target=make_api_request,
                args=(warmup_request_id, warmup_results, lock, json_lock, TRACE_CAPTURE_TIMEOUT)
            )
            warmup_thread.start()
            warmup_thread.join()
            
            warmup_time = time.time() - warmup_start_time
            print(f"Warmup request completed in {warmup_time:.2f} seconds")
            
            if warmup_results and warmup_results[0].success:
                print("Warmup request successful - proceeding with main test")
            else:
                print("Warmup request failed - proceeding with main test anyway")
            
            print("=" * 60)
        
        # Start overall timing
        overall_start_time = time.time()
        
        # Loop over the specified number of loops
        for loop_num in range(args.loops):
            print(f"\n--- Loop {loop_num + 1}/{args.loops} ---")
            threads = []
            loop_start_time = time.time()
            
            # Start all threads for this loop
            for i in range(args.concurrency):
                request_id = str(uuid.uuid4())
                thread = threading.Thread(
                    target=make_api_request,
                    args=(request_id, results, lock, json_lock, args.timeout)
                )
                threads.append(thread)
                thread.start()
                print(f"Started request {i+1}/{args.concurrency} (ID: {request_id})")

            
            # Wait for all threads in this loop to complete
            for thread in threads:
                thread.join()
            
            loop_time = time.time() - loop_start_time
            print(f"Loop {loop_num + 1} completed in {loop_time:.2f} seconds")
            
            # Add batch delay if specified and not the last loop
            if args.batch_delay > 0 and loop_num < args.loops - 1:
                print(f"Sleeping for {args.batch_delay} seconds before next batch...")
                time.sleep(args.batch_delay)
        
        total_time = time.time() - overall_start_time
        
        print("=" * 60)
        print(f"All requests completed in {total_time:.2f} seconds")
        
        # Generate statistics
        successful_requests = [r for r in results if r.success]
        failed_requests = [r for r in results if not r.success]
        
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Failed requests: {len(failed_requests)}")
        
        if successful_requests:
            avg_time = sum(r.processing_time for r in successful_requests) / len(successful_requests)
            min_time = min(r.processing_time for r in successful_requests)
            max_time = max(r.processing_time for r in successful_requests)
            print(f"Average response time: {avg_time:.2f}s")
            print(f"Min response time: {min_time:.2f}s")
            print(f"Max response time: {max_time:.2f}s")
        
        print(f"Results saved to: {JSON_FILENAME}")
        
        # Print summary of failed requests if any
        if failed_requests:
            print("\nFailed requests summary:")
            for result in failed_requests:
                print(f"  {result.request_id}: {result.error}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        # Always close the incremental JSON array, even if script crashes
        close_json_array()

if __name__ == "__main__":
    sys.exit(main())
